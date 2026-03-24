# Architecture

Womblex extracts text from Australian government PDF documents and prepares it for semantic analysis via Isaacus. The codebase is organised into four stages: **Ingest → Process → Analyse → Store**.

## Module Map

```
src/womblex/
├── ingest/
│   ├── detect.py          # Document type detection — drives strategy selection
│   ├── extract.py         # ExtractionResult schema, extract_text() dispatcher
│   ├── strategies.py      # PDF/DOCX extractor implementations (one class per type)
│   ├── paddle_ocr.py      # PaddleOCR wrapper via rapidocr-onnxruntime (det/rec/cls/layout/table)
│   │                      # Also hosts YOLOLayoutAnalyzer for layout region detection
│   ├── spreadsheet.py     # CSV/Excel extraction — one ExtractionResult per row/sheet
│   ├── redaction.py       # Backwards-compatible re-export of redact.detector
│   ├── heuristics_cv2.py  # Image heuristics: skew, blur, table grids, contour analysis
│   └── heuristics_numpy.py # Signal analysis: Otsu threshold bimodality
├── redact/
│   ├── detector.py        # CV2-based RedactionDetector — detect and mask redacted regions
│   └── stage.py           # Pipeline stage logic: annotate_chunks, annotate_extraction
├── pii/
│   ├── cleaner.py         # PERSON candidate detection (regex + cosine similarity context)
│   └── stage.py           # PII cleaning pipeline stage
├── process/
│   └── chunker.py         # semchunk integration with Kanon tokeniser
├── analyse/
│   └── *.py               # Thin wrappers over the Isaacus SDK
├── store/
│   └── output.py          # Parquet output and checkpointing
├── utils/
│   └── models.py          # Local model path resolution (models/ dir + HF snapshot layout)
├── config.py              # Pydantic config models and YAML loader
└── pipeline.py            # Orchestration — ties ingest → process → analyse → store
```

## Stage Detail

### 1. Ingest — Detection

`detect.py` profiles each document before any text extraction occurs. Detection is signal-based: it examines the text layer, embedded images, table structures, and image morphology to assign a `DocumentType`.

**Detection signals, in priority order:**

| Signal | Method | Drives |
|--------|--------|--------|
| Text layer coverage | `page.get_text()` length per page | Native vs scanned split |
| Table coverage | Regex on text + `page.find_tables()`, per-page count | STRUCTURED (≥80%) or structured content flag |
| Image presence | `page.get_images()` | Scanned/hybrid flag |
| Ruled lines | Morphological horizontal line detection | Handwriting signal |
| Glyph regularity | Connected-component height variance | Typed vs handwritten |
| Stroke width variance | Skeleton distance-transform CV | Typed vs handwritten |
| OCR confidence | Per-region confidence scores (0–1) | Typed vs handwritten fallback |

PaddleOCR is only invoked as a fallback when morphological signals (glyph regularity + stroke width) are both inconclusive. Confidence scores per text region are stored in `DocumentProfile.ocr_region_confidences`.

**Classification logic:**

```
if file is .docx → DOCX
if file is .csv/.xlsx → SPREADSHEET
if text_coverage >= 30%:
    if has_tables or has_images → NATIVE_WITH_STRUCTURED
    else → NATIVE_NARRATIVE
elif 10% < text_coverage < 30%:
    → HYBRID (mixed native + scanned pages)
elif has_images:
    if handwriting_signals >= 80% → SCANNED_HANDWRITTEN
    elif has_handwriting → SCANNED_MIXED
    elif morphology_score >= 0.6 → SCANNED_MACHINEWRITTEN
    elif morphology_score < 0.35 → SCANNED_HANDWRITTEN
    elif ocr_confidence >= 70% → SCANNED_MACHINEWRITTEN (fallback)
    else → UNKNOWN
else:
    → UNKNOWN
```

Defensive classification: uncertain documents route to `UNKNOWN` rather than a wrong bucket. High `UNKNOWN` count signals detection gaps to address. See `heuristics_disambiguation.md` for the full function-level reference of CV2 and NumPy heuristics.

**Document types:**

| Type | Meaning |
|------|---------|
| `NATIVE_NARRATIVE` | PDF with selectable text layer, no structure |
| `NATIVE_WITH_STRUCTURED` | PDF with text layer plus tables or images |
| `SCANNED_MACHINEWRITTEN` | Image-only, typed/printed content |
| `SCANNED_HANDWRITTEN` | Image-only, handwritten content |
| `SCANNED_MIXED` | Image-only, mixed typed and handwritten |
| `HYBRID` | Some pages native, some scanned |
| `STRUCTURED` | Pure tabular content |
| `DOCX` | Word document |
| `SPREADSHEET` | CSV or Excel |
| `IMAGE` | Photo / diagram — flagged for review |
| `UNKNOWN` | Detection failed |

### 2. Ingest — Extraction

`extract.py` defines the `ExtractionStrategy` protocol, shared helpers, and the `extract_text()` dispatcher. `strategies.py` implements one extractor class per PDF/DOCX document type. `spreadsheet.py` handles CSV and Excel files.

`extract_text()` logs the strategy selection (`doc, type, confidence, strategy`) at INFO level, then always returns `list[ExtractionResult]`. PDF and DOCX paths return a single-element list. The spreadsheet path returns one result per logical row (for `data` and `glossary` sheets) or one result per sheet (for `narrative` and `key_value` sheets). Each spreadsheet result carries a `document_id` built from the filename and key-column value (e.g. `filename:PR-00006191`).

**Spreadsheet sheet classification** (`_classify_sheet` in `spreadsheet.py`):

| Sheet type | Detection heuristic | Extraction unit |
|------------|---------------------|-----------------|
| `data` | ≥3 columns or short cells | One result per row |
| `glossary` | 2 columns, 50–500 rows | One result per row |
| `key_value` | 2 columns, < 50 rows | One result per sheet |
| `narrative` | 1 column or very long cells | One result per sheet |

**`SpreadsheetExtractor`** in `spreadsheet.py` was separated from `strategies.py` to keep both files under the 750-line cap. `strategies.py` re-exports `SpreadsheetExtractor` for callers that import from there.

**Layout backend** — scanned extractors use `YOLOLayoutAnalyzer` (COCO-pretrained yolov8n) for layout region detection. COCO class names are mapped to document block types via `_YOLO_COCO_LABEL_MAP` (e.g. `dining table` → `table`, `person`/`book` → `paragraph`, screen objects → `figure`). Layout analysis is called from `_layout_blocks_and_tables()` in `strategies.py`.

Scanned PDF strategies (`ScannedMachinewrittenExtractor`, `ScannedHandwrittenExtractor`, `ScannedMixedExtractor`, `HybridExtractor`) all call `_ocr_page()` which:

1. Renders the page to a numpy array at the configured DPI
2. Deskews via Hough-line skew detection
3. Binarises — skipped for clean digital renders (histogram analysis detects low noise + narrow dynamic range); OTSU if bimodal histogram, adaptive Gaussian otherwise (handles binding shadows and scanner gradients)
4. Runs OCR and returns `(text, avg_confidence, preprocessing_steps)`; warns if avg confidence < 40%

After extraction, `extract_text()` runs `_normalise_text()` over every page to fix known document artefacts (broken font encoding, OCR footer noise) before returning.

**Output schema per page:**

```python
PageResult(page_number, text, method)        # one per page
TextBlock(text, position, block_type, confidence)  # semantic blocks
TableData(headers, rows, position, confidence)     # structured tables
FormField(field_name, value, position, confidence) # form widgets
ImageData(alt_text, position, confidence)          # image metadata
```

### 3. Process — Chunking

`chunker.py` wraps [semchunk](https://github.com/isaacus-dev/semchunk) with the `isaacus/kanon-2-tokenizer`. Chunk size defaults to 480 tokens — sized to fit Isaacus classifier and extractor context windows (512 tokens) with 32-token headroom.

The `chunk_document()` entry point:
1. Chunks narrative text with Unicode code-point offset tracking
2. Converts `TableData` objects to markdown tables and chunks separately
3. Tags each chunk with a `content_type` (`"narrative"` or `"table"`)
4. Repairs `[REDACTED]` markers that were split across chunk boundaries

Chunking is gated by `config.chunking.enabled` and table handling by `config.chunking.chunk_tables`.

### 4. Analyse

Thin wrappers in `analyse/` call the Isaacus SDK for:

- **Embeddings** (`kanon-2-embedder`): task type must match — `retrieval/document` for indexed content, `retrieval/query` for search.
- **Classification** (`kanon-universal-classifier`): zero-shot against config-defined hypotheses.
- **Extraction** (`kanon-answer-extractor`): structured field extraction.

### 5. Store

`output.py` writes three Parquet files: `documents.parquet`, `chunks.parquet`, `extractions.parquet`. Checkpointing writes after each batch so interrupted jobs resume without reprocessing.

## Key Design Decisions

**Detection first.** Strategy selection is driven entirely by the document profile. No extraction logic lives in detection code.

**Redaction is a post-extraction concern.** Physical black-box masking inside the extractor caused the redaction detector to misfire on form fields, chart regions, and diagram fills — suppressing text it should keep. Redaction now runs as a separate pipeline stage after extraction, using `redact/stage.py`.

**Config-driven, not hardcoded.** Dataset-specific paths, thresholds, and hypotheses live in YAML. Core modules have no knowledge of specific datasets.

**PaddleOCR via rapidocr-onnxruntime.** The `rapidocr-onnxruntime` package bundles pre-exported PaddleOCR v4 ONNX models (~15 MB wheel) — no PaddlePaddle or PyTorch framework, no separate model download. Layout analysis uses YOLOv8 (`ultralytics` + bundled `yolov8n.pt`). Table recognition uses `rapid-table` (SLANet, model downloads on first use; degrades gracefully in air-gapped environments).

**Local model resolution.** `utils/models.py` provides `resolve_local_model_path(name)` which checks a `models/` directory (sibling of `src/`) before falling back to runtime downloads. Handles the HuggingFace hub snapshot layout (`refs/main` → `snapshots/<hash>/`) and bare files (`.pt`). Override location with `WOMBLEX_MODELS_DIR`. Models loaded lazily — no import cost unless the stage actually runs.

**PII cleaning is context-validated regex.** `pii/cleaner.py` uses title-case and honorific regex patterns to find PERSON candidates, then validates each against reference contexts using cosine similarity with `all-MiniLM-L6-v2`. The regex uses `[^\S\n]+` (non-newline whitespace) as the word-boundary separator to prevent multi-line span capture. Threshold 0.35 (empirically calibrated on Australian government docs). Current coverage: PERSON only — ORGANISATION, URL, phone, and email are not yet detected. See `docs/accuracy/PII_CLEANING.md` for measured recall/precision baseline.

**750-line hard cap per file.** Signals the need to split before files become unwieldy. `extract.py` delegates strategy implementations to `strategies.py`, and `SpreadsheetExtractor` lives in `spreadsheet.py`, for this reason.
