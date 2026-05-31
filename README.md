# Womblex

Document extraction pipeline for converting Australian government documents into ML-friendly corpus or collections. Extracts text from PDFs and Word documents (native, scanned, forms, hybrid). Spreadsheets are ingested and produce one result per logical row, ready for per-record semantic analysis.

## Design disclosure
This project is designed for everyone. All design decisions favour air-gapped edge deployment, running on limited resources. This means Womblex doesn't include many of the more robust 'all in one' OCR models.

Mature OCR models are used to compete with Womblex for evaluations and guide development.

## Add-ons/integrations
Optionally outputs are prepared for semantic analysis via [Isaacus](https://isaacus.com/).

## The Problem

Government document releases arrive as a mix of file formats:
- **PDFs** — native (selectable text), scanned (narrative, forms, tables), hybrid, or redacted
- **Word documents** (`.docx`) — paragraphs and embedded tables
- **Spreadsheets** (`.csv`, `.xlsx`, `.xls`) — row-level data, glossaries, key-value lookups, and narrative sheets

One-size-fits-all OCR fails because each format and sub-type needs a different extraction strategy. Womblex detects the document type first, then routes to the right extractor.

## Installation

```bash
pip install womblex
```

With Isaacus enrichment:

```bash
pip install womblex[isaacus]
```

For development:

```bash
git clone https://github.com/Team-DeepCivic/Womblex.git
cd Womblex
uv sync --extra dev
```

Test fixtures live in a separate repository. Clone them for running benchmarks:

```bash
git clone https://github.com/DeepCivic/womblex-development-fixtures.git fixtures
```

See [THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md) for details.

### System Dependencies

No system-level dependencies beyond Python. All extraction backends are pure Python packages:
- **PyMuPDF** (`fitz`) — native PDF text and structure
- **PaddleOCR** (`rapidocr-onnxruntime`) — scanned-page OCR with layout analysis (no Tesseract or PaddlePaddle required)
- **python-docx** — Word document extraction
- **pandas** + **openpyxl** — spreadsheet ingestion (CSV/Excel)

Once you have extraction working, semantic analysis via Isaacus (embeddings, classification, extractive QA) is straightforward.

### Isaacus API Key (optional)

Required only for the enrichment stage (`pip install womblex[isaacus]`). Text extraction works without it.

```bash
cp .env.example .env
# Edit .env and add your key from https://isaacus.com/
```

Or export directly:

```bash
export ISAACUS_API_KEY="your-key-here"
```

## Quick Start

```bash
# Process a document set using a config (E2E composition)
womblex run --config configs/example.yaml

# Resume from checkpoint after interruption
womblex run --config configs/example.yaml --resume

# Process individual files (PDF, DOCX, CSV, Excel)
womblex extract document.pdf -o output/
womblex extract report.docx -o output/
womblex extract dataset.xlsx -o output/

# Per-stage commands (primary workflow for staged corpora): each consumes the
# prior stage's shard directory and writes its own sidecar in place, with an
# independent resumable CheckpointManager.
womblex chunk  --shards output/<run_id>/documents/                  # *.chunks.parquet
womblex redact --shards output/<run_id>/documents/ --pdfs <pdf_dir> # *.redactions.parquet
womblex enrich --shards output/<run_id>/documents/                  # *.enrichment_entities.parquet (Kanon-2; needs ISAACUS_API_KEY)
womblex link   --shards output/<run_id>/documents/ --config <yaml>  # *.entity_links.parquet (register match)
womblex embed  --shards output/<run_id>/documents/                  # *.embeddings.parquet (Kanon-2 chunk embeddings)

# Audit shard integrity (extraction stage)
womblex verify-shards output/<run_id>/
```

## How It Works

### 1. Per-page profiling + plan-driven orchestrator

PDFs are profiled per-page (`PageProfile` per page) rather than at
document level. The orchestrator dispatches operations page-by-page
based on the profiles, then merges into a single `ExtractionResult`.
A doc-level summary type still surfaces in metadata.

**Per-page operations** the orchestrator can apply:

| Page profile | Operation | Notes |
|---|---|---|
| `has_text_layer` | Native text + tables + forms + blocks | Per-image OCR fires when the page has embedded image regions |
| `needs_ocr` | PaddleOCR + layout blocks + form-pair line scan | YOLO layout for blocks; line-based form-pair extraction on assembled text |
| Mixed-typed | Per-page typed/handwritten classification | Tags blocks as `typed` or `handwritten` |

**Doc-level shape detection** (informs the orchestrator):

| Shape | Detection | Specialised handling |
|---|---|---|
| Spreadsheet-print | Native text + table signal + filename hint | Custom multi-page table extractor with metadata-block capture (`ingest/spreadsheet_print.py`) |
| Hybrid | Mix of native and OCR-needed pages | Per-page dispatch picks the right operation |

**Other formats** — routed by file extension:

| Format | Extensions | Extraction Strategy |
|--------|-----------|---------------------|
| Word | `.docx` | python-docx (paragraphs + tables) |
| Spreadsheet | `.csv`, `.xlsx`, `.xls` | pandas per-row or per-sheet |

**Other formats** — routed by file extension:

| Format | Extensions | Extraction Strategy |
|--------|-----------|---------------------|
| Word | `.docx` | python-docx (paragraphs + tables) |
| Spreadsheet | `.csv`, `.xlsx`, `.xls` | pandas per-row or per-sheet |

### 2. Extraction

Each document type routes to an appropriate extractor. `extract_text()` always returns a `list[ExtractionResult]`:

- **PDFs** return a single-element list. The per-page orchestrator dispatches `_apply_native_page` or `_apply_ocr_page` based on each page's `PageProfile`. PaddleOCR returns per-region confidence scores stored in the document profile. YOLO layout analysis (DocLayNet `yolo11n_doc_layout.pt`, with COCO `yolov8n.pt` as fallback) is called on OCR pages by `_layout_blocks_and_tables` to populate `Element.kind` for the layout regions it detects; a full-page scan whose dominant region is a figure but which OCR's to substantial text is tagged `paragraph` rather than `figure` so its content reaches chunking.
- **DOCX** returns a single-element list with paragraphs and tables interleaved in OOXML body order.
- **Spreadsheets** return one `ExtractionResult` per workbook. Each sheet contributes a leading `kind='sheet_meta'` element followed by one `kind='sheet_cell'` element per non-empty cell.

Each result carries a `document_id` used as the primary key downstream.

Text at the extraction boundary is **verbatim** — `_normalise_text` no longer runs in the extraction hot path. Whatever the producing extractor (native text layer, PaddleOCR, DOCX, spreadsheet-print, …) emits is what lands on the element's `text` field. Downstream stages (PII, redaction, chunking) may rewrite `pages[i].text`, but the parquet writer serialises `elements`, so on-disk content stays extraction-time verbatim. Cleanup (font-encoding artefacts, running OCR footers) belongs to a downstream cleaning stage. See `docs/extraction.md`.

### 3. Redaction

Redaction runs as a post-extraction stage, separate from extraction. This avoids false positives that occur when running redaction detection inside OCR (form fields, chart regions, and diagram fills trigger the detector).

Redacted regions can be replaced with `<REDACTED>` markers (preserving sentence structure) or deleted entirely. The stage is configurable: apply after chunking, after enrichment, or both.

### 4. Chunking

Extracted text is split into semantically meaningful chunks using [semchunk](https://github.com/isaacus-dev/semchunk) with the Kanon tokeniser (default 480 tokens, leaving 32-token headroom for Isaacus 512-token context windows). Tables are converted to markdown and chunked separately, with each chunk tagged as `"narrative"` or `"table"`. `<REDACTED>` markers are preserved across chunk boundaries.

Chunking has two invocation modes that share one engine (`chunk_batch`):

- **Per-stage:** `womblex chunk --shards <run_dir>/documents/` consumes the extraction-stage shards directly and writes `*.chunks.parquet` siblings. Independent `CheckpointManager` so the chunk stage resumes without re-extracting. This is the primary workflow for staged corpus runs.
- **E2E composition:** `womblex run --config <yaml>` extracts and chunks in one process (kept for users with simpler corpora).

Both modes reassemble narrative + tables from each source's element stream, then feed every doc's narratives into a single semchunk call (with overlap) and every doc's table markdowns into another (no overlap), so `processes` parallelises across the whole batch. Chunks carry `(start_char, end_char, page_start, page_end, has_redaction, content_type)`; they join back to `elements` via `source_hash` plus offset-range overlap.

### 5. PII Cleaning

An optional PII cleaning stage strips personal identifiers from chunk text before output or enrichment. Operates on chunks post-chunking as an isolated pipeline stage.

Currently detects: **PERSON** (regex + cosine-similarity context validation via `all-MiniLM-L6-v2`) and **ADDRESS** (street-type anchor regex). See `docs/accuracy/PII_CLEANING.md` for measured baseline.

The `all-MiniLM-L6-v2` model is pre-bundled in `models/` and loaded from disk — no network access required at runtime.

### 6. Embeddings and Enrichment

Clean chunks feed into Isaacus models:

- **kanon-2-embedder**: Semantic embeddings for search/retrieval
- **kanon-universal-classifier**: Zero-shot document classification
- **kanon-answer-extractor**: Structured field extraction (dates, names, references)

### Graph construction

Using Isaacus outputs an entity graph can be created for further analysis.


## Configuration

Configs are YAML files defining paths, detection thresholds, and analysis settings:

```yaml
dataset:
  name: my_dataset

paths:
  input_root: ./data/raw/my_dataset
  output_root: ./data/processed/my_dataset
  checkpoint_dir: ./data/checkpoints/my_dataset

detection:
  min_text_coverage: 0.3
  form_signal_threshold: 0.5
  table_signal_threshold: 0.4

extraction:
  ocr:
    engine: paddleocr
    dpi: 200

chunking:
  tokenizer: "isaacus/kanon-2-tokenizer"
  chunk_size: 480
  enabled: true
  chunk_tables: true

processing:
  batch_size: 25
  checkpoint_every: 25
```

See `configs/example.yaml` for a complete example.

## Output

Each batch writes four sibling Parquet shards. The shard base name is the
caller's choice (e.g. `batch-0001`):

**`batch-NNNN.elements.parquet`** — one row per structural element
(paragraph, heading, table, form, image, sheet cell, …). Canonical
output.

**`batch-NNNN.table_cells.parquet`** — children of `kind='table'`
elements, one row per cell. Joins back via
`(source_hash, parent_elem_order)`.

**`batch-NNNN.form_fields.parquet`** — children of `kind='form'`
elements, one row per field. Same join key.

**`batch-NNNN._manifest.parquet`** — one row per source file with
provenance, status, and element / cell / field counts.

See [docs/extraction.md](docs/extraction.md) for the canonical schema
reference, element kinds, the reassembly query, and the verbatim-text
policy.

With `womblex[isaacus]` enrichment enabled:

**entities.parquet** — Flat entity mentions for filtering

**graph_edges.parquet** — Relationship edges for graph queries

**enrichment_meta.parquet** — Document-level enrichment metadata

## Project Structure

```
womblex/
├── configs/           # Dataset-specific configurations
├── docs/              # Architecture docs, ADRs, accuracy reports
├── fixtures/          # Test fixtures (separate repo, see THIRD_PARTY_DATA.md)
├── src/womblex/
│   ├── cli/                # CLI subpackage — per-topic modules (pipeline, redact, ingest, score, profile)
│   ├── config.py           # Pydantic config models
│   ├── operations/         # Independent operations (extract/redact/chunk/pii/enrich), one module each
│   ├── score.py            # womblex score subcommand — labels-vs-parquet CER scoring
│   ├── profile/            # womblex profile subcommand — column schema inference
│   ├── ingest/
│   │   ├── detect.py            # Doc-level type classification (non-PDF dispatch + summary type for PDFs)
│   │   ├── page_profile.py      # Per-page PageProfile + cheap qualifiers (e.g. spreadsheet-print)
│   │   ├── orchestrator.py      # Plan-driven PDF extractor — walks per-page profiles, dispatches operations
│   │   ├── elements.py          # Element model + kinds + Cell / FieldEntry / BBox (canonical)
│   │   ├── views.py             # ExtractionResult + legacy view types (TableData / FormField / TextBlock / ImageData) as read-only projections over elements
│   │   ├── extract.py           # extract_text() entry point + page-level primitives (re-exports views)
│   │   ├── forms.py             # Form-pair extraction (AcroForm + spatial + line-based for OCR)
│   │   ├── spreadsheet_print.py # Multi-page table extractor for spreadsheet-printed PDFs
│   │   ├── morphology.py        # Page-image morphology helpers (handwriting / glyph regularity)
│   │   ├── grid_projection.py   # Column-aware text reconstruction (block-aware paragraph emission)
│   │   ├── strategies.py        # Re-export shim — non-PDF extractors + legacy ImageExtractor
│   │   ├── strategies_scanned.py # OCR primitives (_ocr_page, _layout_blocks_and_tables) + ImageExtractor
│   │   ├── strategies_file.py   # Non-PDF extractors (DOCX, plain text, non-textual)
│   │   ├── interfaces/
│   │   │   └── protocols.py     # Backend protocols (OCRReader, LayoutAnalyzer, Preprocessor)
│   │   ├── paddle_ocr.py        # PaddleOCR wrapper via rapidocr-onnxruntime + YOLOLayoutAnalyzer
│   │   ├── llm_ocr.py           # Optional LLM-based OCR backend (vision models via OpenAI-compatible API)
│   │   ├── spreadsheet.py       # CSV/Excel extraction — one ExtractionResult per workbook with cells as elements
│   │   ├── gnaf.py              # G-NAF PSV → Parquet ingest (standalone)
│   │   ├── gnaf_schema.py       # G-NAF table schemas — static column definitions
│   │   ├── geospatial.py        # SHP → GeoParquet ingest (standalone)
│   │   ├── redaction.py         # Backwards-compatible re-export of redact.detector
│   │   ├── heuristics_cv2.py    # OpenCV-based detection heuristics
│   │   └── heuristics_numpy.py  # NumPy-based detection heuristics
│   ├── redact/
│   │   ├── detector.py      # CV2 raster + vector-drawing redacted region detection
│   │   ├── stage.py         # Post-extraction redaction stage (vector-first, raster fallback)
│   │   ├── batch.py         # Batch redaction: annotate_redactions_for_shards, validate_redactions_against_labels
│   │   └── utils.py         # Masking utilities
│   ├── pii/
│   │   ├── cleaner.py       # PII detection and stripping
│   │   └── stage.py         # PII cleaning pipeline stage
│   ├── process/
│   │   ├── chunker.py       # semchunk integration — chunk_batch engine + element-stream → ChunkInput helpers
│   │   └── chunk_stage.py   # chunk_shards() over a shard dir — drives `womblex chunk --shards`
│   ├── analyse/
│   │   ├── enrich.py        # Isaacus enrichment wrappers
│   │   ├── graph.py         # Entity graph construction
│   │   ├── models.py        # Enrichment data models
│   │   └── query.py         # Load enrichment graph from Parquet for PII masking
│   ├── store/
│   │   ├── output.py        # Parquet output: elements + table_cells + form_fields + manifest + chunks sidecars + integrity checks
│   │   ├── shard_audit.py   # Directory-level shard integrity + chunks-side audit + reconcile-with-checkpoint
│   │   ├── enrichment_output.py  # Enrichment-specific output
│   │   ├── retention.py     # run_id-based retention policy
│   │   └── checkpoint.py    # Per-stage CheckpointManager
│   ├── utils/
│   │   ├── metrics.py       # WER/CER accuracy metrics
│   │   ├── tabular_metrics.py # Tabular extraction accuracy (structural fidelity, data integrity)
│   │   └── models.py        # Local model path resolution (models/ dir, HF snapshot layout)
│   └── verify/
│       └── engine.py        # Two-pass extraction quality verification
└── tests/
```

## Development

```bash
# Install with dev dependencies
uv sync --all-extras

# Fetch test fixtures (separate repo, see THIRD_PARTY_DATA.md)
git clone https://github.com/DeepCivic/womblex-development-fixtures.git fixtures

# Run unit tests
uv run python -m pytest

# Run OCR and accuracy benchmarks (requires fixture images — takes ~3 min)
uv run python -m pytest tests/test_fixture_accuracy.py tests/test_womblex_collection_accuracy.py -v

# Type checking
uv run mypy src/

# Lint
uv run ruff check src/
```

Accuracy docs (`docs/accuracy/*.md`) are regenerated automatically at the end of each test run — no manual editing needed.

## License

Apache 2.0

## Acknowledgements

- [Isaacus](https://isaacus.com/) for legal AI models
- [semchunk](https://github.com/isaacus-dev/semchunk) for semantic chunking
- [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF handling
- [RapidOCR](https://github.com/RapidAI/RapidOCR) for OCR (bundles PaddleOCR v4 ONNX models, no PaddlePaddle required)
- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8 layout analysis
- [python-docx](https://python-docx.readthedocs.io/) for Word document extraction
- [pandas](https://pandas.pydata.org/) + [openpyxl](https://openpyxl.readthedocs.io/) for spreadsheet ingestion
