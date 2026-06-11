# Architecture

Womblex extracts and normalises Australian government data for semantic analysis via Isaacus. Input files are routed by format to the appropriate ingest path. Operations are independent functions with clear preconditions — callers compose them directly.

```
Input File
│
├─ Narrative (PDF/DOCX/TXT) ──► Extract Text ──► [.txt or .parquet]
├─ Tabular (CSV/XLSX) ────────► Transform Rows ──► [.parquet]
├─ Tabular (PSV/G-NAF) ──────► Standalone Ingest ──► [.parquet]
├─ Register (XML/ABN) ───────► Standalone Ingest ──► [.parquet]
└─ Geospatial (SHP) ─────────► Transform Geometry ──► [GeoParquet]
        │
        ▼  (Optional operations — any combination, caller composes directly)
        ├─ chunk    — split text into token-bounded chunks
        ├─ redact   — detect and tag/mask redacted regions
        ├─ pii      — replace PII spans with <ENTITY_TYPE> tags
        └─ enrich   — Isaacus enrichment (requires chunks + client)
```

Each operation is a standalone function. Preconditions: chunk needs an extraction, enrich needs chunks, build_graph needs enrichment. See `docs/composable-design.md` for the full dependency graph.

## Module Map

```
src/womblex/
├── ingest/
│   ├── detect.py              # Doc-level type classification + non-PDF dispatch
│   ├── page_profile.py        # Per-page PageProfile + cheap qualifiers (e.g. spreadsheet-print)
│   ├── orchestrator.py        # Plan-driven PDF extractor — walks per-page profiles, dispatches operations
│   ├── elements.py            # Element model + kinds + Cell / FieldEntry / BBox (canonical)
│   ├── views.py               # ExtractionResult + legacy view types (TableData / FormField / TextBlock / ImageData) as read-only projections over elements
│   ├── extract.py             # extract_text() entry point + page-level primitives (re-exports views)
│   ├── forms.py               # Form-pair extraction (AcroForm + spatial + line-based)
│   ├── spreadsheet_print.py   # Multi-page table extractor for spreadsheet-printed PDFs
│   ├── morphology.py          # Page-image morphology helpers (handwriting / glyph regularity)
│   ├── grid_projection.py     # Column-aware text reconstruction (block-aware paragraph emission)
│   ├── strategies.py          # Re-export shim — non-PDF extractors + legacy ImageExtractor
│   ├── strategies_scanned.py  # OCR primitives (_ocr_page, _layout_blocks_and_tables) + ImageExtractor
│   ├── strategies_file.py     # Non-PDF extractors (DOCX, plain text, non-textual)
│   ├── interfaces/
│   │   └── protocols.py       # Backend protocols: OCRReader, LayoutAnalyzer, Preprocessor
│   ├── paddle_ocr.py          # PaddleOCR wrapper via rapidocr-onnxruntime (det/rec/cls)
│   │                          # Also hosts YOLOLayoutAnalyzer for layout region detection (COCO yolov8n)
│   ├── llm_ocr.py             # Optional LLM-based OCR backend (vision-capable models via OpenAI-compatible API)
│   ├── spreadsheet.py         # CSV/Excel extraction — one ExtractionResult per workbook, cells as elements
│   ├── gnaf.py                # G-NAF PSV → Parquet ingest (standalone, bypasses NLP pipeline)
│   ├── gnaf_schema.py         # G-NAF table schemas — static column definitions
│   ├── abn_bulk.py            # ABN bulk extract XML → Parquet ingest (standalone, bypasses NLP pipeline)
│   ├── geospatial.py          # SHP → GeoParquet ingest (standalone, bypasses NLP pipeline)
│   ├── redaction.py           # Backwards-compatible re-export of redact.detector
│   ├── heuristics_cv2.py      # Image heuristics: skew, blur, table grids, contour analysis
│   └── heuristics_numpy.py    # Signal analysis: Otsu threshold bimodality
├── redact/
│   ├── detector.py        # CV2-based RedactionDetector — detect and mask redacted regions
│   ├── stage.py           # Redaction operation: vector-first detect_redactions, apply_text_redaction, annotate_*
│   ├── batch.py           # Batch redaction operations: annotate_redactions_for_shards, validate_redactions_against_labels
│   └── utils.py           # Low-level pre-OCR masking helper (not used by extractors)
├── pii/
│   ├── cleaner.py         # PERSON + ADDRESS candidate detection (regex + cosine similarity context); emits <ENTITY_TYPE> tags
│   └── stage.py           # PII cleaning operation (post_extraction / post_chunk / post_enrichment)
├── process/
│   └── chunker.py         # semchunk integration with configurable tokeniser; repairs split <REDACTED> markers
├── analyse/
│   ├── enrich.py          # Isaacus enrichment API wrapper (kanon-2-enricher)
│   ├── graph.py           # Entity graph construction from enrichment results
│   ├── models.py          # ILGS data models (Span, Segment, Person, Location, Term, etc.)
│   └── query.py           # Load enrichment graph from Parquet for PII masking and internal use
├── store/
│   ├── output.py          # Parquet output: elements + table_cells + form_fields + manifest sidecars
│   ├── enrichment_output.py # Parquet output for entity mentions, graph edges, enrichment metadata
│   └── checkpoint.py      # JSON-based checkpoint manager for resumable batch runs
├── verify/
│   └── engine.py          # Two-pass verification: structural checks + weak-signal scan
├── utils/
│   ├── models.py          # Local model path resolution (models/ dir + HF snapshot layout)
│   ├── metrics.py         # CER, WER, CER-s accuracy metrics (numpy-accelerated Levenshtein + spatial sort)
│   ├── tabular_metrics.py # Tabular extraction accuracy (structural fidelity, data integrity, key preservation)
│   └── checksum.py        # Shared streamed MD5 helper for the standalone register ingests
├── profile/               # Column schema inference (womblex profile subcommand)
├── score.py               # Labels-vs-parquet CER scoring (womblex score subcommand)
├── cli/                   # CLI subpackage — per-topic modules:
│   ├── __init__.py        # main() + ALL_COMMANDS aggregation + dispatch
│   ├── _shared.py         # Command NamedTuple, setup_logging, discover_files
│   ├── pipeline.py        # run, extract, chunk subcommands
│   ├── redact.py          # redact, annotate-redactions, validate-redactions subcommands
│   ├── ingest.py          # ingest-gnaf, ingest-geo, ingest-abn subcommands
│   ├── score.py           # score subcommand
│   └── profile.py         # profile subcommand
├── config.py              # Pydantic config models and YAML loader
└── operations/            # Independent operations (one module each) — no orchestrator, callers compose directly
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
    if table_ratio >= 80% → STRUCTURED
    elif has_tables or has_images → NATIVE_WITH_STRUCTURED
    else → NATIVE_NARRATIVE
elif 10% < text_coverage < 30% and has_text and has_images:
    → HYBRID (mixed native + scanned pages)
elif has_images:
    if handwriting_signals >= 80% → SCANNED_HANDWRITTEN
    elif has_handwriting → SCANNED_MIXED
    elif morphology_score >= 0.6 → SCANNED_MACHINEWRITTEN
    elif morphology_score < 0.35 → SCANNED_HANDWRITTEN
    elif ocr_confidence >= 70% → SCANNED_MACHINEWRITTEN (fallback)
    elif ocr_confidence < 70% → UNKNOWN
    else → SCANNED_MACHINEWRITTEN (default when no morphology/OCR signals)
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
| `TEXT` | Plain text file (passthrough) |
| `IMAGE` | Photo / diagram — flagged for review |
| `UNKNOWN` | Detection failed |

### 2. Ingest — Extraction

`extract.py` defines the `ExtractionStrategy` and `PathExtractionStrategy` protocols, shared helpers, and the `extract_text()` dispatcher. PDFs route via the per-page orchestrator (`ingest/orchestrator.py` + `ingest/page_profile.py`); the per-doc `Native*` / `Scanned*` / `Hybrid` / `Structured` strategy classes have been removed and their bodies inlined into the orchestrator's per-page operations (`_apply_native_page`, `_apply_ocr_page`). Non-PDF extractors live in `strategies_scanned.py` (OCR primitives + the legacy `ImageExtractor`) and `strategies_file.py` (DOCX, plain text); `strategies.py` re-exports for back-compat. `spreadsheet.py` handles CSV and Excel files.

`extract_text()` logs the strategy selection (`doc, type, confidence, strategy`) at INFO level, then always returns `list[ExtractionResult]`. PDF, DOCX, and spreadsheet paths each return a single-element list (one result per source file). The list shape is retained for call-site symmetry. Spreadsheet cells live as `kind='sheet_cell'` elements on the single result; `_classify_sheet` survives as a detection-time metadata helper but no longer routes extraction.

**Spreadsheet sheet classification** (`_classify_sheet` in `spreadsheet.py`) — retained as a detection-time metadata helper, but no longer routes extraction. Every workbook now emits a single `ExtractionResult` whose element stream begins with one `kind='sheet_meta'` element per sheet (dimensions, classification) followed by one `kind='sheet_cell'` element per non-empty cell.

**`SpreadsheetExtractor`** in `spreadsheet.py` was separated from `strategies.py` to keep both files under the 750-line cap. Callers import `SpreadsheetExtractor` directly from `ingest.spreadsheet`.

**Layout backend** — scanned extractors use `YOLOLayoutAnalyzer` for layout region detection. The primary checkpoint is DocLayNet `yolo11n_doc_layout.pt` (11 document classes mapped via `_YOLO_DOCLAYNET_LABEL_MAP`, e.g. `Picture` → `figure`, `Section-header` → `heading`); COCO-pretrained `yolov8n.pt` (`_YOLO_COCO_LABEL_MAP`) is retained as a fallback. Layout analysis is called from `_layout_blocks_and_tables()` in `strategies_scanned.py`. When a page's YOLO regions carry no segmented text the fallback collapses the whole page's OCR onto the dominant region's kind; if that kind is non-text (`figure`) but the OCR is substantial (≥5 words) it is promoted to `paragraph` (`_ocr_region_block_type`) so full-page scans are not dropped from chunking. Backend contracts are formalised as `@runtime_checkable` protocols in `interfaces/protocols.py` (`OCRReader`, `LayoutAnalyzer`, `Preprocessor`).

The orchestrator's OCR per-page path (`_apply_ocr_page`) drives `_ocr_page()` which:

1. Renders the page to a numpy array at the configured DPI
2. Deskews via Hough-line skew detection
3. Binarises — skipped for clean digital renders (histogram analysis detects low noise + narrow dynamic range); OTSU if bimodal histogram, adaptive Gaussian otherwise (handles binding shadows and scanner gradients)
4. Runs OCR and returns `(text, avg_confidence, preprocessing_steps)`; warns if avg confidence < 40%

**Text policy at the extraction boundary is verbatim.** `_normalise_text` no longer runs in the extraction hot path. Whatever the producing extractor (native text layer, paddle OCR, docx, xlsx, spreadsheet_print, figure_image) emits is what lands on the element's `text` field. Downstream stages (PII, redaction, chunking) may rewrite `pages[i].text` in place, but the parquet writer reads `elements`, so on-disk content remains extraction-time verbatim. See `docs/extraction.md`.

**Output schema** is a single `elements: list[Element]` stream on `ExtractionResult`, persisted to four sibling parquet files per batch (`*.elements.parquet`, `*.table_cells.parquet`, `*.form_fields.parquet`, `*._manifest.parquet`). Element kinds: `paragraph`, `heading`, `list_item`, `caption`, `header`, `footer`, `signature`, `figure`, `image`, `table`, `form`, `page_break`, `sheet_meta`, `sheet_cell`. Tables nest cells on `Element.cells` in memory and flatten to `table_cells.parquet` on disk; forms flatten the same way to `form_fields.parquet`. Legacy view properties (`result.text_blocks` / `.tables` / `.forms` / `.images`) remain on `ExtractionResult` as read-only derivations for downstream stages that have not migrated. See `docs/extraction.md` for the canonical reference.

### 3. Ingest — G-NAF (Standalone)

`ingest/gnaf.py` provides a standalone ingest path for the [G-NAF](https://data.gov.au/data/dataset/geocoded-national-address-file-g-naf) national address dataset. G-NAF is pure structured relational data distributed as headerless pipe-delimited (`.psv`) files — NLP operations (redaction, chunking, PII, enrichment) are irrelevant and bypassed entirely.

`ingest/gnaf_schema.py` provides static, versioned column definitions for all 35 G-NAF table types (16 Authority Code lookup tables + 19 Standard tables), derived from the official `GNAF_TableCreation_Scripts` SQL.

The ingest reads each PSV via `pyarrow.csv` (streamed, constant memory), applies the schema's column names, and writes one Parquet file per input PSV. Design principles:

- **Zero semantic mutation:** All columns stored as strings. No type coercion, no null inference. Empty strings remain `""`.
- **Provenance metadata:** Each Parquet file carries `gnaf.schema_version`, `gnaf.table_name`, `gnaf.state`, `gnaf.source_file`, `gnaf.row_count`, and `gnaf.source_md5` as key-value metadata.
- **Fail-fast on schema mismatch:** Column count is validated against the static schema. Unrecognised filenames or unknown table names are skipped with a warning.

CLI: `womblex ingest-gnaf <input_dir> -o <output_dir> [--no-md5]`

### 4. Ingest — Geospatial (Standalone)

`ingest/geospatial.py` provides a standalone ingest path for ESRI Shapefiles. Like G-NAF, this bypasses the NLP operations — geospatial data is structured geometry, not narrative text.

The ingest reads SHP files via `pyogrio`, validates geometry with `shapely`, and writes GeoParquet via `geopandas`. Design principles:

- **Zero semantic mutation:** All attributes preserved as-is. Geometry and CRS carried through exactly.
- **Geometry validation:** Invalid geometries are counted and logged as warnings, not silently dropped.
- **Provenance metadata:** Each GeoParquet file carries `geospatial.source_file`, `geospatial.feature_count`, `geospatial.crs`, `geospatial.geometry_type`, `geospatial.invalid_geometries`, and `geospatial.source_md5`.

CLI: `womblex ingest-geo <input_dir> -o <output_dir> [--no-md5]`

### 5. Ingest — ABN Bulk Extract (Standalone)

`ingest/abn_bulk.py` provides a standalone ingest path for the [ABN Lookup bulk extract](https://data.gov.au/data/dataset/abn-bulk-extract) — a weekly snapshot of the Australian Business Register, distributed as 20 XML files (~6 GB uncompressed, ~11M ABNs). Like G-NAF, this is a reference register: NLP operations are irrelevant and bypassed entirely.

Each file is stream-parsed (`ET.iterparse`, constant memory) and projected into two Parquet siblings:

- `<stem>.parquet` — one row per ABR record: ABN/status/dates, entity type, the main entity name or legal-entity name parts (given names kept as separate `given_name_1` / `given_name_2` columns, since a single given name may itself contain a space), state/postcode, ACN, GST.
- `<stem>_names.parquet` — one row per registered name (main/legal, business, trading, DGR fund), keyed by ABN, shaped for `link/` register consumption.

Design principles:

- **Zero semantic mutation:** All columns stored as strings. Absent optional fields become `""`, never null.
- **Provenance metadata:** Each Parquet file carries `abn.schema_version`, `abn.source_file`, `abn.source_md5`, and `abn.row_count` as key-value metadata.
- **Per-file failure isolation:** Any failure (malformed XML, read/write error) logs with the source name, removes partial output, and lets the directory ingest continue. Files whose root element is not `Transfer` are skipped with a warning.

CLI: `womblex ingest-abn <file-or-dir> -o <output_dir> [--no-md5]`

### 6. Redact — Post-Extraction Redaction

`redact/stage.py` runs as a separate operation after extraction. It renders each PDF page as an image, runs the CV2-based `RedactionDetector` to find black-box regions, and applies the configured mode:

- `flag` — sets `has_redaction=True` on affected chunks (no text change)
- `blackout` — prepends `<REDACTED>` to affected page text
- `delete` — clears affected page text entirely

The `RedactionReport` is stored on `ExtractionResult.redaction_report` for downstream stages. Non-PDF documents (spreadsheets, DOCX) are skipped — redaction detection requires a rasterisable page source.

`redact/utils.py` provides a `pre_ocr_mask()` helper for tooling that needs to mask redactions before OCR. This is not called by extraction strategies (see CLAUDE.md — redaction inside `_ocr_page()` caused false positives on form fields and diagram fills).

### 7. Process — Chunking

`chunker.py` wraps [semchunk](https://github.com/isaacus-dev/semchunk) v4 with full parameter exposure. Chunk size defaults to 480 tokens — sized to fit Isaacus classifier and extractor context windows (512 tokens) with 32-token headroom. Uses semchunk's native offset tracking for reliable `(start_char, end_char)` provenance.

**AI chunking + single-enrichment reuse.** Setting `chunking.chunking_model` switches the narrative path to semchunk 4's AI chunking — boundaries follow the Kanon-2 enricher's structure spans instead of the token/recursive split (opt-in, off by default). To avoid enriching the same text twice when the `enrich` stage also runs, the enrich stage persists the raw ILGS Document to `*.enrichment_doc.parquet` and `chunk_batch` reuses it per `source_hash` via `narrative_overrides`, gated by a byte-identity check (`Document.text == reassembled narrative`); on mismatch or absence the doc self-enriches. Run `enrich` before `chunk`. See `docs/decisions.md`.

The `chunk_document()` entry point:
1. Chunks narrative text with native offset tracking (no `text.find()` heuristics)
2. Converts `TableData` objects to markdown tables and chunks separately (no overlap on tables)
3. Tags each chunk with a `content_type` (`"narrative"` or `"table"`) and `has_redaction` flag
4. Repairs `<REDACTED>` markers that were split across chunk boundaries (safe with overlap)

Configurable via `config.chunking`: `overlap` (token or proportional), `memoize`, `max_token_chars`, `processes` (default 1 for Chromebook deployment).

When redaction mode is `flag`, the chunking stage calls `annotate_chunks()` to propagate `has_redaction=True` from the `RedactionReport` to affected chunks.

Chunking is gated by `config.chunking.enabled` and table handling by `config.chunking.chunk_tables`.

### 8. PII — Personal Information Cleaning

PII is **graph-driven**. `pii/cleaner.py` takes its primary candidates from the Kanon-2 enrichment graph — PII-typed entities (`natural`→PERSON, `address`→ADDRESS) mapped onto chunks via mention offsets — so the stage runs *after* enrichment. There is no separate primary detector; recall is flexed by enrichment granularity, not by a second pass. The per-stage entry point is `pii/pii_stage.py` (`pii_shards()` over a shard dir, drives `womblex pii --shards`); `pii/stage.py` holds the in-memory helpers for the E2E `run` path at configurable points (`post_extraction`, `post_chunk`, `post_enrichment`).

A local regex + cosine-context detector remains as an **opt-in backstop** (`pii.use_regex_backstop`, default **off**): title-case and honorific regex for PERSON validated against reference contexts via cosine similarity with `all-MiniLM-L6-v2` (threshold 0.35, calibrated on Australian government docs; the regex uses `[^\S\n]+` as the word boundary to prevent multi-line capture), plus a street-type anchor regex for ADDRESS. It is ~15% precision on this corpus (orgs/headings get tagged PERSON), so it is reserved for recall experiments.

Masking is **terminal** — it never rewrites the raw chunks that feed Isaacus. The stage writes two siblings: `*.pii_spans.parquet` (one row per span, audit/reversible, carrying the graph `entity_id`) and `*.clean_text.parquet` (the masked publishable layer, `<PERSON_1>` / `<ADDRESS_1>` typed and numbered off the graph entity, written by default). Current coverage: PERSON and ADDRESS. See `docs/accuracy/PII_CLEANING.md` for the measured baseline.

### 9. Analyse — Enrichment

Wrappers in `analyse/` call the Isaacus SDK:

- `enrich.py` — calls `kanon-2-enricher` to produce structured ILGS Documents containing segments, entities, and relationships. Handles 429 rate-limit errors with exponential backoff.
- `graph.py` — builds a `DocumentGraph` from enrichment results, mapping entities (persons, locations, terms, external documents) to graph nodes and relationships (cross-references, contact info, dates) to edges. Chunk-level mention links are computed from span offsets.
- `models.py` — ILGS data models: `Span`, `Segment`, `Person`, `Location`, `Term`, `ExternalDocument`, `Quote`, `DateInfo`, `CrossReference`, `EnrichmentResult`, and contact info types.

### 10. Store — Output

`store/output.py` writes four sibling parquet files per batch — `batch-NNNN.elements.parquet`, `batch-NNNN.table_cells.parquet`, `batch-NNNN.form_fields.parquet`, `batch-NNNN._manifest.parquet`. Downstream stages add their own per-batch sidecars over the same shard dir, each via a `womblex <stage> --shards` command and joinable on `source_hash`: `*.chunks.parquet` (chunk, I2), `*.redactions.parquet` (redact, I3), `*.enrichment_entities.parquet` + `*.enrichment_meta.parquet` (enrich, I7), `*.entity_links.parquet` (link, I7), `*.embeddings.parquet` (embed, I7). `store/enrichment_output.py` also has a legacy E2E writer that emits three Parquet files from enrichment results:

- `entities.parquet` — entity type, name, mentions, chunk mapping
- `graph_edges.parquet` — source/target node IDs, relation type, metadata
- `enrichment_meta.parquet` — per-document enrichment summary (segment count, entity counts, etc.)

`store/checkpoint.py` provides `CheckpointManager` for resumable batch runs. Checkpoints are JSON files recording processed document IDs and batch metadata. On resume, already-processed documents are skipped.

### 11. Verify — Quality Checks

`verify/engine.py` runs two-pass verification on the output Parquet:

1. **Structural** — schema validation (required columns present), uniqueness (no duplicate `document_id`), type constraints (confidence in [0,1], non-negative page counts).
2. **Weak-signal scan** — flags documents with low confidence, page count anomalies, garbled text (high non-alphanumeric ratio), or garbled redaction patterns.

Results are classified as `passed`, `warning`, or `failed` based on the ratio of flagged documents.

## Key Design Decisions

**Detection first.** Strategy selection is driven entirely by the document profile. No extraction logic lives in detection code.

**Redaction is a post-extraction concern.** Physical black-box masking inside the extractor caused the redaction detector to misfire on form fields, chart regions, and diagram fills — suppressing text it should keep. Redaction now runs as a separate operation after extraction, using `redact/stage.py`. The `RedactionReport` is stored as a proper field on `ExtractionResult` for type-safe downstream access.

**Config-driven, not hardcoded.** Dataset-specific paths, thresholds, and hypotheses live in YAML. Core modules have no knowledge of specific datasets.

**PaddleOCR via rapidocr-onnxruntime.** The `rapidocr-onnxruntime` package bundles pre-exported PaddleOCR v4 ONNX models (~15 MB wheel) — no PaddlePaddle or PyTorch framework, no separate model download. Layout analysis uses YOLOv8 (`ultralytics` + bundled `yolov8n.pt`).

**Local model resolution.** `utils/models.py` provides `resolve_local_model_path(name)` which checks a `models/` directory (sibling of `src/`) before falling back to runtime downloads. Handles the HuggingFace hub snapshot layout (`refs/main` → `snapshots/<hash>/`) and bare files (`.pt`). Override location with `WOMBLEX_MODELS_DIR`. Models loaded lazily — no import cost unless the stage actually runs.

**No external Levenshtein dependency.** `utils/metrics.py` provides CER, WER, and CER-s (spatially-sorted CER) using a numpy-accelerated Levenshtein implementation. Short strings (≤500 chars) use a pure-Python DP loop; longer strings use numpy vectorised row operations. `spatial_sort_text()` reorders words by bounding-box centroid to isolate recognition errors from reading-order errors. No rapidfuzz or other C-extension dependency.

**PII cleaning is graph-driven, masked after Isaacus.** `pii/cleaner.py` takes its primary candidates from the Kanon-2 enrichment graph (PII-typed entities mapped onto chunks by mention offset); the regex + cosine-context detector (`all-MiniLM-L6-v2`) is an opt-in backstop (`pii.use_regex_backstop`, default off, ~15% precision). Masking is terminal — `*.clean_text.parquet` (`<PERSON_n>`) is written *after* enrich/embed and never rewrites the raw chunks. Current coverage: PERSON and ADDRESS — ORGANISATION, URL, phone, and email are not yet detected. See `docs/accuracy/PII_CLEANING.md` for measured recall/precision baseline.

**750-line hard cap per file.** Signals the need to split before files become unwieldy. The PDF dispatcher (`orchestrator.py` + `page_profile.py` + `extract.py`) and the non-PDF strategy modules (`strategies_scanned.py`, `strategies_file.py`, with `strategies.py` as a re-export shim) are split this way; `SpreadsheetExtractor` lives in `spreadsheet.py`; the CLI is split into a `cli/` subpackage of per-topic modules for the same reason.

**Niche formats get standalone submodules.** Formats with their own structure (e.g. G-NAF's headerless PSV with SQL-defined schemas, ESRI Shapefiles with geometry + CRS) get a dedicated submodule under `ingest/` that reads the format and writes Parquet/GeoParquet directly, bypassing the generic extraction operations. Dependencies (`pyogrio`, `geopandas`, `shapely`) are lazy-imported so they don't affect core pipeline users.

For evaluation metrics and accuracy validation strategy, see [`docs/evaluation.md`](evaluation.md).

================================
# Future State
================================

The remaining unimplemented capabilities. Everything above this line is current state.

**Remaining TODOs:**

1. **AI/Semantic Chunking:** ✅ **Shipped 2026-06 via a different mechanism.**
   Rather than the homegrown boundary-hints layer proposed below, this is now
   delivered by semchunk 4's native AI chunking (`chunking.chunking_model`) plus
   single-enrichment reuse — the enrich stage persists the raw ILGS Document to
   `*.enrichment_doc.parquet` and the chunk stage reuses it (byte-identity
   guarded). See `docs/decisions.md` ("AI chunking — single-enrichment graph
   reuse") and the `### 7. Process — Chunking` section above. The proposed design
   below is retained for historical rationale and is **superseded**; only the
   *Local Enrichment Fallback* (item 2) remains genuine future work.

### AI/Semantic Chunking — Proposed Design (SUPERSEDED — see note above)

#### Problem

The current chunker (`process/chunker.py`) delegates entirely to semchunk, which splits on punctuation and whitespace heuristics. This works well for generic text but ignores document structure that enrichment has already identified — segment boundaries, entity spans, cross-references. Chunks can split mid-paragraph, mid-entity, or across structural boundaries that a human reader would never break.

The TODO called for a provider-agnostic design. The key insight: enrichment spans (`Span(start, end)`) are already provider-agnostic — they're just character offsets. The coupling risk is in *how we obtain* those spans, not in how we consume them.

#### Design Principle: Boundary Hints, Not a New Chunker

Semantic chunking is not a replacement for semchunk. It's a pre-processing layer that identifies preferred split points and no-split zones, then feeds constrained text regions to the existing algorithmic chunker. This keeps semchunk's token-counting, overlap, and offset-tracking logic intact.

```
                          ┌─────────────────────┐
                          │  EnrichmentResult    │
                          │  (from any provider) │
                          └──────────┬──────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │  extract_boundaries  │  → list[SemanticBoundary]
                          │  (analyse/boundaries)│    (preferred splits + no-split zones)
                          └──────────┬──────────┘
                                     │
                                     ▼
┌──────────┐   ┌─────────────────────────────────────┐   ┌──────────────┐
│ full_text │──►│  chunk_document_semantic             │──►│ list[TextChunk]│
└──────────┘   │  (process/chunker.py)                │   └──────────────┘
               │  1. slice text at preferred splits   │
               │  2. sub-chunk each slice via semchunk│
               │  3. repair redaction markers          │
               └──────────────────────────────────────┘
```

#### Boundary Extraction — Provider-Agnostic

A new module `analyse/boundaries.py` converts an `EnrichmentResult` into a flat list of `SemanticBoundary` objects. This is the only module that reads enrichment structure — the chunker never touches `EnrichmentResult` directly.

```python
@dataclass
class SemanticBoundary:
    """A structurally significant point or zone in the document text."""
    offset: int              # character offset in full_text
    kind: str                # "split" | "no_split"
    weight: float            # 0.0–1.0, higher = stronger signal
    source: str              # "segment" | "heading" | "entity" | "crossref"
```

Boundary extraction rules (applied in priority order):

| Source | Kind | Weight | Rationale |
|--------|------|--------|-----------|
| Segment boundaries (level ≤ 2) | split | 1.0 | Chapter/section breaks are natural chunk boundaries |
| Segment boundaries (level > 2) | split | 0.7 | Sub-section breaks — prefer but don't force |
| Heading spans | split | 0.9 | Keep headings at the start of a chunk, not the end |
| Entity mention spans | no_split | 0.8 | Don't split a person name or location across chunks |
| Cross-reference spans | no_split | 0.6 | Keep internal references intact within a chunk |
| Term definition spans | no_split | 0.5 | Keep defined terms with their meaning |

The `extract_boundaries()` function takes an `EnrichmentResult` and returns `list[SemanticBoundary]`. It has no knowledge of chunk sizes, tokenisers, or the chunking algorithm — it only reads spans.

Because `EnrichmentResult` and `Span` are already Womblex's own data models (defined in `analyse/models.py`), any provider that populates these models works. Isaacus does this today via `enrich.py`. A future local model, a different API, or even hand-annotated spans would work identically — the boundary extractor doesn't care where the spans came from.

#### Chunking Algorithm

`chunk_document_semantic()` in `process/chunker.py`:

1. Collect all `split` boundaries, sorted by offset, filtered to `weight >= min_split_weight` (configurable, default 0.7).
2. Slice `full_text` at split points into *regions*. Each region is a contiguous block of text between two structural boundaries.
3. For each region:
   - If the region fits within `chunk_size` tokens → emit as a single `TextChunk`.
   - If the region exceeds `chunk_size` → sub-chunk via the existing `chunk_text()` (semchunk), but with `no_split` zones passed as protected spans. Semchunk handles the actual token-counting and splitting; protected spans are enforced by pre-inserting zero-width markers that semchunk won't split on (or by post-merge if a protected span was split).
4. Re-index all chunks sequentially.
5. Run `_repair_redaction_splits()` as today.

This means the semantic mode produces the same `list[TextChunk]` output with the same offset tracking — downstream operations (PII, enrichment, graph, store) are completely unaffected.

#### Fallback Behaviour

- If no `EnrichmentResult` is available → fall back to `chunk_document()` (current algorithmic mode). No error, no warning beyond a debug log.
- If enrichment produced zero segments (e.g. very short document) → same fallback.
- If a region between split points is empty after whitespace stripping → skip it.

#### Reusing Existing Enrichment

The typical Womblex flow is: extract → chunk → enrich → graph. Semantic chunking inverts the dependency: it needs enrichment *before* chunking. Two paths handle this:

1. **Pre-enrichment mode (new):** `run_chunking()` in `operations.py` checks if `config.chunking.semantic` is enabled. If so, it calls `enrich_document()` on the full text *before* chunking, extracts boundaries, then chunks semantically. The enrichment result is stored on `DocumentResult` so downstream `run_enrichment()` can skip re-enrichment (idempotent — same text, same result).

2. **Cached enrichment mode:** If `DocumentResult.enrichment` is already populated (e.g. from a previous run loaded via `query.py`), `run_chunking()` uses it directly. No API call needed.

This keeps the composable design intact — callers still compose operations directly, and the enrichment dependency is satisfied transparently within `run_chunking()` when semantic mode is active.

#### Configuration

Extend `ChunkingConfig` in `config.py`:

```python
class ChunkingConfig(BaseModel):
    # ... existing fields ...
    semantic: bool = Field(
        default=False,
        description="Use enrichment spans for semantic boundary detection before chunking.",
    )
    semantic_min_split_weight: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Minimum boundary weight to trigger a split in semantic mode.",
    )
```

YAML usage:

```yaml
chunking:
  tokenizer: "isaacus/kanon-2-tokenizer"
  chunk_size: 480
  semantic: true
  semantic_min_split_weight: 0.7
```

When `semantic: false` (default), behaviour is identical to today. No new dependencies, no new API calls.

#### Provider Abstraction — Not an Interface, Just Data

The design deliberately avoids a formal `EnrichmentProvider` interface or plugin system. The project convention is "no premature abstractions" (CLAUDE.md). Instead:

- `EnrichmentResult` is the contract. Any code that populates an `EnrichmentResult` with segments and entity spans is a valid provider.
- `enrich.py` does this for Isaacus today. A future local model would have its own `enrich_local.py` that returns the same `EnrichmentResult`.
- `extract_boundaries()` consumes `EnrichmentResult` — it never imports `isaacus`, never calls an API, never knows which provider was used.
- `run_chunking()` in `operations.py` calls whichever enrichment function the config points to. Today that's `enrich_document()` from `enrich.py`. Swapping providers means changing one function call, not implementing an interface.

This is provider-agnostic through data, not through abstraction.

#### Provider Quality Spectrum

Different providers populate `EnrichmentResult` with varying richness. The boundary extractor works with whatever it gets — fewer spans means fewer boundary hints, which means more reliance on semchunk's punctuation heuristics within each region. This is a graceful degradation, not a failure.

| Provider | Segments | Entities | Cross-refs | Expected boundary quality |
|----------|----------|----------|------------|--------------------------|
| Isaacus (kanon-2-enricher) | Full structural hierarchy (chapter → paragraph) | Persons, locations, terms, external docs | Yes | High — rich split points at every structural level, entity-aware no-split zones |
| Sentence-transformers / local NER | None (or synthetic via topic segmentation) | Named entities only (PER, LOC, ORG) | No | Moderate — entity no-split zones work, but split points fall back to semchunk heuristics between entity clusters |
| spaCy (en_core_web_trf or similar) | Sentence boundaries only | Named entities (PER, LOC, ORG, etc.) | No | Moderate — sentence boundaries as split hints, entity spans as no-split zones |
| No enrichment available | — | — | — | Baseline — pure semchunk, identical to current behaviour |

A lightweight local model (sentence-transformers, spaCy) is a valid provider that produces usable results out of the box. The output quality is lower than Isaacus because the boundary signals are coarser — you get entity protection but not structural segmentation. The tradeoff is: no API dependency, no cost, runs offline, at the expense of less structurally aware chunk boundaries.

#### New Files

| File | Purpose | Lines (est.) |
|------|---------|-------------|
| `analyse/boundaries.py` | `SemanticBoundary` dataclass + `extract_boundaries(EnrichmentResult) → list[SemanticBoundary]` | ~120 |

No new files for the chunker — `chunk_document_semantic()` is added to the existing `process/chunker.py`.

#### Composition Changes

The composable-design dependency graph gains one new valid composition:

```
extract(pdf) → chunk(semantic=true) → done
  └── internally: enrich(full_text) → extract_boundaries → chunk_semantic
```

And the existing enrichment composition remains valid (enrichment is not duplicated):

```
extract(pdf) → chunk(semantic=true) → enrich → build_graph → done
  └── chunk reuses the enrichment it already obtained
```

Invalid compositions remain the same — semantic chunking still requires an extraction, and enrichment still requires chunks.

#### What This Does Not Do

- No new provider interface or plugin system. The abstraction is the `EnrichmentResult` dataclass.
- No changes to `TextChunk`, `chunk_text()`, or `chunk_document()`. The algorithmic path is untouched.
- No new dependencies. Semantic chunking uses the same Isaacus client (or whatever populates `EnrichmentResult`).
- No changes to downstream operations. PII, graph, store, verify all consume `list[TextChunk]` as before.

2. **Local Enrichment Fallback:** The PII cleaner's `post_enrichment` mode and the semantic chunking design both depend on `EnrichmentResult`, which today only comes from Isaacus (`enrich.py`). Without an Isaacus client there are no graph spans, so PII has only the opt-in regex backstop (off by default) and semantic chunking falls back to pure semchunk. A local enrichment provider (e.g. spaCy `en_core_web_trf`, a fine-tuned NER model, or sentence-transformers topic segmentation) that populates `EnrichmentResult` with entity mentions and optionally segments would give both systems something to work with offline — lower quality than Isaacus, but better than regex-only / no boundaries. The provider quality spectrum in the semantic chunking design above applies here too. Implementation: a new `analyse/enrich_local.py` returning `EnrichmentResult`, selected by config (e.g. `enrichment.provider: local`), no changes to downstream consumers.
