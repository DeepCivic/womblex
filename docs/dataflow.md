# Data Flow

End-to-end data movement through Womblex, from raw input to Parquet output.

## Overview

The system has two categories of operation:

1. **Ingest** — format-dependent extraction or transformation. Always runs first.
2. **Operations** — independent functions. Callers compose them directly based on what they need. Each has clear preconditions (e.g. chunk needs an extraction, enrich needs chunks).

Enrichment requires an external Isaacus client.

G-NAF PSV ingest is a standalone path (`womblex ingest-gnaf`) that bypasses extraction entirely — see below.

```
Raw files (PDF / DOCX / CSV)
        │
        ▼
┌───────────────────┐
│  detect_file_type │  → DocumentProfile
│  (ingest/detect)  │    (type, signals, PaddleOCR confidence)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   get_extractor   │  → ExtractionStrategy (selected by doc_type)
│  (ingest/extract) │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  strategy.extract │  → ExtractionResult
│ (ingest/strategies│    (pages, text_blocks, tables, forms, images)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  run_redaction    │  → RedactionReport on ExtractionResult
│  (redact/stage)   │    (flag / blackout / delete mode)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ chunk_document    │  → list[TextChunk]
│ (process/chunker) │    (text, offsets, content_type, has_redaction)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ run_pii_cleaning  │  → PII spans replaced with <ENTITY_TYPE> tags
│ (pii/stage)       │    (PERSON, ADDRESS)
└───────────────────┘
        │
  ── extraction complete, caller decides what to do next ──
        │
        ▼  (caller composes operations as needed)
┌───────────────────┐
│  run_enrichment   │  → EnrichmentResult per document
│  (analyse/enrich) │    (segments, entities, relationships)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  build_document   │  → DocumentGraph
│  _graph           │    (nodes, edges, chunk-level mention links)
│  (analyse/graph)  │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  write_parquet    │  → documents.parquet, chunks.parquet
│  (store/output)   │
│  write_enrichment │  → entities.parquet, graph_edges.parquet,
│  (store/enrichment│     enrichment_meta.parquet
│  _output)         │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  run_verifications│  → VerificationResult
│  (verify/engine)  │    (structural checks + weak-signal scan)
└───────────────────┘
```

## Per-Document Flow

For each file processed via `operations.py`:

```
1. detect_file_type(path) → DocumentProfile
      ├── PDF/DOCX: text_pages, image_pages, table_signals per sampled page
      │     ├── handwriting signals: ruled lines, glyph regularity, stroke width
      │     └── PaddleOCR sampling (fallback): per-region confidence list
      └── Spreadsheet: reads first 500 rows → _classify_sheet() per sheet
            └── populates DocumentProfile.sheet_meta: list[SheetInfo]

2. get_extractor(profile) → ExtractionStrategy | SpreadsheetExtractor | DocxExtractor

3. extract_text(path, profile) → list[ExtractionResult]  (single-element list per source)
      ├── PDF/DOCX path → single-element list
      │     ├── Native pages: page.get_text("text", flags=TEXT_DEHYPHENATE)
      │     ├── Scanned pages: _ocr_page()
      │     │     ├── deskew (Hough line rotation)
      │     │     ├── binarise: OTSU if bimodal histogram, else adaptive Gaussian
      │     │     └── OCR → (text, avg_confidence, preprocessing_steps)
      │     │             └── warn if avg_confidence < 40%
      │     └── all pages: _normalise_text() — fix font encoding artefacts, strip OCR footers
      └── Spreadsheet path → one ExtractionResult per logical unit
            ├── data / glossary sheets → one result per non-empty row
            │     └── document_id = "<stem>:<key_col_value>"
            └── narrative / key_value sheets → one result per sheet

4. Operations (independent — caller composes as needed):

   redact: run_redaction(results, config) — PDF only
      ├── render each page as image at configured DPI
      ├── RedactionDetector.detect() → list[RedactionInfo] per page
      ├── store RedactionReport on ExtractionResult.redaction_report
      ├── annotate_extraction() → add warning strings
      └── apply mode:
            ├── flag: no text change (annotation only)
            ├── blackout: prepend [REDACTED] to affected page text
            └── delete: clear affected page text

   chunk: run_chunking(results, config) → list[TextChunk] per document
      ├── narrative text → semchunk (configurable token budget)
      ├── tables → markdown conversion → semchunk
      ├── each chunk tagged with content_type ("narrative" | "table")
      ├── [REDACTED] markers repaired if split across boundaries
      └── if redaction mode is "flag": annotate_chunks() sets has_redaction=True

   pii: run_pii_cleaning(results, config)
      ├── regex candidates: PERSON (title-case + honorific), ADDRESS (street-type anchor)
      ├── context validation: cosine similarity with all-MiniLM-L6-v2
      ├── at post_enrichment: merge enrichment-derived entity spans
      └── replace PII spans with <ENTITY_TYPE> tags via presidio-anonymizer

   ── caller decides what to do next ──

   enrich: run_enrichment(results, config) — requires chunks
      ├── for each document: client.enrichments.create(texts=[...])
      │     └── retry on 429 with exponential backoff
      ├── convert SDK response → EnrichmentResult (ILGS data models)
      └── build_document_graph(enrichment, chunks) → DocumentGraph
            ├── nodes: document, chunks, segments, persons, locations, terms, external docs
            └── edges: cross-references, contact info, dates, mention-to-chunk links

5. store
      ├── write_batch_parquet(batch, path) → documents.parquet
      ├── write_batch_enrichment(batch, dir) →
      │     ├── entities.parquet
      │     ├── graph_edges.parquet
      │     └── enrichment_meta.parquet
      └── checkpoint written after each batch (JSON, resumable)
```

## Data Structures

### DocumentProfile (output of detect)

| Field | Type | Description |
|-------|------|-------------|
| `doc_type` | `DocumentType` | Drives strategy selection |
| `page_count` | `int` | Total pages in document |
| `has_text_layer` | `bool` | At least one page has native text |
| `text_coverage` | `float` | Fraction of pages with native text |
| `has_images` | `bool` | At least one page has embedded images |
| `has_tables` | `bool` | Table structure detected |
| `has_handwriting_signals` | `bool` | Handwriting indicators found |
| `ocr_confidence` | `float \| None` | Average OCR confidence (0–100 scale) |
| `ocr_region_confidences` | `list[float] \| None` | Per-region PaddleOCR scores (0–1) |
| `glyph_regularity` | `float \| None` | 0–1: high = typed, low = handwritten |
| `stroke_consistency` | `float \| None` | 0–1: high = typed, low = handwritten |
| `confidence` | `float` | Classifier confidence in the detected type |
| `sheet_meta` | `list[SheetInfo] \| None` | Per-sheet classification (spreadsheets only) |

### ExtractionResult (output of extract)

`extract_text()` returns `list[ExtractionResult]`. PDFs, DOCX, and
spreadsheets each return a single-element list (one result per
source). The previous one-result-per-row spreadsheet shape was
removed — a spreadsheet's cells now live as `kind='sheet_cell'`
elements on a single result.

| Field | Type | Description |
|-------|------|-------------|
| `pages` | `list[PageResult]` | Per-page text; mutable so PII / redaction can rewrite `page.text` |
| `elements` | `list[Element]` | Canonical structural stream — what the parquet writer serialises |
| `method` | `str` | Strategy used (e.g. `native`, `scanned_machinewritten`, `spreadsheet`, `docx`) |
| `error` | `str \| None` | Error message if extraction failed |
| `metadata` | `ExtractionMetadata` | Strategy, confidence, timing, preprocessing steps |
| `warnings` | `list[str]` | Blank page warnings, redaction annotations |
| `document_id` | `str \| None` | Source identifier used as `doc_id` |
| `redaction_report` | `RedactionReport \| None` | Set by redaction stage |

Derived read-only views (compat for downstream callers that haven't
migrated to the element stream): `.text_blocks`, `.tables`, `.forms`,
`.images`. The `.tables` view also synthesises one `TableData` per
spreadsheet sheet so the chunker continues to see a unified
table-shaped surface.

### TextChunk (output of chunking)

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | Chunk text content |
| `start_char` | `int` | Unicode code-point start offset in source text |
| `end_char` | `int` | Unicode code-point end offset in source text |
| `chunk_index` | `int` | Sequential index within the document |
| `content_type` | `str` | `"narrative"` or `"table"` |
| `has_redaction` | `bool` | True if source pages contain redacted regions (flag mode) |

### Parquet Output

Each batch writes four sibling parquet shards via `store/output.py`.
See [`docs/extraction.md`](extraction.md) for the canonical schema
reference.

- `batch-NNNN.elements.parquet` — one row per element
- `batch-NNNN.table_cells.parquet` — children of `kind='table'`
- `batch-NNNN.form_fields.parquet` — children of `kind='form'`
- `batch-NNNN._manifest.parquet` — one row per source file

Sidecars join back to elements via `(source_hash, parent_elem_order)`
matching `(source_hash, elem_order)` with the corresponding kind.

**chunks.parquet** — one row per text chunk (planned; not yet written by `store/output.py`)

| Column | Description |
|--------|-------------|
| `chunk_id` | Unique identifier |
| `document_id` | FK to documents.parquet |
| `text` | Chunk text |
| `token_count` | Tokens in chunk |
| `embedding` | Float array from kanon-2-embedder |
| `classification` | Top label from kanon-universal-classifier |
| `classification_score` | Confidence for top label |

**entities.parquet** — one row per entity mention (from enrichment)

| Column | Description |
|--------|-------------|
| `document_id` | FK to documents.parquet |
| `entity_type` | Entity category (person, location, term, etc.) |
| `entity_name` | Resolved entity name |
| `mention_spans` | List of (start, end) offsets in document text |
| `chunk_indices` | Chunk indices where the entity appears |

**graph_edges.parquet** — one row per relationship (from enrichment)

| Column | Description |
|--------|-------------|
| `document_id` | FK to documents.parquet |
| `source_id` | Source node identifier |
| `target_id` | Target node identifier |
| `relation` | Relationship type |

**enrichment_meta.parquet** — one row per enriched document

| Column | Description |
|--------|-------------|
| `document_id` | FK to documents.parquet |
| `segment_count` | Number of structural segments |
| `person_count` | Number of persons identified |
| `location_count` | Number of locations identified |
| `term_count` | Number of defined terms |

## G-NAF Standalone Ingest

G-NAF PSV files bypass extraction entirely. The flow is:

```
.psv files (headerless, pipe-delimited)
        │
        ▼
┌───────────────────┐
│  discover_psv     │  → list[Path] (recursive .psv glob)
│  _files           │
└───────────────────┘
        │
        ▼  (per file)
┌───────────────────┐
│  _parse_filename  │  → (state, table_name) from filename pattern
│  (ingest/gnaf)    │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  ingest_psv       │  → Parquet file with provenance metadata
│  (ingest/gnaf)    │    (schema from gnaf_schema.py, all columns as strings)
└───────────────────┘
```

CLI: `womblex ingest-gnaf <root_dir> -o <output_dir>`

## Geospatial Standalone Ingest

SHP files bypass extraction entirely. The flow is:

```
.shp files (+ sidecar .dbf, .prj, .shx)
        │
        ▼
┌───────────────────┐
│  ingest_shapefile │  → GeoParquet file with provenance metadata
│  (ingest/         │    (geometry + CRS + attributes preserved exactly)
│   geospatial)     │
└───────────────────┘
```

CLI: `womblex ingest-geo <root_dir> -o <output_dir>`

## Batch Processing and Checkpointing

The `womblex run` CLI command processes documents in batches (default: 100). After each batch:

1. Results are appended to the Parquet files.
2. A checkpoint record is written (`CheckpointManager` in `store/checkpoint.py`) noting processed document IDs, batch number, and success/failure counts.

On resume (`--resume` flag), the CLI reads the checkpoint JSON and skips already-processed documents via `filter_unprocessed()`. Individual document errors are logged and recorded in `documents.parquet` without stopping the batch.

================================
# Future State Dataflow
================================

The remaining unimplemented data flow capabilities. Everything above this line is current state.

1. **AI/Semantic Chunking** — provider-agnostic semantic chunking mode using enrichment spans as boundary hints. Adds `analyse/boundaries.py` for span-to-boundary extraction and `chunk_document_semantic()` to `process/chunker.py`. Enabled via `chunking.semantic: true` in config. Full design in `architecture.md` § AI/Semantic Chunking.
