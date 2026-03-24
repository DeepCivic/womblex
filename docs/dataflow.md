# Data Flow

End-to-end data movement through the Womblex pipeline, from raw PDF to Parquet output.

## Pipeline Overview

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
│ chunk_document    │  → list[TextChunk]
│ (process/chunker) │    (text, offsets, content_type)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  isaacus API      │  → embeddings / labels / extractions
│  (analyse/*.py)   │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  write_parquet    │  → documents.parquet
│  (store/output)   │     chunks.parquet
└───────────────────┘     extractions.parquet
```

## Per-Document Flow

For each file processed by `pipeline.py`:

```
1. detect_file_type(path) → DocumentProfile
      ├── PDF/DOCX: text_pages, image_pages, table_signals per sampled page
      │     ├── handwriting signals: ruled lines, glyph regularity, stroke width
      │     └── PaddleOCR sampling (fallback): per-region confidence list
      └── Spreadsheet: reads first 500 rows → _classify_sheet() per sheet
            └── populates DocumentProfile.sheet_meta: list[SheetInfo]

2. get_extractor(profile) → ExtractionStrategy

3. extract_text(path, profile) → list[ExtractionResult]
      ├── PDF/DOCX path → single-element list
      │     ├── Native pages: page.get_text("text", flags=TEXT_DEHYPHENATE)
      │     ├── Scanned pages: _ocr_page()
      │     │     ├── detect + mask redactions (CV2 heuristics)
      │     │     ├── deskew (Hough line rotation)
      │     │     ├── binarise: OTSU if bimodal histogram, else adaptive Gaussian
      │     │     └── OCR → (text, avg_confidence, preprocessing_steps)
      │     │             └── warn if avg_confidence < 40%
      │     └── all pages: _normalise_text() — fix font encoding artefacts, strip OCR footers
      └── Spreadsheet path → one ExtractionResult per logical unit
            ├── data / glossary sheets → one result per non-empty row
            │     └── document_id = "<stem>:<key_col_value>"
            └── narrative / key_value sheets → one result per sheet

4. For each ExtractionResult:
      chunk_document(full_text, tables) → list[TextChunk]
            ├── narrative text → semchunk (480 token budget)
            ├── tables → markdown conversion → semchunk
            ├── each chunk tagged with content_type ("narrative" | "table")
            └── [REDACTED] markers repaired if split across boundaries

5. For each chunk:
      ├── embeddings.create(chunk, task="retrieval/document")
      ├── classifier.create(chunk, hypotheses=[...])
      └── extractor.create(chunk, questions=[...])

6. store.write(document_record, chunk_records, extraction_records)
      └── checkpoint written after each batch
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

`extract_text()` returns `list[ExtractionResult]`. PDFs and DOCX return a single-element list. Spreadsheets return N elements — one per row (data/glossary sheets) or one per sheet (narrative/key_value sheets).

| Field | Type | Description |
|-------|------|-------------|
| `pages` | `list[PageResult]` | Per-page text and extraction method |
| `method` | `str` | Strategy used (e.g. `native_narrative`, `scanned_machinewritten`, `spreadsheet`) |
| `error` | `str \| None` | Error message if extraction failed |
| `tables` | `list[TableData]` | Structured table content |
| `forms` | `list[FormField]` | Form field label-value pairs |
| `images` | `list[ImageData]` | Image metadata |
| `text_blocks` | `list[TextBlock]` | Positional text segments with type classification |
| `metadata` | `ExtractionMetadata` | Strategy, confidence, timing, preprocessing steps |
| `document_id` | `str \| None` | Set by spreadsheet extractor; used as `doc_id` in pipeline |

### Parquet Output

**documents.parquet** — one row per source file

| Column | Description |
|--------|-------------|
| `document_id` | Unique identifier |
| `source_path` | Path to original file |
| `doc_type` | Detected document type |
| `extraction_strategy` | Strategy applied |
| `confidence` | Detection confidence |
| `page_count` | Total pages |
| `status` | `success` or `error` |
| `error` | Error message if failed |

**chunks.parquet** — one row per text chunk

| Column | Description |
|--------|-------------|
| `chunk_id` | Unique identifier |
| `document_id` | FK to documents.parquet |
| `text` | Chunk text |
| `token_count` | Tokens in chunk |
| `embedding` | Float array from kanon-2-embedder |
| `classification` | Top label from kanon-universal-classifier |
| `classification_score` | Confidence for top label |

**extractions.parquet** — one row per extracted field per chunk

| Column | Description |
|--------|-------------|
| `extraction_id` | Unique identifier |
| `chunk_id` | FK to chunks.parquet |
| `field` | Question / field name |
| `value` | Extracted answer |
| `score` | Extraction confidence |

## Batch Processing and Checkpointing

The pipeline processes documents in batches (default: 25). After each batch:

1. Results are appended to the Parquet files.
2. A checkpoint record is written noting the last processed document ID.

On resume (`--resume` flag), the pipeline reads the checkpoint and skips already-processed documents. Individual document errors are logged and recorded in `documents.parquet` without stopping the batch.
