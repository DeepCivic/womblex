# Data Flow

End-to-end data movement through Womblex, from raw input to Parquet output.

## Overview

The diagram below is the in-memory data path (the `operations` package): each
step's input and the structure it produces. On disk, `womblex run` writes one
extraction shard set per batch, and each downstream stage writes its own
sidecars into the same shard directory, joinable on `source_hash`. G-NAF PSV,
geospatial SHP and ABN bulk extract XML take standalone paths that produce
Parquet directly and never enter this flow.

```
Raw files (PDF / DOCX / MD / CSV / XLSX)
        │
        ▼
┌───────────────────┐
│  detect_file_type │  → DocumentProfile
│  (ingest/detect)  │    (doc-level type, signals, PaddleOCR confidence)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   extract_text    │  → list[ExtractionResult]  (single-element list per source)
│  (ingest/extract) │    PDFs route via orchestrator (per-page dispatch);
│                   │    images too (the PDF seam opens one as a 1-page doc);
│                   │    only DOCX / spreadsheet / text via get_extractor.
└───────────────────┘
        │
        ▼
┌───────────────────────────────────────────┐
│  PDF: extract_pdf_with_plan               │  → ExtractionResult
│  (ingest/orchestrator)                    │    elements: list[Element]
│   ├── profile_pages → list[PageProfile]   │    (paragraph / heading / table /
│   ├── run_layout_step (PP-DocLayout)      │     form / image / sheet_cell / …)
│   ├── _apply_native_page (page.get_text)  │    + legacy view properties
│   └── _apply_ocr_page   (OCR + regions)   │      (.pages / .text_blocks / .tables /
│                                           │       .forms / .images) as read-only
│  Non-PDF: strategy.extract                │      derivations over elements.
│  (strategies_file / spreadsheet)          │
└───────────────────────────────────────────┘
        │
        ▼
┌───────────────────┐
│  run_redaction    │  → RedactionReport on ExtractionResult
│ (operations/      │    (flag / blackout / delete mode; delegates to
│  redact)          │     redact/stage helpers)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  run_chunking     │  → list[TextChunk]
│ (operations/chunk)│    (text, offsets, content_type, has_redaction;
│                   │     engine is process/chunker.chunk_batch)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ run_pii_cleaning  │  → PII spans replaced with <ENTITY_TYPE> tags
│ (operations/pii)  │    (PERSON, ADDRESS; delegates to pii/stage)
└───────────────────┘
        │
  ── extraction complete, caller decides what to do next ──
        │
        ▼  (caller composes operations as needed)
┌───────────────────┐
│  run_enrichment   │  → EnrichmentResult per document
│ (operations/      │    (segments, entities, relationships;
│  enrich)          │     API call in analyse/enrich)
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
│  write_batch_     │  → batch-NNNN.{elements, table_cells,
│  parquet          │     form_fields, _manifest}.parquet
│ (operations/      │     (via store/output.write_results)
│  persist)         │
│  write_batch_     │  → entities.parquet, graph_edges.parquet,
│  enrichment       │     enrichment_meta.parquet (via
│ (operations/      │     store/enrichment_output.write_enrichment_
│  persist)         │     metadata / write_entity_mentions /
│                   │     write_graph_edges)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ verify_shard_     │  → cumulative on-disk size (raises
│ persistence       │    ShardVerificationError on any anomaly)
│ (store/output)    │
└───────────────────┘
```

## Per-Document Flow

For each file processed via the `operations` package:

```
1. detect_file_type(path) → DocumentProfile
      └── spreadsheets also populate DocumentProfile.sheet_meta: list[SheetInfo]

2. get_extractor(profile) → SpreadsheetExtractor | DocxExtractor | TextExtractor | MarkdownExtractor
      (path-based formats only; every seam-openable input goes to the orchestrator)

3. extract_text(path, profile) → list[ExtractionResult]  (single-element list per source)
      ├── PDF / image → orchestrator: per-page PageProfile → native or OCR page
      │     operation → elements appended in page order
      ├── DOCX → elements in OOXML body order
      └── Spreadsheet → one ExtractionResult per workbook: a kind='sheet_meta'
            element per sheet, then kind='sheet_cell' elements

4. Operations (in memory, over list[DocumentResult]):
      ├── run_redaction    → ExtractionResult.redaction_report (list[RedactionInfo]
      │                      per page) + warning strings; blackout / delete modes
      │                      rewrite pages[i].text, never elements
      ├── run_chunking     → DocumentResult.chunks: list[TextChunk]
      ├── run_pii_cleaning → page or chunk text masked in place
      └── run_enrichment   → EnrichmentResult + DocumentGraph per document

5. store
      ├── write_batch_parquet(batch, path) → batch-NNNN.{elements, table_cells,
      │     form_fields, _manifest}.parquet
      ├── write_batch_enrichment(batch, dir) → entities / graph_edges /
      │     enrichment_meta (legacy E2E writer; the per-stage enrich sidecars
      │     reuse the same schemas, keyed on source_hash)
      ├── downstream sidecars, each written by its own stage over the shard dir
      ├── write_run_manifest(shard_dir) → <run_root>/manifest.parquet — every batch
      │     _manifest consolidated into the published source_hash → doc_id/filename
      │     table. Written at the end of `womblex run`; regenerable with
      │     `womblex manifest --shards`.
      └── per-stage CheckpointManager (JSON, resumable; reconciled against the
            shards on resume via reconcile_stage_checkpoint_with_shards)
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
source); a spreadsheet's cells live as `kind='sheet_cell'` elements on
that single result.

| Field | Type | Description |
|-------|------|-------------|
| `pages` | `list[PageResult]` | Per-page text; mutable so PII / redaction can rewrite `page.text` |
| `elements` | `list[Element]` | Canonical structural stream — what the parquet writer serialises |
| `method` | `str` | For PDFs / images the doc-level `DocumentType` value (e.g. `native_narrative`, `scanned_machinewritten`, `hybrid`); otherwise the extractor (`spreadsheet`, `docx`, `text`, `markdown`) |
| `error` | `str \| None` | Error message if extraction failed |
| `document_metadata` | `dict[str, str]` | Document-level key/value metadata (populated by spreadsheet-print extraction; empty otherwise) |
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
| `page_start` / `page_end` | `int \| None` | Pages covering the chunk; `None` for sources without page semantics |
| `elem_order` | `int \| None` | Document-order anchor; set for table chunks only |
| `token_count` | `int \| None` | Chunk length in the chunker's own tokens; set by `chunk_batch` |

### Chunk and Enrichment Parquet

**batch-NNNN.chunks.parquet** — one row per chunk

Written beside the four extraction shards. Joins
back to elements on `source_hash` plus offset-range overlap with the
reassembled element-stream text — not via `elem_order`, because a
narrative chunk straddles multiple elements. Table chunks are the
exception: each comes from exactly one table element, so they carry that
element's `elem_order` as a document-order anchor (see below).

| Column | Type | Description |
|--------|------|-------------|
| `source_hash` | string | FK to `_manifest.parquet` |
| `chunk_index` | int32 | 0-based, per source_hash |
| `text` | string | Chunk text |
| `start_char` | int32 | Offset into the reassembled element-stream narrative (for `content_type='narrative'`) or table markdown (for `content_type='table'`) |
| `end_char` | int32 | Exclusive end offset |
| `content_type` | string | `"narrative"` \| `"table"` |
| `has_redaction` | bool | True if the chunk text contains the `<REDACTED>` marker (set by chunk_batch). Flag-mode redaction may flip this later. |
| `page_start` | int32 (nullable) | Page covering `start_char`; `null` for sources without page semantics (DOCX, spreadsheets) |
| `page_end` | int32 (nullable) | Page covering `end_char-1`; `null` for sources without page semantics |
| `elem_order` | int32 (nullable) | Document-order anchor: the `elem_order` of the table element this chunk came from. Set for `content_type='table'` only — `null` for narrative chunks (they straddle elements) and for spreadsheet sheets (a sheet aggregates many `sheet_cell`s and has no narrative to be ordered against). Recovering narrative ↔ table document order needs the elements too, since the anchor and a narrative `start_char` are in different coordinate spaces: `chunker.element_spans(elements)` maps each element to its narrative offsets, and `chunker.chunks_in_document_order(rows, spans)` interleaves the two projections (read the elements under the same `text_source` overlay the chunks were written under). Back-filled as `null` when reading shards written before the column existed. |
| `token_count` | int32 (nullable) | The chunk's length in tokens, measured by the chunker's own token counter over the final chunk text — the number semchunk budgeted against `chunk_size`, not a re-estimate. Counted after the `<REDACTED>` split repair, so a merged chunk reports the merged text. Character and word counts are derivable from `text`; a token count is not without loading the tokeniser, which is why it is a column. **Exact at or below `chunk_size`, a floor above it**: semchunk derives `max_token_chars` from the tokeniser's vocabulary and its counter then short-circuits over-budget text to `chunk_size + 1` instead of tokenising it, and the only chunk that can exceed budget is a repair-merged one. Back-filled as `null` when reading shards written before the column existed. |

**entities.parquet** — one row per entity *mention* (from enrichment)

| Column | Description |
|--------|-------------|
| `source_hash` | FK to `_manifest.parquet` |
| `entity_id` | Entity identifier (shared across a person/location/term's mentions) |
| `entity_label` | `person` \| `location` \| `term` \| `external_document` |
| `name` | Resolved entity name |
| `entity_type` | Entity subtype (`natural`, `corporate`, `politic`, `country`, `state`, etc.) |
| `role` | Person role (`seller`, `buyer`, `other`, …); empty for non-person entities |
| `mention_start`, `mention_end` | Offset of this single mention in the text named by `text_layer` |
| `chunk_index` | Narrative chunk this mention falls in; `-1` if not mapped to a chunk, and always `-1` for a table mention |
| `text_layer` | The text the offsets index: the narrative under its element-text layer (`elements` / `normalised` / `spellfix`), or `table_markdown` for a table enriched on its own. Null on files written before contract 1.4 (narrative) |
| `elem_order`, `sheet` | The table element, or the spreadsheet sheet, a `table_markdown` mention lies in; null for narrative mentions |

**graph_edges.parquet** — one row per relationship edge *property* (from enrichment)

| Column | Description |
|--------|-------------|
| `source_hash` | FK to `_manifest.parquet` |
| `source_id` | Source node identifier |
| `target_id` | Target node identifier |
| `relation` | Relationship type |
| `prop_key`, `prop_value` | One edge-property pair per row; empty-string pair for edges with no properties |

**enrichment_meta.parquet** — one row per enriched document

| Column | Description |
|--------|-------------|
| `source_hash` | FK to `_manifest.parquet` |
| `doc_type_enriched` | `statute` \| `regulation` \| `decision` \| `contract` \| `other` |
| `jurisdiction`, `title` | Document-level metadata Kanon-2 inferred |
| `segment_count` | Number of structural segments |
| `person_count` | Number of persons identified |
| `location_count` | Number of locations identified |
| `term_count` | Number of defined terms |
| `external_doc_count`, `date_count`, `heading_count`, `junk_span_count` | Further per-document enrichment counts |
| `table_count` | Tables sent to the enricher on their own; null when none were, when `enrichment.include_tables` is off, and for a document with a table that failed. A document with tables but no narrative still gets a row |

## G-NAF Standalone Ingest

G-NAF PSV files bypass extraction entirely. The flow is:

```
.psv files (pipe-delimited; a header row is detected and skipped by _has_header_row)
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

## ABN Bulk Extract Standalone Ingest

ABN Lookup bulk extract XML files bypass extraction entirely. Each input
(`yyyymmddPublicNN.xml`, ~6 GB uncompressed across 20 files) is stream-parsed
with constant memory and projected into two Parquet siblings. The flow is:

```
ABN bulk extract .xml (Transfer → many ABR records)
        │
        ▼
┌───────────────────┐
│  ingest_abn_xml   │  → <stem>.parquet        (one row per ABR record)
│  (ingest/         │  → <stem>_names.parquet  (one row per registered name,
│   abn_bulk)       │     keyed by ABN, for link/ register consumption)
└───────────────────┘
```

Records carry ABN/status, entity type, the main or legal-entity name parts
(given names split into `given_name_1` / `given_name_2`), state/postcode, ACN
and GST; absent optionals are `""`. Provenance (`abn.schema_version`,
`abn.source_file`, `abn.source_md5`, `abn.row_count`) rides as Parquet
metadata. Failures are isolated per file: a malformed or unreadable file logs
with its source name, removes any partial output, and lets the directory
ingest continue.

CLI: `womblex ingest-abn <file-or-dir> -o <output_dir>`

## Batch Processing and Checkpointing

The `womblex run` CLI command processes documents in batches (default: 100). After each batch:

1. The batch's completed results are written as their own `batch-NNNN.*` shard set (elements, table_cells, form_fields, _manifest) — earlier shards are never appended to.
2. A checkpoint record is written (`CheckpointManager` in `store/checkpoint.py`) noting processed document IDs, batch number, and success/failure counts.

On resume (`--resume` flag), the CLI reads the checkpoint JSON and skips already-processed documents via `filter_unprocessed()`. Individual document errors are logged (with the document ID, in the console and the batch log) and counted in the checkpoint's failure count without stopping the batch. Failed documents are excluded from the shards — `write_batch_parquet` writes completed results only — so they have no `_manifest.parquet` row.
