# Composable Design

This document describes Womblex's architecture for composable operations.

## Completed Refactor

`pipeline.py` has been split into the `operations/` package (one module per operation, re-exported from `operations/__init__.py`); the thin CLI command layer lives in `cli/pipeline.py`. The orchestrator (`run_pipeline`, `STAGE_REGISTRY`, `_resolve_stages`, `process_file`, `process_batch`) and `config.stages` have been removed. Operations are independent functions (`run_extraction`, `run_chunking`, …) that callers compose directly.

## Target Model

There are two categories of operation: **ingest** (format-dependent, produces an output file or extraction result) and **transform** (operates on extraction/chunk output).

### Ingest Operations

Each input format has its own ingest path. These are not interchangeable — the format determines which function to call.

```
Input Format         Function                         Output
──────────────────── ──────────────────────────────── ────────────────────────────
PDF / DOCX / TXT     extract(path) → ExtractionResult  ExtractionResult (in memory)
                     extract(path) → .txt file          single-file text (CLI only)
                     extract(path) → .parquet file      Parquet (CLI or batch)
CSV / XLSX           extract(path) → ExtractionResult  ExtractionResult (in memory)
                     extract(path) → .parquet file      Parquet (CLI or batch)
PSV (G-NAF)          ingest_gnaf(dir) → .parquet files  one Parquet per PSV file
SHP                  ingest_geo(dir) → .parquet files   one GeoParquet per SHP file
```

G-NAF and geospatial ingest are standalone — they produce Parquet directly and do not return `ExtractionResult`. They cannot be followed by transform operations (chunk, redact, PII, enrich). This is by design: structured relational data and geometry are not narrative text.

The single-file `.txt` output is a CLI convenience for single-unit extractions only (PDF, DOCX, TXT input producing exactly one extraction unit). Multi-unit inputs (spreadsheets) must use `.parquet` output.

### Transform Operations

Each transform is a standalone function. The only contract is: provide the right input type.

```
Operation               Input                    Output                   Precondition
─────────────────────── ──────────────────────── ──────────────────────── ────────────────────
chunk(extraction)       ExtractionResult         list[TextChunk]          extraction exists
redact_tag(extraction)  ExtractionResult         ExtractionResult         extraction exists (PDF only)
pii_clean(extraction)   ExtractionResult         ExtractionResult         extraction exists
pii_clean(chunks)       list[TextChunk]          list[TextChunk]          chunks exist
enrich(extraction)      ExtractionResult         EnrichmentResult         extraction exists
embed(chunks)           list[TextChunk]          list[Embedding]          chunks exist (impl: analyse/embed_stage.py, I7)
link(enrich entities)   entity mentions          entity_links             enrichment exists (impl: link/stage.py, I7)
build_graph(enrichment) EnrichmentResult         DocumentGraph            enrichment exists
pii_clean(chunks, graph) list[TextChunk] + graph list[TextChunk]          graph exists
load_graph(parquet_dir) Parquet files            EntityMention + Edge     enrichment Parquet exists
```

### Valid Compositions (examples, not exhaustive)

```
extract(pdf) → done                                          just get text out
extract(pdf) → .txt                                          single file text output
extract(pdf) → chunk → done                                  text + chunks
extract(pdf) → redact_tag → chunk → pii_clean → done
extract(pdf) → enrich → build_graph → done                   enrichment of full doc; chunk-level mention map omitted
extract(pdf) → chunk → enrich → build_graph → pii_clean(advanced) → done
extract(pdf) → chunk → embed → done
extract(csv) → .parquet                                      tabular to Parquet
ingest_gnaf(dir) → done                                      PSV to Parquet, nothing else
ingest_geo(dir) → done                                       SHP to GeoParquet, nothing else
load_graph(parquet_dir) → pii_clean(chunks, graph) → done    re-run PII from saved graph
extract(pdf) → enrich → chunk(chunking_model) → done         AI chunking reuses enrich's Document
```

The last row is the AI-chunking single-enrichment reuse seam — the same
persisted-output-reuse shape as `load_graph → pii` (a later stage consumes an
earlier stage's sidecar rather than recomputing). `enrich` writes the raw ILGS
Document to `*.enrichment_doc.parquet`; `chunk` reuses it when `chunking_model`
is set, guarded by byte-identity of `Document.text` against the reassembled
narrative. It is an *ordering* requirement, not a hard dependency: run out of
order or without the sidecar and `chunk` self-enriches (composable fallback,
the same "missing overlay falls back to verbatim" idiom as `text_source`).

### Invalid Compositions (precondition violations)

```
chunk without extract — no input
enrich without extract — enrichment needs full document text
build_graph without enrich — graph needs enrichment
pii_clean(advanced) without build_graph — advanced PII needs graph
ingest_gnaf → chunk — G-NAF output is Parquet, not ExtractionResult
ingest_geo → pii_clean — GeoParquet is geometry, not text
extract(csv, 10k rows) → .txt — multi-unit, must use .parquet
```

Enforcement is **pragmatic**, not blanket: a config-disabled stage passes
through (`enabled=False` → return unchanged) and a per-document data gap in an
otherwise-valid batch is skipped — neither is an error. Genuine *misuse* raises
`operations.PreconditionError`. The enforced case today is graph-driven PII
without a graph: `run_pii_cleaning(pipeline_point="post_enrichment")` when no
completed document carries enrichment (the `pii_clean(advanced) without
build_graph` row above). A partially-enriched batch is tolerated — un-enriched
docs fall back per-document. The remaining rows are structural impossibilities
(wrong output type) that fail naturally at the type boundary.

## CLI

- `womblex run --config` calls operations directly based on enabled flags in config
- `womblex extract <file> --format txt|parquet` calls `run_extraction()` directly
- `womblex ingest-gnaf` calls `ingest_gnaf_directory()` directly
- `womblex ingest-geo` calls `ingest_geospatial_directory()` directly
- `womblex chunk`, `womblex redact` call individual operations directly
