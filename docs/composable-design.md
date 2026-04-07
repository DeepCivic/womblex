# Composable Design

This document describes Womblex's architecture for composable operations.

## Completed Refactor

`pipeline.py` has been renamed to `operations.py`. The orchestrator (`run_pipeline`, `STAGE_REGISTRY`, `_resolve_stages`, `process_file`, `process_batch`) and `config.stages` have been removed. Operations are independent functions that callers compose directly.

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
enrich(chunks)          list[TextChunk]          EnrichmentResult         chunks exist
embed(chunks)           list[TextChunk]          list[Embedding]          chunks exist (TODO)
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
extract(pdf) → chunk → enrich → build_graph → pii_clean(advanced) → done
extract(pdf) → chunk → embed → done
extract(csv) → .parquet                                      tabular to Parquet
ingest_gnaf(dir) → done                                      PSV to Parquet, nothing else
ingest_geo(dir) → done                                       SHP to GeoParquet, nothing else
load_graph(parquet_dir) → pii_clean(chunks, graph) → done    re-run PII from saved graph
```

### Invalid Compositions (precondition violations)

```
chunk without extract — no input
enrich without chunk — enrichment needs chunks
build_graph without enrich — graph needs enrichment
pii_clean(advanced) without build_graph — advanced PII needs graph
ingest_gnaf → chunk — G-NAF output is Parquet, not ExtractionResult
ingest_geo → pii_clean — GeoParquet is geometry, not text
extract(csv, 10k rows) → .txt — multi-unit, must use .parquet
```

## CLI

- `womblex run --config` calls operations directly based on enabled flags in config
- `womblex extract <file> --format txt|parquet` calls `run_extraction()` directly
- `womblex ingest-gnaf` calls `ingest_gnaf_directory()` directly
- `womblex ingest-geo` calls `ingest_geospatial_directory()` directly
- `womblex chunk`, `womblex redact` call individual operations directly
