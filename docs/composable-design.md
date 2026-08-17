# Composable Design

This document describes Womblex's architecture for composable operations.

## Completed Refactor

`pipeline.py` has been split into the `operations/` package (one module per operation — `extract.py`, `redact.py`, `chunk.py`, `pii.py`, `enrich.py` — plus two shared-helper modules, `models.py` (`DocumentResult`, `BatchResult`, `PreconditionError`) and `persist.py` (`write_batch_parquet`, `write_batch_enrichment`), re-exported from `operations/__init__.py`); the thin CLI command layer lives in `cli/pipeline.py`. `config.stages` and the old registry-driven orchestrator (`STAGE_REGISTRY`, `_resolve_stages`, `process_file`) have been removed. Operations are independent functions (`run_extraction`, `run_chunking`, …) — each takes/mutates a `list[DocumentResult]` batch gated by its own `config.<stage>.enabled` flag — that callers compose directly.

A `process_batch()` was later reintroduced in `src/womblex/batch.py` (I7, cloud scale-out) — not the removed orchestrator, but a plain sequencing of the same composable operations (`run_extraction → run_redaction → run_chunking → run_pii_cleaning → write_batch_parquet`) so `womblex run` (local) and the cloud worker (`cloud/worker.py`) execute byte-identically. It does not reintroduce `STAGE_REGISTRY`/`config.stages`.

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
XML (ABN bulk)       ingest_abn(file|dir) → .parquet    records + names Parquet per XML file
SHP                  ingest_geo(dir) → .parquet files   one GeoParquet per SHP file
```

G-NAF, ABN bulk extract, and geospatial ingest are standalone — they produce Parquet directly and do not return `ExtractionResult`. They cannot be followed by transform operations (chunk, redact, PII, enrich). This is by design: structured relational data, registers, and geometry are not narrative text.

The single-file `.txt` output is a CLI convenience for single-unit extractions only (PDF, DOCX, TXT input producing exactly one extraction unit). Multi-unit inputs (spreadsheets) must use `.parquet` output.

### Transform Operations

Each transform is a standalone function. The only contract is: provide the right input type. The names below are shorthand for the composition pattern, not literal function names — the real entry points are `run_extraction` / `run_redaction` / `run_chunking` / `run_pii_cleaning` / `run_enrichment` in `operations/`, each operating on a `list[DocumentResult]` batch (`operations/models.py`), not a bare `ExtractionResult`.

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
money(elements+cells)   *.elements.parquet +     *.money_spans.parquet +   extraction Parquet exists
                        *.table_cells.parquet    *.money_columns.parquet  (impl: process/money_stage.py)
load_graph(parquet_dir) Parquet files            EntityMention + Edge     enrichment Parquet exists (impl: analyse/query.py's load_entity_mentions / load_graph_edges)
```

`money` is an **offline annotation op** in the mould of `quality`: API-free, no
ordering dependency on enrich, and it **never rewrites element or chunk text**.
It reads the extraction Parquet (`*.elements.parquet` + its `*.table_cells.parquet`
sibling) rather than a `list[DocumentResult]` — it operates over a shard directory,
annotating three loci (narrative offsets into the `processing.text_source` layer,
`table_cell`, `sheet_cell`) and writing two joinable sidecars per batch, gated by
`config.money.enabled`. The narrative locus scans the text *reassembled from the
element stream* (`reassemble_narrative`, the same reconstruction chunking uses),
not the `*.chunks.parquet` — chunks are **not** a money precondition. Because its
narrative offsets index the same `processing.text_source` space enrichment
mentions and chunks use, the spans *join* to them downstream; cell spans anchor
to their own coordinates. See [money-extraction.md](money-extraction.md).

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
ingest_abn(dir) → done                                       ABN XML to Parquet, nothing else
ingest_geo(dir) → done                                       SHP to GeoParquet, nothing else
load_graph(parquet_dir) → pii_clean(chunks, graph) → done    re-run PII from saved graph
extract(pdf) → enrich → chunk(chunking_model) → done         AI chunking reuses enrich's Document
extract(pdf) → money → done                                  annotate amounts, no rewrite
extract(xlsx) → .parquet → money → done                      column-evidenced amounts from a register
extract(pdf) → chunk → enrich → build_graph → money → done  graph + money over the one run
```

The `chunk(chunking_model)` row is the AI-chunking single-enrichment reuse seam — the same
persisted-output-reuse shape as `load_graph → pii` (a later stage consumes an
earlier stage's sidecar rather than recomputing). `enrich` writes the raw ILGS
Document to `*.enrichment_doc.parquet`; `chunk` reuses it when `chunking_model`
is set, guarded by byte-identity of `Document.text` against the reassembled
narrative. It is an *ordering* requirement, not a hard dependency: run out of
order or without the sidecar and `chunk` self-enriches (composable fallback,
the same "missing overlay falls back to verbatim" idiom as `text_source`).

The `build_graph → money` row is not a data dependency — it is two independent
sidecars over one run. `money` reads the extraction Parquet (`*.elements.parquet`
+ `*.table_cells.parquet`), never the graph, so it produces byte-identical output
whether run before or after `enrich`/`build_graph`; the arrow only records that
both land in the same shard directory. Everything joins on `source_hash`, so the
run ends up with `*.graph_edges.parquet` **and** `*.money_spans.parquet` keyed to
the same documents, and — because `money`'s narrative offsets index the same
`processing.text_source` space enrichment mentions and chunks use — amounts can
be *joined* to the chunk a mention falls in at query time. That join is an
offset overlap performed downstream, not something the money stage does: it reads
the element stream, never `*.chunks.parquet`, so `chunk` is not a precondition.
In practice this is `womblex money --shards
<run>/documents/` (or `run-stage --stage money` in the object store) run over a
directory the earlier stages already wrote to — its own resumable checkpoint
means re-running only annotates batches without a money sidecar yet.

### Invalid Compositions (precondition violations)

```
chunk without extract — no input
enrich without extract — enrichment needs full document text
build_graph without enrich — graph needs enrichment
pii_clean(advanced) without build_graph — advanced PII needs graph
ingest_gnaf → chunk — G-NAF output is Parquet, not ExtractionResult
ingest_abn → enrich — register Parquet is not ExtractionResult
ingest_geo → pii_clean — GeoParquet is geometry, not text
ingest_gnaf → money — register Parquet has no *.elements.parquet / *.table_cells.parquet to scan
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
- `womblex ingest-abn` calls `ingest_abn_xml()` / `ingest_abn_directory()` directly
- `womblex chunk`, `womblex redact` call individual operations directly
- `womblex money --shards <dir>` calls `money_shards()` directly — the offline annotation op that reads the extraction Parquet and writes `*.money_spans.parquet` + `*.money_columns.parquet`
