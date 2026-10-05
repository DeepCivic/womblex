# Composable Design

Stage order, stage contracts, valid and invalid compositions, and the CLI that
composes them.

## Operations

There are two categories of operation: **ingest** (format-dependent, produces an output file or extraction result) and **transform** (operates on extraction/chunk output). Ingest always runs first.

### Ingest Operations

Each input format has its own ingest path. These are not interchangeable — the format determines which function to call.

```
Input Format         Function                         Output
──────────────────── ──────────────────────────────── ────────────────────────────
PDF/DOCX/TXT/MD      extract(path) → ExtractionResult  ExtractionResult (in memory)
                     extract(path) → .txt file          single-file text (CLI only)
                     extract(path) → .parquet file      Parquet (CLI or batch)
Standalone image     extract(path) → ExtractionResult  as a one-page PDF (orchestrator OCR path)
CSV / XLSX           extract(path) → ExtractionResult  ExtractionResult (in memory)
                     extract(path) → .parquet file      Parquet (CLI or batch)
PSV (G-NAF)          ingest_gnaf(dir) → .parquet files  one Parquet per PSV file
XML (ABN bulk)       ingest_abn(file|dir) → .parquet    records + names Parquet per XML file
SHP                  ingest_geo(dir) → .parquet files   one GeoParquet per SHP file
```

G-NAF, ABN bulk extract, and geospatial ingest are standalone — they produce Parquet directly and do not return `ExtractionResult`. They cannot be followed by transform operations (chunk, redact, PII, enrich). This is by design: structured relational data, registers, and geometry are not narrative text.

The single-file `.txt` output is a CLI convenience for single-unit extractions only (PDF, DOCX, TXT input producing exactly one extraction unit). Multi-unit inputs (spreadsheets) must use `.parquet` output.

### Transform Operations

Transforms run in one of two forms, over the same primitives.

**In-memory operations** (`operations/`) take and mutate a `list[DocumentResult]` batch (`operations/models.py`), each gated by its own `config.<stage>.enabled` flag. These are the Python API, the `--config` modes of `womblex chunk` / `womblex redact`, and — for extraction and redaction only — the body of `process_batch()`:

```
Operation            Entry point            Input                      Output                          Precondition
──────────────────── ────────────────────── ────────────────────────── ─────────────────────────────── ──────────────────────
extract              run_extraction         paths                      DocumentResult.extraction       —
redact               run_redaction          extraction                 redaction report + annotations  extraction exists (PDF only)
chunk                run_chunking           extraction                 DocumentResult.chunks           extraction exists
enrich (+ graph)     run_enrichment         extraction (+ chunks)      .enrichment + .graph            extraction exists
pii (post_extract.)  run_pii_cleaning       extraction                 page text masked in place       extraction exists
pii (post_chunk)     run_pii_cleaning       chunks                     chunk text masked in place      chunks exist
pii (post_enrich.)   run_pii_cleaning       chunks + enrichment        chunk text masked in place      enrichment exists
```

**Shard stages** operate over a shard directory on disk (`*_shards()` functions), reading and writing sibling Parquet sidecars joined on `source_hash`. Each is declared as a `StageContract` in `cloud/stage_contracts.py` (`STAGE_CONTRACTS`), which the local per-stage commands, `run-stage` and the queue worker all execute. Redaction has no stage contract: detection runs in-batch (`process_batch`), and `womblex redact --shards --pdfs` is a separate annotation command that writes `*.redactions.parquet`.

| Stage | Required inputs | Conditional inputs (config-derived) | Outputs |
|---|---|---|---|
| `normalise` | `.elements`, `.table_cells`, `._manifest` | — | `.normalised_text` |
| `spellfix` | `.elements`, `.table_cells`, `._manifest` | `.normalised_text` (chains on it when present) | `.spellfix_text`, `.spellfix_corrections` |
| `enrich` | `.elements`, `._manifest` | `text_source` overlay (strict); `.chunks` (adds mention→chunk edges) | `.enrichment_entities`, `.enrichment_meta`, `.graph_edges` (+ `.enrichment_doc` when `persist_document` or `chunking_model`) |
| `chunk` | `.elements`, `.table_cells`, `._manifest` | `text_source` overlay (strict); `.enrichment_doc` when `chunking_model` is set | `.chunks` |
| `graph-refresh` | `.enrichment_entities`, `.graph_edges`, `.chunks`, `._manifest` | — | `.enrichment_entities`, `.graph_edges` (rewritten in place) |
| `embed` | `.chunks`, `._manifest` | — | `.embeddings` |
| `money` | `.elements`, `.table_cells`, `._manifest` | `text_source` overlay (strict; `money.text_source` outranks `processing.text_source`) | `.money_spans`, `.money_columns` |
| `link` | `.enrichment_entities`, `._manifest` | — | `.entity_links` |
| `pii` | `.chunks`, `._manifest` | `.enrichment_entities` (the primary candidate source) | `.pii_spans` (+ `.clean_text` when `write_clean_text`) |
| `quality` | `.chunks` | — | `.chunk_quality` (run-scoped: every base in one pass) |

Every suffix is `*<suffix>.parquet`. `manifest` is not a stage: `womblex finalize` (object store) and the end of `womblex run` (local) consolidate the `._manifest` shards.

### Valid Compositions

Stage order is a dependency DAG over the inputs above, not one fixed sequence. `PIPELINE_ORDER` (`pipeline_order.py`) declares the default full-run order — one valid topological sort:

```
extract → normalise → spellfix → enrich → chunk → graph-refresh → embed → money → link → pii → quality
```

Its two non-obvious edges are config-derived, so the contracts alone cannot express them: normalise and spellfix precede enrich and chunk because `processing.text_source` makes both reassemble from their overlays, and enrich precedes chunk so AI chunking reuses the persisted Document. `DOWNSTREAM_STAGES` is the subset a dispatcher (`enqueue-stages`, the console) may queue automatically — every stage except `extract` (the batch queue itself), `pii` (irreversible masking, always a deliberate act) and `quality` (run-scoped). Both remain reachable through `run-stage`.

| After | Valid immediate next stages |
|---|---|
| `extract` (shards) | `normalise`, `spellfix`, `enrich`, `chunk`, `money`, `done` |
| `extract` → `.txt` | `done` |
| `normalise` | `spellfix`, `enrich`, `chunk`, `money`, `done` |
| `spellfix` | `enrich`, `chunk`, `money`, `done` |
| `enrich` | `chunk` (reuses `.enrichment_doc` under `chunking_model`), `link`, `money`, `done` |
| `chunk` | `enrich` (if not yet run), `graph-refresh` (once enriched), `embed`, `pii`, `quality`, `money`, `done` |
| `graph-refresh` | `embed`, `link`, `pii`, `money`, `done` |
| `embed` | `link`, `pii`, `money`, `done` |
| `link` | `pii`, `money`, `done` |
| `money` | any stage whose inputs exist, `done` |
| `pii` | `quality`, `done` (terminal for the text layer) |
| `quality` | `done` |
| `ingest_gnaf` / `ingest_abn` / `ingest_geo` | `done` |

Notes:

- `pii` is terminal. It runs after enrich and embed, reads `*.chunks.parquet` and never rewrites it, and writes the masked layer as a separate `*.clean_text.parquet`. `embed` reads `*.chunks.parquet` too — never the masked text — so embedding before or after `pii` produces the same vectors; the order matters because masking is irreversible and nothing downstream of it should need raw text.
- `chunk → enrich` is valid: enrich reassembles the narrative from the element stream, and present chunks only add mention→chunk edges. Running enrich first is the default because AI chunking can then reuse its Document.
- `graph-refresh` needs both enrichment and chunks; it rebuilds mention→chunk edges after chunking, in place, and is never skipped on output existence.
- `money` is an offline annotation: API-free, it reads only the extraction Parquet (`*.elements.parquet` + `*.table_cells.parquet`, never `*.chunks.parquet` or the graph) and never rewrites element or chunk text, so it may run anywhere after `extract` and produces byte-identical output whether run before or after `enrich` / `graph-refresh`. Its narrative offsets index the same `processing.text_source` space as enrichment mentions and chunks, so amounts join to the chunk a mention falls in by an offset overlap performed downstream, not by the money stage. Its own resumable checkpoint means re-running over a directory the earlier stages already wrote to annotates only batches without a money sidecar yet.
- CSV/XLSX extraction follows the shard row; the documented spreadsheet path is `extract → money`.

The `chunk` reuse of `*.enrichment_doc.parquet` is the AI-chunking single-enrichment
seam — a later stage consumes an earlier stage's sidecar rather than recomputing.
`enrich` writes the raw ILGS Document to `*.enrichment_doc.parquet`; `chunk` reuses
it when `chunking_model` is set, guarded by byte-identity of `Document.text` against
the reassembled narrative. The reuse is an *ordering* requirement, not a hard
dependency (the contract marks it non-strict): run out of order or without the
sidecar and `chunk` self-enriches — as it does for a document enrich had to split,
whose Document is never persisted.

The `text_source` overlay is different. The contracts mark it **strict**, and the
object-store runner refuses a base whose selected overlay is absent
(`InputContractError`). The local per-stage commands do not: `load_overlay` warns
and reassembly proceeds on verbatim text. The render path (`build_ground_truth`)
declares its `text_source` — no default — and calls `load_overlay(..., required=True)`:
a declared non-`elements` overlay that is missing raises rather than baselining
verbatim text, so a ground-truth baseline is never silently produced under a
declared cleaning layer it did not apply.

### Invalid Compositions (precondition violations)

```
chunk / enrich / money without extract — no extraction shards to discover
graph-refresh without enrich or chunk — needs .enrichment_entities, .graph_edges and .chunks
embed / pii / quality without chunk — no .chunks
link without enrich — no .enrichment_entities
pii (post_enrichment, in memory) without enrichment — graph-driven PII needs a graph
ingest_gnaf → chunk — G-NAF output is Parquet, not extraction shards
ingest_abn → enrich — register Parquet is not extraction shards
ingest_geo → pii — GeoParquet is geometry, not text
ingest_gnaf → money — register Parquet has no *.elements.parquet / *.table_cells.parquet to scan
extract(csv, 10k rows) → .txt — multi-unit, must use .parquet
```

Enforcement is **pragmatic**, not blanket: a config-disabled stage passes
through (`enabled=False` → return unchanged) and a per-document data gap in an
otherwise-valid batch is skipped — neither is an error. Where misuse is caught,
the surface depends on the path:

- **In-memory operations** raise `operations.PreconditionError` for genuine
  misuse. The enforced case is graph-driven PII without a graph:
  `run_pii_cleaning(pipeline_point="post_enrichment")` when no completed
  document carries enrichment. A partially-enriched batch is tolerated —
  un-enriched docs fall back per document.
- **Shard stages** (`cloud/stage_runner.py`): `prepare_stage_context` raises
  `StagePreconditionError` before any base is attempted (stage preflight such as
  `link`'s reference register, an unresolvable Isaacus deployment, or a failed
  model check). On the object-store path, a base missing a required input raises
  `NotReady` — logged with the producing stage and skipped, a non-zero exit only
  when every base is blocked — and a base missing a strict conditional input
  raises `InputContractError`, counted as failed while the other bases continue.
- **Queue worker** (`cloud/worker.py`): a stage job whose every base is
  `NotReady` raises `StageNotReady` and is **released** rather than failed, so a
  stage claimed before its upstream has published does not spend a retry.

The register-ingest rows are structural impossibilities: their output has no
extraction sidecars, so shard discovery finds no batch bases.

## CLI

- `womblex run --config` runs extraction only: each batch goes through `batch.process_batch` (extract → redaction detection when `redaction.enabled` → shard write), then the run manifest is consolidated. It does **not** run downstream stages; `config.<stage>.enabled` declares pipeline membership for the dispatchers, it does not make a stage run here.
- `womblex extract <file> --format txt|parquet` calls `run_extraction()` directly.
- `womblex chunk --shards <dir>` calls `chunk_shards()`; `womblex chunk --config` extracts and chunks through `run_extraction` / `run_chunking`.
- `womblex redact --shards <dir> --pdfs <dir>` calls `redact.batch.annotate_redactions_for_shards()` and writes `*.redactions.parquet`; `womblex redact --config` extracts and redacts through `run_extraction` / `run_redaction`.
- `womblex normalise`, `spellfix`, `enrich`, `graph-refresh`, `embed`, `money`, `link`, `pii` and `quality` each take `--shards <dir>` and call their stage function directly (`normalise_shards`, `spellfix_shards`, `enrich_shards`, `refresh_graph_edges`, `embed_shards`, `money_shards`, `link_shards`, `pii_shards`, `quality_shards`).
- `womblex run-stage --stage <name>` runs one `StageContract` — over a local `--shards` directory or an object-store `--store` / `--run-id` prefix — with its preflight; ordering is the operator's.
- `womblex enqueue-stages` writes queue rows for the run's `DOWNSTREAM_STAGES` that the config enables, in `PIPELINE_ORDER`, for the workers to claim; it runs nothing itself.
- `womblex ingest-gnaf`, `ingest-geo` and `ingest-abn` call `ingest_gnaf_directory()`, `ingest_geospatial_directory()` and `ingest_abn_xml()` / `ingest_abn_directory()` directly.
