# Womblex

Document extraction pipeline for converting Australian government documents into ML-friendly corpus or collections. Extracts text from PDFs and Word documents (native, scanned, forms, hybrid). Spreadsheets are ingested as cell-grained element streams with automatic header/preamble detection, ready for per-record semantic analysis. Reference registers (G-NAF, ABN bulk extract, geospatial) have standalone Parquet ingests that bypass the NLP pipeline.

## Runs on your laptop, scales to a cluster

**Local is the default and always works.** `pip install womblex` gives you the
entire pipeline — extraction, OCR, chunking, PII — running CPU-only against the
local filesystem. No cloud account, no object store, no database, no API key,
no network access at runtime (models are bundled or resolved from `models/`).
A Chromebook is a supported deployment, not a degraded one.

**Cloud is additive, not a different product.** When one machine stops being
enough, the same wheel runs behind object storage and a shared job queue, and
you buy throughput by adding workers — `--scale worker=8` is the whole
operation. What does *not* change when you scale out:

- the extraction logic (`womblex run` and the cloud worker call a byte-identical
  `process_batch` body)
- the OCR engine (the bundled CPU one, unless you explicitly opt into `[cloud-ocr]`)
- the output layout (distributed runs land the ordinary shard layout, so every
  local `--shards` command consumes them unchanged)

Scaling out is not a lock-in: a distributed run's shards sync down and every
local per-stage command (`womblex manifest`, `chunk --shards`, …) consumes them
unchanged, or you run those stages in place with `run-stage` and never sync at
all. See [Environment-Agnostic Execution](#environment-agnostic-execution).

## Design disclosure
This project is designed for everyone with a focus on inexpensive processing. This means Womblex doesn't include many of the more robust 'all in one' OCR models.

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

Pick the row that matches where you are running it. The pipeline logic is
identical in every row — the extras add *reach* (object storage, a shared
queue, hosted APIs), never a different extraction path.

| Deployment | Install | Adds |
|---|---|---|
| **Local CPU** (laptop, Chromebook, air-gapped box) | `pip install womblex` | — the base install is the local deployment |
| **Cloud CPU** (scalable, S3 + Postgres) | `pip install womblex[cloud]` | fsspec + s3fs staging, psycopg3 job queue |
| Enrichment / embeddings | `pip install womblex[isaacus]` | Isaacus SDK — hosted API (`ISAACUS_API_KEY`) or a private SageMaker deployment (`ISAACUS_SAGEMAKER_ENDPOINTS`) |
| Hosted VLM OCR *(advanced)* | `pip install womblex[cloud-ocr]` | boto3 → Mistral Pixtral Large via AWS Bedrock |

`pip install womblex[local]` is accepted and resolves to the plain base
install — it exists so a deployment can state which mode it is, and so
`[local]` and `[cloud]` read as a pair.

Two things worth being explicit about, because they are the usual source of
over-installing:

- **`[cloud]` does not pull in `[cloud-ocr]`.** S3 access goes through
  s3fs → aiobotocore → botocore; boto3 is imported at exactly one site,
  `ingest/llm_ocr.py`, for the Bedrock OCR engine. So a scalable AWS-native
  deployment keeps the cheap, bundled CPU OCR engine unless you opt in.
  `[cloud-ocr]` is the only extra that changes OCR cost and behaviour —
  per-token billing, network egress, and a Pixtral Large model grant.
- **Local vs cloud is a runtime choice, not a build-time one.** The same
  wheel does both; see [Environment-Agnostic Execution](#environment-agnostic-execution).
  Installing `[cloud]` does not commit you to running in the cloud.

For development:

```bash
git clone https://github.com/DeepCivic/womblex.git
cd womblex
uv sync --extra dev
```

A minimal test-fixture set is vendored in this repo (`fixtures/fixtures/`), so a
fresh clone runs most of the suite with no extra setup. The full benchmark set
lives in a separate repository — see [THIRD_PARTY_DATA.md](https://github.com/DeepCivic/womblex/blob/main/THIRD_PARTY_DATA.md)
for how to obtain it.

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

### Isaacus on Amazon SageMaker (private deployment)

Isaacus models can also run [inside your own AWS
account](https://docs.isaacus.com/integrations/amazon-sagemaker), fully
air-gapped — no API key, no egress. Deploy the Marketplace package(s), then set
`ISAACUS_SAGEMAKER_ENDPOINTS` *instead of* `ISAACUS_API_KEY`; every stage that
calls Kanon-2 (`chunk` with AI chunking, `enrich`, `embed`) routes through the
endpoints with no other change.

Subscriptions are per model plus a universal one, so declare what you actually
deployed — comma-separated `name[@region][=model|model|...]`, where an entry
with no `=models` part serves every model:

```bash
export ISAACUS_SAGEMAKER_ENDPOINTS="kanon-2-universal-001"                        # one endpoint, all models
export ISAACUS_SAGEMAKER_ENDPOINTS="embed-001=kanon-2-embedder,enrich-001=kanon-2-enricher"  # per-feature
export ISAACUS_SAGEMAKER_ENDPOINTS="embed-001=kanon-2-embedder,universal-001"     # mixed: plus a catch-all

export ISAACUS_SAGEMAKER_REGION="ap-southeast-2"   # optional; else the AWS SDK default
export ISAACUS_SAGEMAKER_PROFILE="my-aws-profile"  # optional; else the AWS SDK default
```

A stage whose model no endpoint serves fails before its first request, naming
the model and listing what the endpoints do serve. AWS credentials are resolved
by boto3 as usual (SigV4-signed `/invocations` calls). Chunk-size token
counting is unaffected: the Kanon-2 tokeniser is vendored and stays local.

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
womblex normalise --shards output/<run_id>/documents/               # *.normalised_text.parquet (offline text cleanup)
womblex spellfix  --shards output/<run_id>/documents/               # *.spellfix_text.parquet + *.spellfix_corrections.parquet (offline OCR repair)
womblex chunk     --shards output/<run_id>/documents/               # *.chunks.parquet
womblex quality   --shards output/<run_id>/documents/               # *.chunk_quality.parquet (offline chunk annotation)
womblex money     --shards output/<run_id>/documents/               # *.money_spans.parquet + *.money_columns.parquet (offline amount annotation)
womblex redact    --shards output/<run_id>/documents/ --pdfs <dir>  # *.redactions.parquet
womblex enrich    --shards output/<run_id>/documents/               # *.enrichment_entities.parquet (Kanon-2; needs ISAACUS_API_KEY)
womblex link      --shards output/<run_id>/documents/ --config <yaml> # *.entity_links.parquet (register match)
womblex embed     --shards output/<run_id>/documents/               # *.embeddings.parquet (Kanon-2 chunk embeddings)
womblex pii       --shards output/<run_id>/documents/               # *.pii_spans.parquet (audit) + *.clean_text.parquet (masked, terminal)

# Standalone register ingests (bypass the NLP pipeline, write Parquet directly)
womblex ingest-gnaf "G-NAF/G-NAF FEBRUARY 2026" -o output/gnaf   # G-NAF PSV → Parquet
womblex ingest-abn  extracts/ -o output/abn                      # ABN bulk extract XML → records + names Parquet
womblex ingest-geo  shapefiles/ -o output/geo                    # SHP → GeoParquet

# Audit shard integrity (extraction stage)
womblex verify-shards output/<run_id>/
```

## Environment-Agnostic Execution

The system scales from minimum hardware (e.g., a Chromebook) 
to distributed cloud clusters without altering extraction behavior. 
Configurable "knobs," such as parallel thread limits, allow you to optimize 
resource usage for your specific infrastructure.

**You do not need any of this to use Womblex.** Everything below is the
scale-out path for when a single machine is the bottleneck; `womblex run` on a
local directory remains fully supported and produces the same shards.

```bash
pip install womblex[cloud]   # fsspec + s3fs + psycopg3
```

### Selecting the backend

There is no `STORAGE_TYPE` / `QUEUE_TYPE` switch to set, because there is no
branch for one to select. Both choices fall out of what you already pass:

| Choice | Local | Cloud | Selected by |
|---|---|---|---|
| Storage | `--store /data/runs` | `--store s3://womblex` | the URI scheme |
| Execution | `womblex run` | `womblex enqueue` + `womblex worker` | which command you invoke |

`RemoteStore.from_uri` hands the URI to `fsspec.core.url_to_fs`, which returns
a `LocalFileSystem` for a bare path or `file://` and an `S3FileSystem` for
`s3://` (likewise `gs://`, `az://`). The staging code above it is one code
path — a local `--store` runs the whole stage-in → `process_batch` → stage-out
cycle with s3fs never imported. Credentials follow the same rule: the standard
`AWS_*` vars and `WOMBLEX_S3_ENDPOINT` (for MinIO) are read only for `s3://`,
so a local store needs no configuration at all.

Execution mode is the command, not a setting. `womblex run` processes batches
in-process and checkpoints to `CheckpointManager`; `enqueue`/`worker` put the
same `process_batch` body behind the Postgres queue, where the job row's
`status` *is* the checkpoint. Both call byte-identical pipeline code, which is
why a distributed run's output is the ordinary shard layout.

```bash
# 1. Plan: list source docs in object storage, split into batches, enqueue.
#    Idempotent on (run_id, batch_num) — re-run to resume.
womblex enqueue --store s3://womblex --input-prefix inputs/demo \
    --config configs/example.yaml --create-schema

# 2. Process: run as many workers as you like (separate hosts/containers).
#    Each claims batches via FOR UPDATE SKIP LOCKED — no double-processing.
womblex worker --store s3://womblex --config configs/example.yaml \
    --stale-timeout 900            # requeue batches orphaned by crashed workers

# 3. Watch progress.
womblex jobs --run-id <run_id>     # pending/running/done/failed counts

# 4. Finalise once the fleet drains: consolidate the per-batch shard manifests
#    into <store>/runs/<run_id>/manifest.parquet (the local `run` does this at
#    its end; a distributed run has no single end, so it's an explicit step).
womblex finalize --store s3://womblex --run-id <run_id>

# 5. Run downstream stages in the store, without syncing the run down.
#    `run-stage` generalises finalize's shape to the per-batch sidecar stages:
#    one batch staged in at a time, all declared outputs published or none.
#    Idempotent — re-run as more batches land. Ordering is yours to pick.
womblex run-stage --stage normalise --store s3://womblex --run-id <run_id>
womblex run-stage --stage chunk --store s3://womblex --run-id <run_id> \
    --config configs/example.yaml
womblex run-stage --stage embed --store s3://womblex --run-id <run_id>
```

`run-stage` covers `normalise`, `spellfix`, `chunk`, `money`, `enrich`, `embed`,
`link`, `pii`, `graph-refresh` and `quality`. `manifest` is deliberately absent —
`finalize` already does it. Two stages are special: `graph-refresh` rewrites
`*.enrichment_entities.parquet` / `*.graph_edges.parquet` **in place**, so it is
never skipped by output existence and relies on its own idempotency; `quality`
is **run-scoped**, staging every batch's chunks in one pass because its
duplicate-cluster ids are corpus-wide. Pass `--shards <dir>` instead of
`--store`/`--run-id` to run the same contract locally.

Connection details come from `--store`/`WOMBLEX_STORE_URI`, `--dsn`/`WOMBLEX_DB_DSN`
(or `DATABASE_URL`), and the standard `AWS_*` / `WOMBLEX_S3_ENDPOINT` env vars
(MinIO works as an S3 endpoint). Shards land at `<store>/runs/<run_id>/documents/`
in the **ordinary layout**, so once synced down, `womblex manifest` /
`chunk --shards` / every per-stage command consume a distributed run exactly
like a local one — or run them in place with `run-stage`, above.

### Scaling out

Throughput is workers. Each one claims batches with `FOR UPDATE SKIP LOCKED`,
so they cooperate without a broker, without double-processing, and without
coordinating with each other — which means you can add and remove workers
mid-run, on the same host or across hosts, with no reconfiguration and no
restart of the ones already going. A worker that dies mid-batch is not a lost
batch: `--stale-timeout` returns its claim to `pending` and another worker
picks it up. `--idle-timeout` exits a worker that finds no work, so a fleet can
scale to zero on its own once the run drains.

A ready-to-run stack (Postgres + MinIO + scalable workers) lives in
`docker-compose.yml`:

```bash
docker compose up -d postgres minio createbuckets init
docker compose run --rm womblex enqueue --input-prefix inputs/demo \
    --config configs/example.yaml --create-schema
docker compose up --scale worker=4 worker     # raise or lower at any time
```

### Console (optional)

`womblex ui` serves a read-only HTTP API over artefacts a run has already
written — `/api/runs` and `/api/runs/{run_id}/manifest` today. It is a sidecar,
never in-process with the pipeline, and reads either a local run root or the
object store a distributed run published to:

```bash
pip install womblex[ui]                        # add [cloud] to read a store
womblex ui --output-root output/               # local runs, at :8080
docker compose up -d ui                        # or beside the stack above
```

It adds no pipeline logic and writes nothing to a run — the compose service
runs `read_only`. There is no authentication, so it binds to loopback unless
`--host` says otherwise; put your own control in front of anything wider. The
screens that consume this API are planned in [`docs/ui-plan.md`](docs/ui-plan.md).

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
| Spreadsheet | `.csv`, `.xlsx`, `.xls` | pandas cell-grained element stream with header/preamble detection |

### 2. Extraction

Each document type routes to an appropriate extractor. `extract_text()` always returns a `list[ExtractionResult]`:

- **PDFs** return a single-element list. The per-page orchestrator dispatches `_apply_native_page` or `_apply_ocr_page` based on each page's `PageProfile`. PaddleOCR returns per-region confidence scores stored in the document profile. YOLO layout analysis (DocLayNet `yolo11n_doc_layout.pt`, with COCO `yolov8n.pt` as fallback) is called on OCR pages by `_layout_blocks_and_tables` to populate `Element.kind` for the layout regions it detects; a full-page scan whose dominant region is a figure but which OCR's to substantial text is tagged `paragraph` rather than `figure` so its content reaches chunking.
- **DOCX** returns a single-element list with paragraphs and tables interleaved in OOXML body order.
- **Spreadsheets** return one `ExtractionResult` per workbook. Each sheet contributes a leading `kind='sheet_meta'` element followed by one `kind='sheet_cell'` element per non-empty cell. Export products that open with title rows or `key: value` metadata blocks above the real header (e.g. AusTender contract-notice exports) are handled: the header is detected by run-scoring (the candidate row starting the longest run of table-consistent rows below it), preamble rows land verbatim on `sheet_meta.meta["preamble"]`, and row 0 of the cell grid is always the real header. Ragged CSVs (a one-field title row above a wide header) parse rather than fail.

Each result carries a `document_id` used as the primary key downstream.

Text at the extraction boundary is **verbatim** — `_normalise_text` no longer runs in the extraction hot path. Whatever the producing extractor (native text layer, PaddleOCR, DOCX, spreadsheet-print, …) emits is what lands on the element's `text` field. Downstream stages (PII, redaction, chunking) may rewrite `pages[i].text`, but the parquet writer serialises `elements`, so on-disk content stays extraction-time verbatim. Cleanup (font-encoding artefacts, running OCR footers, OCR character-confusions) belongs to downstream offline stages — `womblex normalise` writes a `*.normalised_text.parquet` overlay and `womblex spellfix` writes a `*.spellfix_text.parquet` overlay, both leaving the verbatim `elements` untouched. See `docs/extraction.md`.

### 3. Redaction

Redaction runs as a post-extraction stage, separate from extraction. This avoids false positives that occur when running redaction detection inside OCR (form fields, chart regions, and diagram fills trigger the detector).

Redacted regions can be replaced with `<REDACTED>` markers (preserving sentence structure) or deleted entirely. The stage is configurable: apply after chunking, after enrichment, or both.

### 4. Chunking

Extracted text is split into semantically meaningful chunks using [semchunk](https://github.com/isaacus-dev/semchunk) with the Kanon tokeniser (default 480 tokens, leaving 32-token headroom for Isaacus 512-token context windows). Tables are converted to markdown and chunked separately, with each chunk tagged as `"narrative"` or `"table"`. `<REDACTED>` markers are preserved across chunk boundaries.

Chunking has two invocation modes that share one engine (`chunk_batch`):

- **Per-stage:** `womblex chunk --shards <run_dir>/documents/` consumes the extraction-stage shards directly and writes `*.chunks.parquet` siblings. Independent `CheckpointManager` so the chunk stage resumes without re-extracting. This is the primary workflow for staged corpus runs.
- **E2E composition:** `womblex run --config <yaml>` extracts and chunks in one process (kept for users with simpler corpora).

Both modes reassemble narrative + tables from each source's element stream, then feed every doc's narratives into a single semchunk call (with overlap) and every doc's table markdowns into another (no overlap), so `processes` parallelises across the whole batch. Chunks carry `(start_char, end_char, page_start, page_end, has_redaction, content_type)`; they join back to `elements` via `source_hash` plus offset-range overlap.

**AI chunking (optional).** Setting `chunking.chunking_model` (e.g. `kanon-2-enricher`) switches narrative chunking to semchunk 4's AI chunking — boundaries follow the Isaacus enricher's document structure instead of the offline token split. Off by default, so non-Kanon setups are unaffected. When the `enrich` stage also runs, enrich it once: run `womblex enrich` **before** `womblex chunk`, and enrich persists the graph (`*.enrichment_doc.parquet`) for chunk to reuse instead of enriching twice. A byte-identity guard ensures reuse only happens when the persisted text matches the chunk source; otherwise it self-enriches.

### 5. PII Cleaning

An optional PII stage masks personal identifiers in chunk text. It is **graph-driven**: the primary candidates are PII-typed entities from the Kanon-2 enrichment graph (`natural`→PERSON, `address`→ADDRESS), mapped onto chunks via mention offsets — so PII runs *after* enrichment, not before. Recall is flexed by enrichment granularity, not by a separate detector.

A local regex + cosine-context backstop (PERSON via `all-MiniLM-L6-v2`, ADDRESS via street-type regex) exists but is **opt-in and off by default** (`pii.use_regex_backstop = false`): on this corpus it is low-precision (~15% — orgs and headings get tagged PERSON), so it is reserved for recall experiments. The `all-MiniLM-L6-v2` model is pre-bundled in `models/` and loaded from disk — no network access at runtime.

Masking is **terminal**. The stage writes two siblings and never rewrites the raw chunks that feed Isaacus:

- **`*.pii_spans.parquet`** — one row per detected span (audit/reversible), carrying the graph `entity_id` and its `<PERSON_n>` replacement.
- **`*.clean_text.parquet`** — the masked, publishable text layer (`<PERSON_1>`, `<ADDRESS_1>`, … — typed and numbered off the graph entity), written by default (`pii.write_clean_text = true`).

See `docs/accuracy/PII_CLEANING.md` for the measured baseline and [docs/decisions.md](https://github.com/DeepCivic/womblex/blob/main/docs/decisions.md) for why masking is terminal.

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

See [docs/extraction.md](https://github.com/DeepCivic/womblex/blob/main/docs/extraction.md) for the canonical schema
reference, element kinds, the reassembly query, and the verbatim-text
policy.

With `womblex[isaacus]` enrichment enabled, the per-stage `womblex enrich --shards` writes sidecars alongside each batch:

**`batch-NNNN.enrichment_entities.parquet`** — flat entity mentions for filtering / PII candidates

**`batch-NNNN.enrichment_meta.parquet`** — document-level enrichment metadata

**`batch-NNNN.enrichment_doc.parquet`** — *(only with `enrichment.persist_document`, auto-enabled when AI chunking is on)* the raw ILGS Document per doc, reused by the chunk stage for AI chunking

The E2E graph path (`womblex run`) additionally emits `entities.parquet` and `graph_edges.parquet` for graph queries.

## Project Structure

A file-level map of the source tree lives in
[docs/project-structure.md](https://github.com/DeepCivic/womblex/blob/main/docs/project-structure.md). At a glance:

```
womblex/
├── configs/           # Dataset-specific configurations
├── docs/              # Architecture docs, ADRs, accuracy reports
├── fixtures/          # Test fixtures (separate repo, see THIRD_PARTY_DATA.md)
├── src/womblex/
│   ├── cli/           # CLI subpackage — per-topic command modules
│   ├── operations/    # Independent operations (extract/redact/chunk/pii/enrich)
│   ├── ingest/        # Detection, per-page profiling, PDF/non-PDF extraction
│   ├── redact/        # Redaction detection + post-extraction stage
│   ├── pii/           # Graph-driven PII detection + terminal masking
│   ├── process/       # Chunking + offline text/annotation stages (normalise/spellfix/quality/money)
│   ├── link/          # Record linkage to reference registers
│   ├── analyse/       # Isaacus enrichment + embeddings + entity graph
│   ├── store/         # Parquet schemas, sidecar IO, checkpoints, retention
│   ├── utils/         # Metrics + local model path resolution
│   └── verify/        # Two-pass extraction quality verification
└── tests/
```

See [docs/project-structure.md](https://github.com/DeepCivic/womblex/blob/main/docs/project-structure.md) for the full
per-module breakdown.

## Development

```bash
# Install with dev dependencies (pytest, ruff, mypy live in the extras)
uv sync --all-extras

# A minimal fixture set is vendored; the full benchmark set is optional —
# see THIRD_PARTY_DATA.md.

# Run the suite (no addopts filter — runs everything; heavy tests skip on a
# bare checkout). Use -m "not slow and not benchmark" for the fast subset.
uv run python -m pytest

# Run OCR and accuracy benchmarks (need the full fixtures; minutes-long)
uv run python -m pytest tests/test_fixture_accuracy.py tests/test_womblex_collection_accuracy.py -v

# Type checking
uv run mypy src/

# Lint
uv run ruff check src/
```

Accuracy docs (`docs/accuracy/*.md`) are regenerated automatically at the end of each test run — no manual editing needed.

### Commit hook

Two checks run on the files you stage: a secret scan (`detect-secrets`) and SAST over the
local semgrep rulesets in `.semgrep/rules/`. Install it once per clone:

```bash
pip install pre-commit==4.6.1 && pre-commit install

# Run both over the whole tree rather than just staged files
pre-commit run --all-files
```

`pre-commit` is deliberately not in the `[dev]` extra — adding it would change
`pyproject.toml` and rewrite `uv.lock`, which is a dependency-scoped decision of its own.

The hook is the **only** thing here that stops an action, and `git commit --no-verify`
walks straight past it. CI re-runs both over the whole tree and adds a scan of reachable
history, which is what catches a credential committed behind `--no-verify` and removed
later. Nothing blocks a merge — that is branch protection, a GitHub setting.

If a scan flags something you have checked and know to be safe, record the reason rather
than switching the check off: `# pragma: allowlist secret` for detect-secrets, or
`# nosemgrep: <rule-id> -- <reason>` for semgrep. Both rulesets document their known
false-positive classes in their file headers. When a new benign finding is real drift
rather than a one-off, regenerate the baseline with the exclusions recorded in
`.pre-commit-config.yaml` and review every new entry before committing it.

### Environment check

```bash
bash .github/scripts/doctor.sh
```

Compares what `.env.example` declares against what is actually set, and any declared
runtime pins (`mise.toml`, `.tool-versions`) against the interpreters on PATH. It reads
variable *names* only and never prints a value. Variables under an `# Optional:` comment
are reported but never failed, so an unset `ISAACUS_API_KEY` on an extraction-only clone
is a note rather than an error. Not wired into CI, where none of these are set.

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
