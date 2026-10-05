# Womblex

Womblex turns messy document releases (FOI bundles, government papers,
scanned forms, spreadsheets) into a versioned Parquet corpus that other
systems can trust. Extraction comes first and keeps text verbatim. After it,
independent stages clean, chunk, annotate, enrich and mask the corpus, each one
writing its own sidecar file beside the extraction output.

It runs on a laptop with no network access, and the same code scales out to a
fleet of workers over object storage and a Postgres job queue. The models
behind each step are swappable without changing any output schema.

What Womblex must do is specified in
[docs/functional_requirements.md](docs/functional_requirements.md). This README
is the overview and the command reference.

## What it does

### Extraction
- **PDFs and images** are profiled page by page. Each page is routed on its own
  signals: native text, OCR, tables, forms, or a spreadsheet printed to PDF. A
  single FOI bundle that mixes a cover letter, a scanned form and a printed
  table is handled page by page rather than forced through one strategy.
- **Word, Markdown and spreadsheets** are read natively, in document order.
  Spreadsheets become a cell-grained stream with header and preamble detection,
  and merged regions are kept.
- **Output is an ordered element stream** (paragraphs, headings, tables, forms,
  sheet cells) written as Parquet. Text is verbatim from the extractor; any
  cleanup happens in later stages, as overlays.

### Downstream stages
Each stage reads the extraction shards (and sometimes another stage's
sidecar), writes its own sidecar, and checkpoints independently. A stage can
run whenever its inputs exist, so several orderings are valid.

| Stage | What it writes | Needs |
|---|---|---|
| `normalise` | Cleaned text overlay (whitespace, footer glyphs, configured substitutions) | Offline |
| `spellfix` | OCR character-confusion repairs, gated by an en_AU dictionary, plus an audit file | Offline |
| `redact` | Detected redaction regions | Offline |
| `chunk` | Token-bounded chunks of narrative and tables ([semchunk](https://github.com/isaacus-dev/semchunk)) | Offline; AI chunking is optional and calls a hosted model |
| `money` | Monetary amounts in narrative text and table cells, with exact values | Offline |
| `enrich` | Entities, relationships and a document graph | A hosted enrichment model ([Isaacus](https://isaacus.com/) Kanon-2) |
| `graph-refresh` | Mention-to-chunk edges, rebuilt after chunking | Offline |
| `embed` | One vector per chunk | A hosted embedding model (Isaacus) |
| `link` | Entity mentions matched to a reference register you supply | Offline |
| `pii` | PII spans (audit) and a masked text layer (`<PERSON_1>`) | Offline; reads the enrichment graph |
| `quality` | Chunk-quality annotations, including corpus-wide duplicates | Offline |

### Reference registers
G-NAF, the ABN bulk extract and shapefiles have their own ingests. They write
typed Parquet or GeoParquet directly and bypass the text stages.

### How you use it
- **Other software** integrates in one of two ways: it calls the
  authenticated `/v1` service API (`womblex serve`), or it reads the versioned
  Parquet a run writes, under the data contract. See
  [Integrating with Womblex](#integrating-with-womblex).
- **Operators** drive runs with the `womblex` CLI, where every capability is a
  subcommand.
- **The console** (`womblex ui`) is an optional admin and debugging UI over a
  run's files: queue and stage progress, a document grid, a chunk inspector
  with entity, PII and money overlays, and a pipeline composer. It has no
  authentication, binds to loopback, and is not an integration surface.

## Support status

- **Layout detection is not currently supported for local deployment.** A
  local run still applies the bundled PP-DocLayout-M model to scanned pages,
  but its output (OCR table regions, redaction exclusion zones) has not been
  validated, so don't rely on it. The layout model can be swapped through the
  `womblex.models.layout` plugin slot. Local support returns once layout runs
  as its own stage; see
  [docs/permissive-deps-plan.md](docs/permissive-deps-plan.md).
- **Licensing is in transition.** Womblex is Apache-2.0, but PDF handling still
  depends on PyMuPDF (AGPL-3.0). The plan above replaces it with permissively
  licensed libraries.
- **Alpha.** Schemas are versioned by the data contract (below), but the Python
  surface outside `womblex.__all__` may change between minor releases.

## Installation

Requires Python 3.11 or later.

```bash
pip install womblex          # the whole pipeline, CPU-only
pip install womblex[api]     # + the /v1 service API (womblex serve)
pip install womblex[ui]      # + the admin console (womblex ui)
```

Object-storage staging, the Postgres job queue and the SDKs for the hosted
models are all in the base install. They stay dormant until configured: with
no credentials or endpoints set, nothing calls out. `[local]` and `[cloud]` are
accepted but empty.

**Model files.** The wheel ships only the small artefacts: the en_AU dictionary
and the Kanon-2 tokeniser. The larger local models (PaddleOCR v5,
PP-DocLayout-M, the PII context model) are resolved from `WOMBLEX_MODELS_DIR`,
or from `src/womblex/_models/` and `models/` in a source checkout. Without
them, OCR falls back to the PaddleOCR v4 models inside the `rapidocr` wheel,
and layout detection is skipped. Every model, local or hosted, is listed in
[docs/models.md](docs/models.md), along with how to connect the hosted ones.

## Quick start

```bash
# Extract one file to text or Parquet
womblex extract report.pdf -o output/
womblex extract register.xlsx -o output/ --format parquet

# Extract a corpus described by a config (resumable)
womblex run --config configs/example.yaml
womblex run --config configs/example.yaml --resume
```

`womblex run` does extraction (and redaction detection) only. Every other stage
is its own command over the run's shard directory:

```bash
SHARDS=output/<run_id>/documents
womblex normalise --shards $SHARDS
womblex chunk     --shards $SHARDS --config configs/example.yaml
womblex money     --shards $SHARDS
womblex pii       --shards $SHARDS
womblex manifest  --shards $SHARDS     # rebuild the run-level manifest
```

`configs/default-isaacus.yaml` is the reference full pipeline (extract,
normalise, enrich, chunk, graph-refresh, embed, money, link), with the command
sequence and the reason for each setting commented inside it. With AI chunking
on, run `enrich` before `chunk`, so chunking reuses the enrichment instead of
paying for it a second time. Copy a config
under `configs/` to start a new dataset; dataset-specific settings belong in
config, never in code.

### Other commands

| Command | Purpose |
|---|---|
| `ingest-gnaf`, `ingest-abn`, `ingest-geo` | Reference-register ingests |
| `verify-shards` | Audit a run's shard files; optionally diff two runs |
| `resolve-source` | Map output rows back to their source files, verified by content hash |
| `egress` | Export a finished run as a bundle |
| `ground-truth`, `score` | Build review baselines from a run, and score corrected ones |
| `profile` | Print the inferred column schema of a tabular file |
| `api-token`, `serve` | Mint service tokens; run the service API |
| `ui`, `seed-demo` | Run the console; seed a demo corpus for it |

`womblex <command> --help` lists each command's options.

## Output

Extraction writes four Parquet files per batch, joined on `source_hash` (the
SHA-256 of the source file):

| File | Holds |
|---|---|
| `batch-NNNN.elements.parquet` | One row per element, in document order |
| `batch-NNNN.table_cells.parquet` | Cells of table elements, joined by `(source_hash, parent_elem_order)` |
| `batch-NNNN.form_fields.parquet` | Fields of form elements, same join |
| `batch-NNNN._manifest.parquet` | One row per source file: status, counts, `content_digest` |

Each stage adds siblings named after the same batch, such as
`batch-NNNN.chunks.parquet`. At the end of a run, `<run>/manifest.parquet`
consolidates the batch manifests, and its footer holds the run record: version,
commit, config digest, the stages observed and the models used.

**The data contract.** Every file's footer carries a contract version and a
sensitivity (`raw`, `masked` or `none`). Only `masked` and `none` files are safe
to pass beyond your trust boundary. Content is deterministic given the same
source, version, config and models, which `content_digest` lets you check.
See [docs/contract.md](docs/contract.md) for the consumer rules and
[docs/extraction.md](docs/extraction.md) for the element schema.

## Integrating with Womblex

### The service API
`womblex serve` is a `/v1` HTTP API over a store, a job queue and an ingest
location. Software uploads documents, submits a run over them, polls it, and
reads the results back. Workers do the processing; the API only writes queue
rows and reads output.

```bash
pip install womblex[api]
womblex api-token --client my-app --scope submit --scope read   # token shown once, plus its registry entry
WOMBLEX_API_CLIENTS=clients.yaml womblex serve --store <uri> --dsn <dsn> --ingest <uri>   # 127.0.0.1:8081
```

```bash
H="Authorization: Bearer $TOKEN"
curl -H "$H" -F files=@report.pdf http://localhost:8081/v1/uploads          # returns input_prefix
curl -H "$H" -H "Content-Type: application/json" \
     -d '{"input_prefix": "<input_prefix>"}' http://localhost:8081/v1/runs  # returns the run id
curl -H "$H" http://localhost:8081/v1/runs/<run_id>                         # pending, running, done or failed
curl -H "$H" http://localhost:8081/v1/runs/<run_id>/files                   # keys, row counts, contract footers
curl -H "$H" "http://localhost:8081/v1/runs/<run_id>/documents/<source_hash>/text?layer=elements"
```

- **Tokens and scopes.** Callers present static service tokens; the registry
  stores only their hashes. Scopes are `submit`, `read`, `read_raw` and `admin`.
  `serve` refuses to start with an empty registry.
- **Ownership.** A run belongs to the client that submitted it. Another
  client's run answers 404, a non-admin can submit only over its own upload
  folder, and only `admin` sees runs started from the CLI or console.
- **Stages.** A submitted `preset` or `config` is validated before anything is
  queued, and it only selects which downstream stages run; workers run them
  under their own config. A config that enables no stage gives an
  extraction-only run.
- **Text layers.** The default layer is `masked`. Raw layers (`chunks`,
  `elements`) need `read_raw`. The masked layer is written by the `pii` stage,
  which is never dispatched automatically, because masking is irreversible: an
  operator runs it deliberately with `run-stage --stage pii`. Until then a
  document's masked text answers 404.

Callers poll; there are no webhooks. Endpoints, status rules and error codes are
in [docs/service-api.md](docs/service-api.md), and the OpenAPI surface is
pinned by a test.

### The data contract
A consumer can instead read a run's Parquet directly, or receive it as an
egress bundle (`womblex egress <run> --to <dest>`): the corpus, the deduplicated
source files and an index between them. See [docs/egress.md](docs/egress.md).
Files are self-describing through their footers, as described under
[Output](#output). In Python, `womblex.__all__` is the stable surface.

## Models

Each model sits in a named slot (OCR engine, layout analyser, PII context
model, chunk tokeniser, spellfix dictionary), and the default set is CPU-only.
An installed package can register its own model for a slot through an entry
point, and a config selects it by name, with no schema change. Built-in
alternatives include a hosted VLM OCR engine on AWS Bedrock and a local Ollama
vision model.

Before a run, worker or stage processes anything, the models it needs are
checked (`--models-check off|load|smoke`), and every output file records which
model filled each slot. See [docs/model-plugins.md](docs/model-plugins.md).

## Scaling out

You don't need any of this to use Womblex. A distributed run calls the same
batch code as `womblex run` and writes the same shard layout, so its output can
be synced down and used by every local command, or processed in place.

The backend follows from what you pass: `--store` takes a local path or an
`s3://` (or `gs://`, `az://`) URI, and execution is either `womblex run` or
`enqueue` plus `worker`.

```bash
womblex enqueue  --store s3://bucket --ingest s3://bucket/inbox \
                 --config configs/example.yaml --create-schema
womblex worker   --store s3://bucket --ingest s3://bucket/inbox \
                 --config configs/example.yaml --stale-timeout 900
womblex jobs     --run-id <run_id>
womblex finalize --store s3://bucket --run-id <run_id>    # consolidate the manifest

# Downstream stages, in place in the store...
womblex run-stage --stage chunk --store s3://bucket --run-id <run_id> --config <yaml>
# ...or queued for the workers, ordered and gated by the config
womblex enqueue-stages --run-id <run_id> --config <yaml>
```

Workers claim batches with `FOR UPDATE SKIP LOCKED`, so you can add or remove
them mid-run. A crashed worker's batch returns to the queue after
`--stale-timeout`. `enqueue-stages` never dispatches `pii` (masking is
irreversible) or `quality` (run-scoped); run those with `run-stage`.

Connections come from `WOMBLEX_STORE_URI`, `WOMBLEX_INGEST_URI` and
`WOMBLEX_DB_DSN`. S3 credentials go on `WOMBLEX_S3_ACCESS_KEY_ID`,
`WOMBLEX_S3_SECRET_ACCESS_KEY` and `WOMBLEX_S3_ENDPOINT` (MinIO works as an S3
endpoint). Womblex owns one table, `womblex_jobs` (schema in
`sql/womblex_jobs.sql`), and writes no vectors to the database.

### Docker Compose

`docker-compose.yml` runs Postgres, MinIO, scalable workers and the console,
using published images. Postgres and MinIO sit behind the `local` profile, so
setting the connection variables points the same file at external services
instead. The service API runs under the `api` profile
(`docker compose --profile api up -d api`, port 8081) with the client registry
mounted read-only.

```bash
export COMPOSE_FILE=docker-compose.yml:docker-compose.local.override.yml  # build your working tree
docker compose --profile local up -d postgres minio createbuckets init
docker compose run --rm womblex enqueue --config configs/example.yaml --create-schema
docker compose up --scale worker=4 worker
```

Without the override file, Compose pulls released images; pin a release with
`docker compose --env-file deploy/images.env up`. Each service's image is
listed in [docs/deployment-images.md](docs/deployment-images.md).

## Documentation

| Document | Covers |
|---|---|
| [functional_requirements.md](docs/functional_requirements.md) | What Womblex must do |
| [architecture.md](docs/architecture.md), [project-structure.md](docs/project-structure.md) | How it is built, module by module |
| [dataflow.md](docs/dataflow.md), [composable-design.md](docs/composable-design.md) | How data moves between stages, and the stage contracts |
| [extraction.md](docs/extraction.md), [contract.md](docs/contract.md) | The output schema and the consumer contract |
| [models.md](docs/models.md), [model-plugins.md](docs/model-plugins.md) | Every model, and how to plug in your own |
| [decisions.md](docs/decisions.md) | Why things are the way they are, and dead ends not to retry |
| [steering.md](docs/steering.md), [CHANGELOG.md](CHANGELOG.md) | What's next, and what has shipped |
| [accuracy/](docs/accuracy/) | Benchmark reports, generated by [womblex-benchmark](https://github.com/DeepCivic/womblex-benchmark) |

## Development

```bash
git clone https://github.com/DeepCivic/womblex.git && cd womblex
uv sync --extra dev --extra ui --extra api

uv run ruff check src/ tests/
uv run mypy src/
uv run python -m pytest tests/ -v                               # whole suite
uv run python -m pytest tests/ -v -m "not slow and not benchmark"  # fast subset
```

These three checks are the CI gate. A minimal fixture set is vendored, so a bare
clone runs most of the suite; tests that need the full fixture set, credentials
or optional extras skip. See [THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md) for the
full fixtures. Accuracy benchmarks live in womblex-benchmark, not here.

The commit hook runs a secret scan and the semgrep rules in `.semgrep/rules/`:

```bash
pip install pre-commit==4.6.1 && pre-commit install
```

Record a reviewed false positive at the site (`# pragma: allowlist secret` or
`# nosemgrep: <rule-id> -- <reason>`) rather than switching a check off.
`bash .github/scripts/doctor.sh` compares your environment with `.env.example`.

Contributor conventions (file and merge size caps, what belongs in core) are in
[CLAUDE.md](CLAUDE.md).

## Licence

Apache-2.0. See the support status above for the PyMuPDF dependency.

## Acknowledgements

- [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF handling
- [RapidOCR](https://github.com/RapidAI/RapidOCR) for OCR, running PaddleOCR ONNX models
- [PP-DocLayout](https://huggingface.co/PaddlePaddle/PP-DocLayout-M) for layout detection
- [semchunk](https://github.com/isaacus-dev/semchunk) for chunking
- [python-docx](https://python-docx.readthedocs.io/), [pandas](https://pandas.pydata.org/) and [openpyxl](https://openpyxl.readthedocs.io/) for Office formats
- [Isaacus](https://isaacus.com/) for the Kanon-2 enrichment and embedding models
