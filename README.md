# Womblex

**Womblex turns messy government document releases (FOI bundles, scanned forms,
printed spreadsheets) into a verbatim, versioned Parquet corpus you can search,
analyse and trace back to its source.**

[![CI](https://github.com/DeepCivic/womblex/actions/workflows/ci.yml/badge.svg)](https://github.com/DeepCivic/womblex/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/womblex)](https://pypi.org/project/womblex/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue)](pyproject.toml)
[![Licence: Apache-2.0](https://img.shields.io/badge/licence-Apache--2.0-blue)](LICENSE)

```bash
pip install womblex          # Python 3.12
```

Extract a redacted decision notice (a synthetic document from this repository's
test set):

```bash
curl -LO https://raw.githubusercontent.com/DeepCivic/womblex/main/fixtures/synthetic/documents/quokka-care-decision-notice_redacted.pdf
womblex extract quokka-care-decision-notice_redacted.pdf -o out/
```

`out/quokka-care-decision-notice_redacted.txt` (4,012 characters) begins:

```text
Notice of Administrative Decision

Quokka Cove Out of School Hours Care
Service approval number: SE-40099001 Provider approval number: PR-00099017
Approved provider: Rottnest Community Services Incorporated
Address: 1 Thomson Bay Road, ROTTNEST, WA 6161
The Regulatory Authority has decided to issue a compliance direction to the approved
provider under the Wildlife Care Services National Law.
```

<!-- TODO: add examples/ with the end-to-end walkthrough. -->
See [examples/](https://github.com/DeepCivic/womblex/tree/main/examples) for an end-to-end walkthrough.

## What it does

Government document releases (FOI bundles, papers, scanned forms, printed
spreadsheets) are hard to get text out of without losing key details. Womblex extracts it first and keeps it verbatim, as an ordered stream
of paragraphs, headings, tables and forms. Independent stages then clean, chunk,
annotate, enrich and mask the corpus, each writing its own sidecar beside the
extraction output, so every derived row points back at the source it came from.

It runs on a laptop with no network access, and the same code scales out to a
fleet of workers.

## Features

- **Page-level routing for PDFs and images.** Every page is profiled and routed
  on its own signals (native text, OCR, tables, forms, a spreadsheet printed to
  PDF), so a bundle mixing a cover letter, a scanned form and a printed table is
  handled page by page.
- **Word, Markdown and spreadsheets** read natively, in document order;
  spreadsheets become a cell-grained stream with header detection.
- **Composable downstream stages** — `normalise`, `spellfix`, `redact`, `chunk`,
  `money`, `enrich`, `graph-refresh`, `embed`, `link`, `pii`, `quality` — each
  reading the shards, writing a sidecar and checkpointing independently.
- **Evidence on every annotation.** Money, entity-link and PII rows carry a
  reference to the element, page and character span they came from, labelled
  with whether it was checked.
- **Swappable models.** OCR, layout, PII context, tokeniser, dictionary,
  enrichment and embedding are named slots; a package can register its own model
  through an entry point, with no schema change. The default set is CPU-only.
- **Provenance in the files.** Every Parquet footer records its contract version
  and sensitivity; a run's files also record the run, version, config digest and
  the model that filled each slot.
- **Reference registers.** G-NAF, the ABN bulk extract and shapefiles have their
  own ingests to typed Parquet or GeoParquet.

## Installation

Requires Python 3.12 (the OCR engine, `rapidocr-onnxruntime`, does not install
on 3.13).

```bash
pip install womblex          # the whole pipeline, CPU-only
pip install womblex[api]     # + the /v1 service API (womblex serve)
pip install womblex[pgvector] # + a workflow's Postgres destination (pgvector)
```

The wheel ships only the small model artefacts. The larger local models
(PaddleOCR v5, PP-DocLayout-M, the PII context model) are resolved from
`WOMBLEX_MODELS_DIR`, or from `src/womblex/_models/` and `models/` in a source
checkout; without PP-DocLayout-M the pre-run model check stops `womblex run`.
Every model, and how to connect a hosted one, is in [docs/models.md](docs/models.md).

## Quick start

```bash
# One file, to text or Parquet
womblex extract report.pdf -o output/
womblex extract register.xlsx -o output/ --format parquet

# A corpus described by a config: copy configs/example.yaml and set paths.input_root
womblex run --config my-dataset.yaml            # add --resume to continue after a failure
```

`womblex run` does extraction only. Every other stage is its own command over
the run's shard directory:

```bash
SHARDS=<output_root>/<run_id>/documents
womblex normalise --shards $SHARDS
womblex chunk     --shards $SHARDS --config my-dataset.yaml
womblex money     --shards $SHARDS
womblex manifest  --shards $SHARDS     # rebuild the run-level manifest.parquet
```

[`configs/default-isaacus.yaml`](configs/default-isaacus.yaml) is the reference
full pipeline, with the command sequence and the reason for each setting
commented inside it. With AI chunking on, run `enrich` before `chunk`, so
chunking reuses the enrichment instead of paying for it twice. Dataset-specific
settings belong in config, never in code.

To publish a sample run into a run root:

```bash
womblex seed-demo --output-root runs/   # publishes run-synthetic-demo
```

### Commands

`womblex <command> --help` lists each command's options.

| Command | Purpose |
|---|---|
| `extract`, `run` | Extract one file; extract a corpus from a config |
| `normalise`, `spellfix`, `redact`, `chunk`, `money`, `enrich`, `graph-refresh`, `embed`, `link`, `pii`, `quality` | Downstream stages over a shard directory |
| `manifest` | Rebuild the run manifest (`--lineage OUT`: also as an OpenLineage RunEvent) |
| `ingest-gnaf`, `ingest-abn`, `ingest-geo` | Reference-register ingests |
| `verify-shards` | Audit a run's shard files; optionally diff two runs |
| `verify-evidence` | Re-check every money, entity-link and PII span against the source text it points at |
| `resolve-source` | Map output rows back to their source files, verified by content hash |
| `egress` | Export a finished run as a bundle |
| `enqueue`, `worker`, `jobs`, `finalize`, `run-stage`, `enqueue-stages` | Distributed runs over object storage and the job queue |
| `retry-stage` | Retry a failed distributed stage from its first failed unit |
| `run-workflow`, `validate`, `plan` | Run a workflow file's steps over a local shard directory; check it would run, without running it; count what its paid steps would send, calling nothing |
| `dispatch-workflow`, `schedule-workflow` | Queue a workflow file's steps for the fleet; store its copy and schedule its cron triggers for workers to fire |
| `rerun-collection` | Run a workflow file's steps over its whole collection; unchanged published units are skipped |
| `add-to-collection`, `remove-workflow` | Fire a stored workflow once now, adding its input prefix's new documents to its collection; remove a workflow's schedules and stored copy (its run and shards stay) |
| `deliver` | Deliver a finished run to its workflow file's destinations again, replacing what was delivered |
| `validate-redactions` | Validate the redaction detector against a labels packet (`annotate-redactions` is a deprecated alias of `redact --shards --pdfs`) |
| `ground-truth`, `score` | Build review baselines from a run, and score corrected ones |
| `profile` | Print the inferred column schema of a tabular file |
| `api-token`, `serve` | Mint service tokens; run the service API |
| `seed-demo` | Publish the demo run into a run root or store |

## Integrating with Womblex

Other software integrates in one of two ways.

**The `/v1` service API.** `womblex serve` is an authenticated HTTP API over a
store, a job queue and an ingest location. A client uploads documents, submits a
run, polls it and reads results back; runs are owner-scoped, so a client sees
only its own. Workers do the processing; the API only writes queue rows and
reads output. Everything an operator interface needs is a call here too
(composing and validating a config, presets, ingest previews, a run's stage
presence, chunk detail, logs and progress, the queue, the deployment's
resources), so a user interface is a renderer over the API.

```bash
womblex api-token --client my-app --scope submit --scope read   # token shown once, plus its registry entry
WOMBLEX_API_CLIENTS=clients.yaml womblex serve --store <uri> --dsn <dsn> --ingest <uri>   # 127.0.0.1:8081

H="Authorization: Bearer $TOKEN"
curl -H "$H" -F files=@report.pdf http://localhost:8081/v1/uploads          # returns input_prefix
curl -H "$H" -H "Content-Type: application/json" \
     -d '{"input_prefix": "<input_prefix>"}' http://localhost:8081/v1/runs  # returns the run id
curl -H "$H" http://localhost:8081/v1/runs/<run_id>                         # pending, running, done, failed or cancelled
curl -H "$H" "http://localhost:8081/v1/runs/<run_id>/documents/<source_hash>/text"   # masked layer by default
```

Scopes, ownership, text layers and error codes are in
[docs/service-api.md](docs/service-api.md).

**The Parquet data contract.** A consumer can read a run's Parquet directly, or
receive it as an egress bundle (`womblex egress <run> --to <dest>`). Every file's
footer carries a contract version and a sensitivity (`raw`, `masked` or `none`);
only `masked` and `none` are safe to pass beyond your trust boundary. The rules
are in [docs/contract.md](docs/contract.md), the schemas in
[docs/extraction.md](docs/extraction.md) and the bundle in
[docs/egress.md](docs/egress.md). In Python, `womblex.__all__` is the stable
surface.

## Documentation

The API reference covers every public module, generated from its docstrings.
Build it with `uv run --group docs mkdocs serve` (CI builds it on every PR).
<!-- TODO: link the hosted API reference here, and add it to [project.urls], once it is published. -->

| Document | Covers |
|---|---|
| [functional_requirements.md](docs/functional_requirements.md) | What Womblex must do |
| [architecture.md](docs/architecture.md), [project-structure.md](docs/project-structure.md) | How it is built, and where each thing lives |
| [dataflow.md](docs/dataflow.md), [composable-design.md](docs/composable-design.md) | How data moves between stages, and the stage contracts |
| [extraction.md](docs/extraction.md), [contract.md](docs/contract.md) | The output schema and the consumer contract |
| [service-api.md](docs/service-api.md), [egress.md](docs/egress.md) | The `/v1` service API, and the egress bundle |
| [models.md](docs/models.md), [model-plugins.md](docs/model-plugins.md) | Every model, connecting hosted providers, and plugging in your own |
| [workflows.md](docs/workflows.md) | Workflow files: what they hold, and how they are checked, run, dispatched and scheduled |
| [layout.md](docs/layout.md) | Layout settings and the `layout_regions` sidecar |
| [money-extraction.md](docs/money-extraction.md) | The `money` annotation |
| [evaluation.md](docs/evaluation.md), [accuracy/](docs/accuracy/) | Evaluation metrics, and benchmark reports generated by [womblex-benchmark](https://github.com/DeepCivic/womblex-benchmark) |
| [heuristics_disambiguation.md](docs/heuristics_disambiguation.md) | The CV2 and NumPy heuristics used in detection |
| [deployment-images.md](docs/deployment-images.md) | Docker Compose services and their images |
| [decisions.md](docs/decisions.md) | Why things are the way they are, and dead ends not to retry |
| [steering.md](docs/steering.md), [CHANGELOG.md](CHANGELOG.md) | What's next, and what has shipped |

## Project status

Womblex is **alpha** and pre-1.0.

- **Stable:** the on-disk data contract, versioned apart from the package
  (additive changes are minor bumps with back-fill; renames are major bumps with
  a reader shim), and the Python names in `womblex.__all__`, which change only
  after one minor release of deprecation warnings. See [STABILITY.md](STABILITY.md) for
  the policy and [docs/contract.md](docs/contract.md) for the schema.
- **Moving:** everything outside `__all__`, the CLI options, the stored-workflow
  commands.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to branch, check and open a pull
request, and [STABILITY.md](STABILITY.md) for what may change between releases.

```bash
git clone https://github.com/DeepCivic/womblex.git && cd womblex
uv sync --extra dev --extra api

uv run ruff check src/ tests/
uv run mypy src/
uv run python -m pytest tests/ -v -m "not benchmark"               # what CI runs
uv run python -m pytest tests/ -v -m "not slow and not benchmark"  # fast subset
```

All three are the CI gate. The suite reads only the synthetic fixtures in
[fixtures/synthetic/](fixtures/synthetic/README.md), so a bare clone runs it;
tests that need a paid API key or a Postgres DSN skip without one. Install the
commit hook (secret scan and the semgrep rules in `.semgrep/rules/`) with
`pip install pre-commit==4.6.1 && pre-commit install`. Conventions — file and
merge size caps, what belongs in core — are in [CLAUDE.md](CLAUDE.md); third-party
data policy is in [THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md).

## Licence

[Apache-2.0](LICENSE).

## Acknowledgements

Womblex is a thin layer over mature libraries, including
[pypdfium2](https://github.com/pypdfium2-team/pypdfium2) and
[pdfplumber](https://github.com/jsvine/pdfplumber) (PDF),
[RapidOCR](https://github.com/RapidAI/RapidOCR) running PaddleOCR ONNX models (OCR),
[PP-DocLayout](https://huggingface.co/PaddlePaddle/PP-DocLayout-M) (layout),
[semchunk](https://github.com/isaacus-dev/semchunk) (chunking),
[Presidio](https://github.com/microsoft/presidio) (PII replacement),
[DBOS](https://github.com/dbos-inc/dbos-transact-py) (workflows), and
[python-docx](https://python-docx.readthedocs.io/),
[pandas](https://pandas.pydata.org/) and
[openpyxl](https://openpyxl.readthedocs.io/) (Office formats).
