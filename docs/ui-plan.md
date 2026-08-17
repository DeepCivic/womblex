# Womblex Console — UI Plan

A proposal for the optional Womblex UI. Visual system: [`../DESIGN.md`](../DESIGN.md).
Screen requirements: the *Womblex Platform UX Requirements* brief (five domains —
Dashboard, Pipeline Composer, Corpus Inspector, Semantic Chunk Inspector,
Resources Console).

Status (verified against the tree): merges 1–6 and 8–11 are complete end to
end — design system, read API skeleton, sidecar image, frontend shell, Corpus
Inspector, Chunk Inspector, Dashboard, Pipeline Composer, Resources Console and
Execution Controls all have both their endpoints and their screens. Merge 8
(dashboard) is now complete: `GET /api/dashboard` is joined by
`ui/src/routes/dashboard` — a run-scoped, self-refreshing screen with KPI
tiles over the exact queue counts, a `locked_by` worker fleet, stale-job
detection that names what a worker's `--stale-timeout` would recover, the
`womblex_jobs` list itself, and per-stage checkpoint progress read from inside
the selected run. It renders in both deployments: with no queue configured the
checkpoint half still shows, so a local operator sees stage progress without a
DSN. Merge 11 (execution controls) is likewise complete: `GET
/api/execute/status` and `POST /api/execute/enqueue` are joined by
`ui/src/routes/execute` — a capability banner that names the missing piece
when the console cannot dispatch, and a configure-and-run form that enqueues
an extraction run and points the run selector at it.

Merge 7 is the one remaining **partial**: the feedback write path (`POST
/api/runs/{id}/feedback` + one-file-per-report writer) is in, but the
`ReportIssue` control is not wired into any screen yet.

In short: every planned *endpoint* exists and every *screen* but one is built.
One *control* (7) — the report action on the inspector screens — remains, over
an endpoint that is already there. Per-merge state is tracked in the §5 table.

## 1. The governing principle

**The console reads artefacts the pipeline already writes. It adds no pipeline
logic.**

This is the [CLAUDE.md](../CLAUDE.md) thin-adapter rule on a new surface: the UI
is *configuration, invocation and output formatting* over library functions. Any
iteration, aggregation or orchestration it appears to need belongs in Womblex
proper, behind a Python API the CLI can call too.

Three consequences worth stating up front:

- **v1 surfaces only what already exists.** No new schemas, no new columns, no
  new instrumentation for the UI's benefit. Where a screen wants a number the
  system does not record, the screen shows less — it does not grow a side table.
  §4 is the honest list of what that costs.
- **No writes to stage outputs.** Reviewers report problems; they do not edit
  parquet (§4).
- **The whole designed workflow, on a screen.** The mission is that someone can
  use Womblex as designed without a CLI or a coding agent — so the console
  covers configure, run, per-stage run, verify and inspect, not a read-only
  subset (§4).

## 2. Architecture — the console is a sidecar

**The UI runs as its own container, never in-process with the pipeline.** It is
a separate process reading shared state, so the same image serves a local
operator and a cloud deployment with nothing but env vars changing — the shape
`docker-compose.yml` already uses for workers.

```
womblex[ui]  →  fastapi + uvicorn        (no new core dependencies)

src/womblex/ui/
├── app.py         # FastAPI app factory; serves the built SPA as static files
├── deps.py        # Resolve store / output_root / optional JobQueue per request
├── routes/        # runs, corpus, chunks, stages, resources, jobs, execute
└── readers.py     # Thin pyarrow readers over the shard sidecars
ui/                     # SvelteKit SPA; built during the image build
src/womblex/cli/ui.py   # `womblex ui [--port 8080] [--audit-only]`
```

State reaches the sidecar exactly the way it reaches a worker — no new
configuration surface:

| Deployment | Runs | Queue |
|---|---|---|
| Local | `output_root` bind-mounted read-only | Absent; dashboard falls back to checkpoints |
| Cloud | `WOMBLEX_STORE_URI` + `WOMBLEX_S3_ENDPOINT` via `store/remote.py` | `WOMBLEX_DB_DSN` via `JobQueue` |

A `ui` service joins `docker-compose.yml` alongside `worker`, sharing the same
`x-cloud-env` anchor and publishing one port.

Because the cloud sidecar reads a bucket rather than a filesystem, **remote
reads are in scope from the first merge**, not deferred behind a flag. Runs are
mounted read-only in both shapes and the console has no code path that writes to
one; its only writable surface is the feedback prefix (§4), a sibling of `runs/`
rather than a child of any run.

Frontend stack matches DeepCivic exactly — SvelteKit, Tailwind v4 `@theme inline`
tokens, shadcn-svelte, svelte-motion, Material Symbols — so components and token
names port between the two codebases. The one divergence is that fonts and icons
are **self-hosted in the bundle**, not fetched from Google Fonts, because the
pipeline's no-network-at-runtime property should not be broken by its own UI.

`[ui]` sits alongside the existing deployment-shaped extras (`[local]`,
`[cloud]`) and, like `[cloud]`, must not pull boto3 of its own. (boto3 is a
core dependency now — for the Bedrock VLM OCR engine and the SageMaker SigV4
client — but the console never imports it: it reads artefacts, makes no AWS
call.)

## 3. Screen → data source

Every screen maps onto artefacts that exist today — the pipeline is already
instrumented for this UI, largely by accident of having been built shard-first.

| Screen | Reads | Notes |
|---|---|---|
| **Dashboard** | `JobQueue.stats()` + its read-only views (`list_jobs`, `workers`, `stale_jobs`, `throughput`); per-stage `CheckpointState` JSON | Queue counts are exact. Throughput is derived from batch completion times; CPU/memory is **not** recorded and v1 does not show it — see §4 |
| **Pipeline Composer** | `config.py` Pydantic models; `cloud/stage_contracts.py` | The composer is a form over the config models plus a graph drawn from `STAGE_CONTRACTS` |
| **Corpus Inspector** | `<run_id>/manifest.parquet`; `store/shard_audit.audit_shard_directory()` | `MANIFEST_SCHEMA` is already the documents table: `doc_id`, `filename`, `ext`, `extraction_method`, element/cell/field counts, `status`, `error` |
| **Chunk Inspector** | `*.chunks.parquet`, `*.enrichment_entities.parquet`, `*.graph_edges.parquet`, `*.pii_spans.parquet`, `*.clean_text.parquet`, `*.money_spans.parquet`, `*.chunk_quality.parquet` | All joinable on `source_hash` (+ `chunk_index`) |
| **Resources Console** | `store/remote.storage_options_from_env()`, `is_remote_uri`; `utils/isaacus_client.unserved_models()`; `JobQueue` connectivity | Connection *testing* already exists as library code |

**`STAGE_CONTRACTS` is the Pipeline Composer's graph.** Each `StageContract`
already declares `required_inputs`, config-derived `conditional_inputs`,
`outputs`, `scope`, `mutation`, and whether it `needs_isaacus_api`. The
requirement that the composer be "the only place logical guardrails are enforced
(e.g. ensuring extraction precedes chunking)" is satisfiable by rendering that
existing structure — the guardrail is `chunk.required_inputs` naming the
elements sidecar, not a rule re-typed in TypeScript. **Do not hand-code the DAG
in the frontend**; serve it from `STAGE_CONTRACTS` so it cannot drift.

Which config section carries a stage's `enabled` toggle is *not* derivable
from `StageContract` — contracts name parquet suffixes, not config fields — so
`ui/composer.py` declares that map (`CONFIG_SECTION`) beside the config models
and serves it per node, under a test that every name is a real `WomblexConfig`
field. Same rule one step out: the frontend re-types nothing the library knows.

**Stage checkpoints live inside the run.** Every shard stage writes its
`CheckpointState` to a dot-directory under the run root
(`<run>/.chunk-checkpoint/`), and `StageContract.checkpoint_dirname` already
names each one. So the queue-less dashboard reads stage progress out of the
run the console is *already* pointed at — no checkpoint path to configure,
and the same code serves local and store-backed deployments. Derive that map
from `STAGE_CONTRACTS`, as with the composer's DAG; do not re-type it.

**Lifecycle checkpoints are sidecar presence.** "Jump between Raw → Extracted →
Redacted → Enriched" is, concretely, which sidecar suffixes exist for a given
`source_hash`. `scan_sidecar_directory(shard_dir, suffix)` already answers this
per stage.

**`verify-shards` already emits JSON.** `womblex verify-shards --format json`
calls `format_audit_json`, so the Corpus Inspector's "trigger shard-level
integrity verification" action is an existing function call, not new work.

## 4. What v1 does not show, and why

The console surfaces what the system records. Where the brief asks for something
the system does not record, v1 shows less rather than growing instrumentation.

| Brief asks for | Recorded today? | v1 behaviour |
|---|---|---|
| Cluster CPU / memory | **No.** `womblex_jobs` has no heartbeat or metrics columns | Not shown. Adding them is a queue schema migration — a library change on its own merits, not a UI prerequisite |
| Real-time throughput | Derivable | Shown, from `updated_at` deltas on queue rows and checkpoint mtimes. No new schema |
| Live local-run progress | Batch-granular | Shown as batch-granular and labelled as such. `CheckpointState` writes once per batch; a smoother bar would be fiction |
| Stalled-job identification | Yes | Shown. A `running` row past `--stale-timeout` is exactly what `requeue_stale` already acts on |
| Worker fleet status | Partly | Shown from `locked_by` on running rows: which workers hold which batches. Not liveness — an exited worker leaves a stale lock, which reads as stalled |

The one genuinely missing read is a **run index**: `store/retention.list_runs()`
returns paths, and the run selector wants run_id, document count, stages present
and timestamps — a small `describe_run()` helper the CLI benefits from too.

### Editing settings without becoming a secret store

The Resources Console and Pipeline Composer share **one writable config volume**
— the only mount besides feedback that is not read-only. Both write YAML that
Pydantic has already validated, activated by restart: leaner than live reload,
and it keeps a running job's config immutable for its whole life.

**Secrets stay in the environment**, displayed masked and labelled as
env-provided, with no field that accepts one. Everything else — bucket URIs,
endpoints, paths, intervals, batch sizes, stage toggles — is editable. That line
is what keeps the console cheap: accepting a credential means storing one, and
storing one means encryption at rest, key rotation and an access log. Worth
doing deliberately later; not worth acquiring by accident in v1.

### Running the pipeline from the screen

The console reaches the whole designed workflow **by calling the same library
functions the CLI calls** — `process_batch()`, `chunk_shards()`,
`enrich_shards()` — never by shelling out. The CLI is already a thin argparse
wrapper over those functions and the console is a second one over the same
ones, so no web request can become an arbitrary command.

Dispatch is the queue in both deployments: the console enqueues, workers do the
work, which is what makes an hours-long job observable and survivable. The cost
is that a screen-driven local deployment runs the compose stack (Postgres +
MinIO, already in `docker-compose.yml`) rather than a bare `womblex run`; a
queue-less local console would need its own background runner and progress
reporting. Deferred, and worth revisiting if that requirement bites.

`--audit-only` remains the switch: passing it gives a pure auditing console.
Execution is on by default; the switch is enforced in one place —
`ui/execute.py`'s guard, which every write action calls before it touches the
store or the queue — and refuses with 403 (audit-only) or 409 (no store /
no queue configured), the two states `ExecutionCapability` distinguishes.
Execution is queue-only, so it needs *both* a `--store` and a `--dsn`; a
local `output_root`-only console can configure and audit but not dispatch, and
`GET /api/execute/status` says so rather than half-working.

### Scale-to-zero is in scope, and already supported

Womblex is event-driven by construction. `run_worker()` already takes `once` and
`idle_timeout`: a worker that finds an empty queue exits and the container goes
away. Enqueue is the event, drain is the shutdown signal, and cold start has the
same latency character as invoking the CLI locally.

So the Resources Console shows fleet state (workers holding batches, queue
depth, idle timeout) and can enqueue work, which is what brings workers up.
Womblex still implements no scheduler: the platform starts containers — compose
`--scale`, a Kubernetes Job, an ECS task — and Womblex contributes a worker that
knows how to exit. That division makes scale-to-zero free rather than a
feature.

### Reporting a bad record, instead of editing it

No screen writes to a stage output. Masking is terminal by design and the
pipeline owns its parquet; a console that rewrote `clean_text.parquet` would be
a second, unversioned producer of it.

Instead, any record in the Corpus Inspector or Chunk Inspector carries a
**report action** that writes **one file per report** — never an append, so
there is no read-modify-write and no lost update when two reviewers click at
once:

```
<store>/feedback/<run_id>/<iso8601>-<short-uuid>.json     # S3 in cloud
<feedback_dir>/<run_id>/<iso8601>-<short-uuid>.json       # disk locally

{"reported_at": "2026-08-16T04:11:09Z", "reported_by": "…", "run_id": "…",
 "record_type": "chunk", "source_hash": "9f2c…", "chunk_index": 47,
 "row": {"text": "…", "content_type": "narrative", "has_redaction": true},
 "note": "PERSON mask missed a signature block"}
```

Same layout in both deployments — only the storage location differs, so one
code path serves both. **Feedback cannot affect a run**: it is a sibling of
`runs/`, so re-running a stage, applying retention, or deleting a run neither
disturbs accumulated feedback nor is disturbed by it. The row is embedded
rather than referenced for the same reason. `reported_by` comes from an env var
or trusted header and is **advisory, not verified** — there is no auth (§6) —
but it costs one string now and a migration later.

**Operator-saved presets take the same shape.** The Pipeline Composer can save
a composed config as a named preset and delete one it saved (a built-in is
code, so only saved presets are deletable). Each preset is one JSON file, and
`ui/readers.py` owns the local-vs-store split exactly as it does for feedback:
locally under a writable `--presets-dir` (`$WOMBLEX_UI_PRESETS_DIR`; absent
disables saving, built-ins still serve), and in store-backed mode under the
store's own `presets/` prefix — a sibling of `runs/` and `feedback/`. So a
store-backed console saves presets without any writable mount, and the compose
`ui` service stays `read_only` with no volume change. A preset is a *partial*
config — `dataset`/`paths` are stripped on save (they name the run, not the
shape) and the overlay is validated against the same `WomblexConfig(**raw)`
the built-ins are, so a preset that would not load is refused at save (400)
rather than 500-ing whoever later picks it. The composer also hands a composed
run off to the queue: the queue carries no config (workers read their own
`--config` at launch), so this is only `paths.input_root` → `input_prefix`
(confirmed by the operator, since it may be absolute/local while the prefix is
store-relative) and `dataset.run_id` → `run_id`, through the same enqueue the
Execution Controls use.

## 5. Delivery sequence

Sized to the 500-changed-line merge cap. Each merge stands alone and leaves the
tree green.

| # | Merge | Contents | Status |
|---|---|---|---|
| 1 | **Design system** | `DESIGN.md` + this plan *(this change)* | ✅ Done |
| 2 | **Read API skeleton** | `[ui]` extra, `src/womblex/ui/app.py`, `womblex ui` command, `describe_run()`, `/api/runs` + `/api/runs/{id}/manifest`, local and store-backed, tests against a fixture shard dir | ✅ Done |
| 3 | **Sidecar image** | `Dockerfile.ui`, the compose `ui` service, read-only container | ✅ Done |
| 4 | **Frontend shell** | `ui/` SvelteKit workspace + the SPA build stage, tokens from `DESIGN.md`, top bar + side nav, run selector, theme + density toggles | ✅ Done |
| 5 | **Corpus Inspector** | Documents grid, lifecycle-checkpoint switcher, failure filter, `verify-shards` action | ✅ Done (endpoints + screen) |
| 6 | **Chunk Inspector** | Chunk reader endpoints, `ChunkCard`, entity/PII/money overlays | ✅ Done (endpoints + screen) |
| 7 | **Report action** | One-file-per-report writer + `POST /api/runs/{id}/feedback`; `ReportIssue` control on the inspector screens | ⚠️ Write path only — endpoint + writer landed; `ReportIssue` control not yet on any screen |
| 8 | **Dashboard** | Queue stats, job list, stale detection, fleet view from `locked_by`, KPI tiles and throughput | ✅ Done (endpoints + screen; run-scoped, self-refreshing, renders the checkpoint half with no queue configured) |
| 9 | **Pipeline Composer** | `STAGE_CONTRACTS` graph endpoint, config form, validation, YAML download, save/delete operator presets, enqueue hand-off | ✅ Done (endpoints + screen; landed across commits — graph, then form, then save/delete presets + enqueue hand-off) |
| 10 | **Resources Console** | Connection cards, credential masking, test actions, fleet + queue-depth state | ✅ Done (endpoints + screen) |
| 11 | **Execution controls** | `--audit-only`, configure-and-run, per-stage runs, log streaming | ✅ Done (endpoints + screen; configure-and-run enqueues, capability banner names the missing piece — "log streaming" is the Dashboard's queue/checkpoint feed, not duplicated here) |

**Remaining work is one control (7)** — the `ReportIssue` action attached to
the inspector screens, over an endpoint that already exists and is tested.
The next merge is frontend-only: wire that control onto the Corpus and Chunk
Inspectors.

Merge 9 came in two commits — the DAG, then the config form over it — because
a screen carrying both a graph renderer and a recursive JSON-Schema form does
not fit the 500-line cap. Each stands alone: the first leaves a composer that
shows the pipeline, the second makes it edit one.

The SPA build stage moved from merge 3 to merge 4 when 3 was built: a build
stage for a directory that does not exist yet cannot build, and defining one
that nothing references is dead weight. It lands with `ui/`, where it can be
exercised.

## 6. Decisions

| Decision | Resolution |
|---|---|
| Repo layout | `ui/` lives in this repo. Same project, separate containers; the image build needs both halves |
| Frontend | **Svelte 5 + SvelteKit**, matching DeepCivic (its `DESIGN.md` uses runes — `$effect`). Node is a *build-time* tool only: Vite compiles the app to static JS/CSS and the sidecar serves those files from FastAPI. **No JavaScript runtime ships in the image** |
| CI | A Node job lints and builds the SPA on changes under `ui/`, independent of the Python matrix, so a frontend break never masks a pipeline break |
| SPA delivery | Built during the image build, not vendored into the wheel |
| Remote reads | In scope from merge 2 — a cloud sidecar has no filesystem to read |
| Feedback store | One file per report, sibling of `runs/`, same layout local and remote (§4). Operator-saved presets take the same shape: one file each, a store `presets/` prefix (sibling of `runs/`/`feedback/`) or a local `--presets-dir`, so a store-backed console needs no writable mount |
| Settings | Editable via a shared writable config volume; secrets env-only; restart activates |
| Auth | **None.** Not deployed discoverably at any layer. `reported_by` is advisory |
| Execution | The console covers the full designed workflow, calling the same library functions the CLI calls — no subprocess, no shell-out. Dispatch is always the queue and **on by default**. `--audit-only` switches it off for read/inspect-only deployments |

One deferred item: the plan assumes a small number of trusted operators.
Multi-user means auth, per-user preferences and a real audit log — a larger
project, and the point at which `reported_by` must be verified, not declared.
