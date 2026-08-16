# Womblex Console — UI Plan

A proposal for the optional Womblex UI. Visual system: [`../DESIGN.md`](../DESIGN.md).
Screen requirements: the *Womblex Platform UX Requirements* brief (five domains —
Dashboard, Pipeline Composer, Corpus Inspector, Semantic Chunk Inspector,
Resources Console).

Status: **proposal, nothing implemented.**

---

## 1. The governing principle

**The console reads artefacts the pipeline already writes. It adds no pipeline
logic.**

This is the [CLAUDE.md](../CLAUDE.md) thin-adapter rule applied to a new surface,
and the corpus-vs-library rule pointed the other way: the UI is *configuration,
invocation and output formatting* over library functions. Any iteration,
aggregation or orchestration it appears to need belongs in Womblex proper, behind
a normal Python API that the CLI can call too.

Three consequences worth stating up front:

- **v1 surfaces only what already exists.** No new schemas, no new columns, no
  new instrumentation for the UI's benefit. Where a screen wants a number the
  system does not record, the screen shows less — it does not grow a side table.
  §4 is the honest list of what that costs.
- **No writes to stage outputs.** Reviewers report problems; they do not edit
  parquet (§4).
- **Read-only by default.** Execution controls ship behind an explicit
  `--allow-execute` flag. A console that can enqueue jobs is a different
  security object from one that can only look at parquet.

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
├── routes/        # runs, corpus, chunks, stages, resources, jobs
└── readers.py     # Thin pyarrow readers over the shard sidecars

ui/                # SvelteKit SPA; built during the image build, not vendored
src/womblex/cli/ui.py   # `womblex ui [--port 8080] [--allow-execute]`
```

State reaches the sidecar exactly the way it reaches a worker — no new
configuration surface:

| Deployment | Runs | Queue |
|---|---|---|
| Local | `output_root` bind-mounted read-only | Absent; dashboard falls back to checkpoints |
| Cloud | `WOMBLEX_STORE_URI` + `WOMBLEX_S3_ENDPOINT` via `store/remote.py` | `WOMBLEX_DB_DSN` via `JobQueue` |

```yaml
  ui:                       # added to docker-compose.yml
    build: {context: ., dockerfile: Dockerfile.ui}
    environment: *cloud-env
    ports: ["8080:8080"]
    command: ["ui", "--port", "8080"]
```

Because the cloud sidecar reads a bucket rather than a filesystem, **remote
reads are in scope from the first merge**, not deferred behind a flag. The
mount is read-only in both shapes; the console has no code path that writes to
a run.

Frontend stack matches DeepCivic exactly — SvelteKit, Tailwind v4 `@theme inline`
tokens, shadcn-svelte, svelte-motion, Material Symbols — so components and token
names port between the two codebases. The one divergence is that fonts and icons
are **self-hosted in the bundle**, not fetched from Google Fonts, because the
pipeline's no-network-at-runtime property should not be broken by its own UI.

`[ui]` sits alongside the existing deployment-shaped extras (`[local]`,
`[cloud]`, `[cloud-ocr]`) and, like `[cloud]`, must not pull boto3.

## 3. Screen → data source

Every screen maps onto artefacts that exist today. This is the plan's main
finding: the pipeline is already instrumented for this UI, largely by accident of
having been built shard-first.

| Screen | Reads | Notes |
|---|---|---|
| **Dashboard** | `JobQueue.stats()`; per-stage `CheckpointState` JSON; shard mtimes | Queue counts are exact. Throughput is derived from batch completion times; CPU/memory is **not** recorded and v1 does not show it — see §4 |
| **Pipeline Composer** | `config.py` Pydantic models; `cloud/stage_contracts.py` | The composer is a form over the config models plus a graph drawn from `STAGE_CONTRACTS` |
| **Corpus Inspector** | `<run_id>/manifest.parquet`; `store/shard_audit.audit_shard_directory()` | `MANIFEST_SCHEMA` is already the documents table: `doc_id`, `filename`, `ext`, `extraction_method`, element/cell/field counts, `status`, `error` |
| **Chunk Inspector** | `*.chunks.parquet`, `*.enrichment_entities.parquet`, `*.graph_edges.parquet`, `*.pii_spans.parquet`, `*.clean_text.parquet`, `*.money_spans.parquet`, `*.chunk_quality.parquet` | All joinable on `source_hash` (+ `chunk_index`) |
| **Resources Console** | `store/remote.storage_options_from_env()`, `is_remote_uri`; `utils/isaacus_client.unserved_models()`; `JobQueue` connectivity | Connection *testing* already exists as library code |

Three of these deserve emphasis:

**`STAGE_CONTRACTS` is the Pipeline Composer's graph.** Each `StageContract`
already declares `required_inputs`, config-derived `conditional_inputs`,
`outputs`, `scope`, `mutation`, and whether it `needs_isaacus_api`. The
requirement that the composer be "the only place logical guardrails are enforced
(e.g. ensuring extraction precedes chunking)" is satisfiable by rendering that
existing structure — the guardrail is `chunk.required_inputs` naming the
elements sidecar, not a rule re-typed in TypeScript. **Do not hand-code the DAG
in the frontend**; serve it from `STAGE_CONTRACTS` so it cannot drift.

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

Two smaller scope decisions in the same spirit: the **Pipeline Composer** loads
and validates config through the existing Pydantic models and offers the result
as a YAML download — the server does not write config files. **Auth** is out of
scope; the sidecar binds inside the deployment's network and exposure is the
deployer's problem, as it already is for Postgres and MinIO.

The one genuinely missing read is a **run index**: `store/retention.list_runs()`
returns paths, and the run selector wants run_id, document count, stages present
and timestamps. That is a small `describe_run()` helper the CLI benefits from
too.

### Scale-to-zero is in scope, and already supported

Womblex is event-driven by construction, and the console should reflect that.
`run_worker()` already takes `once` and `idle_timeout`: a worker that finds an
empty queue exits, and the container goes away. Enqueue is the event; drain is
the shutdown signal. Cold start on the next enqueue is acceptable — it is the
same latency character as invoking the CLI locally, which is how the pipeline is
used today.

So the Resources Console shows fleet state (workers holding batches, queue
depth, idle timeout) and can enqueue work, which is what causes workers to come
up. Womblex still does not implement a scheduler: the platform starts containers
— compose `--scale`, a Kubernetes Job, an ECS task — and Womblex's contribution
is a worker that knows how to exit. That division is what makes scale-to-zero
free rather than a feature.

### Reporting a bad record, instead of editing it

No screen writes to a stage output. Masking is terminal by design and the
pipeline owns its parquet; a console that rewrote `clean_text.parquet` would be
a second, unversioned producer of it.

Instead, any record in the Corpus Inspector or Chunk Inspector carries a
**report action**. Reporting appends one JSON line to a feedback log —
`<run_root>/feedback.jsonl`, or the store equivalent — containing the row
itself plus the reviewer's note:

```json
{"reported_at": "2026-08-16T04:11:09Z", "run_id": "run-20260816-0353",
 "record_type": "chunk", "source_hash": "9f2c…", "chunk_index": 47,
 "row": {"text": "…", "content_type": "narrative", "has_redaction": true},
 "note": "PERSON mask missed a signature block"}
```

Append-only, human-readable, outside the shard schemas, and useful as an
evaluation input later. The row is embedded rather than referenced so the log
stays meaningful after a re-run replaces the shard.

## 5. Delivery sequence

Sized to the 500-changed-line merge cap. Each merge stands alone and leaves the
tree green.

| # | Merge | Contents |
|---|---|---|
| 1 | **Design system** | `DESIGN.md` + this plan *(this change)* |
| 2 | **Read API skeleton** | `[ui]` extra, `src/womblex/ui/app.py`, `womblex ui` command, `describe_run()`, `/api/runs` + `/api/runs/{id}/manifest`, local and store-backed, tests against a fixture shard dir |
| 3 | **Sidecar image** | `Dockerfile.ui`, the compose `ui` service, read-only mount, SPA build stage |
| 4 | **Frontend shell** | `ui/` SvelteKit workspace, tokens from `DESIGN.md`, top bar + side nav, run selector, theme + density toggles |
| 5 | **Corpus Inspector** | Documents grid, lifecycle-checkpoint switcher, failure filter, `verify-shards` action |
| 6 | **Chunk Inspector** | Chunk reader endpoints, `ChunkCard`, entity/PII/money overlays |
| 7 | **Report action** | `ReportIssue` control + append-only `feedback.jsonl` writer, both inspectors |
| 8 | **Dashboard** | Queue stats, job list, stale detection, fleet view from `locked_by`, KPI tiles and throughput |
| 9 | **Pipeline Composer** | `STAGE_CONTRACTS` graph endpoint, config form, validation, YAML download |
| 10 | **Resources Console** | Connection cards, credential masking, test actions, fleet + queue-depth state |
| 11 | **Execution controls** | `--allow-execute`, enqueue, log streaming |

Merges 2–7 deliver a genuinely useful auditing console. Everything from 8 on is
additive; the sequence can stop at any point without leaving a half-built screen.

## 6. Open decisions

Two remain — both change the work materially and are the user's call, not mine:

1. **Separate repo for `ui/`, or a workspace in this one?** Same-repo keeps the
   API and its client honest with each other, and the sidecar image build needs
   both. Recommendation: same repo.
2. **Multi-user, or single-operator?** The plan assumes one operator reaching a
   sidecar inside a trusted network. Multi-user means auth, per-user preferences
   and an audit log — a substantially larger project, and the point at which
   `feedback.jsonl` needs an author field.

Settled by the sidecar decision: the SPA is built during the image build rather
than vendored into the wheel, and remote (S3) reads are in scope from merge 2,
because a cloud sidecar has no filesystem to read.
