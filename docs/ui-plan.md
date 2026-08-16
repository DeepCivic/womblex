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

Two consequences worth stating up front:

- **No new schemas for the UI's benefit.** If a screen needs a number that isn't
  written today, the fix is a library change that the CLI also gets (§4), not a
  UI-only side table.
- **Read-only by default.** Execution controls ship behind an explicit
  `--allow-execute` flag. A console that can start jobs is a different security
  object from one that can only look at parquet.

## 2. Architecture

```
womblex[ui]  →  fastapi + uvicorn        (no new core dependencies)

src/womblex/ui/
├── app.py         # FastAPI app factory; mounts the built SPA as static files
├── deps.py        # Resolve config / output_root / optional JobQueue per request
├── routes/        # runs, corpus, chunks, stages, resources, jobs
└── readers.py     # Thin pyarrow readers over the shard sidecars

ui/                # SvelteKit SPA, static-adapter build, output vendored at
                   # src/womblex/ui/static/ so `pip install womblex[ui]` is enough

src/womblex/cli/ui.py   # `womblex ui --config … [--port 8080] [--allow-execute]`
```

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
| **Dashboard** | `JobQueue.stats()`; per-stage `CheckpointState` JSON; shard mtimes | Queue counts are exact. Throughput is derived from batch completion times; CPU/memory is **not** available — see §4 |
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

## 4. Gaps — what the UI needs that does not exist

Honest accounting. Each is a library change, sized, and each stands alone.

| Gap | Needed by | Proposal | Size |
|---|---|---|---|
| **No run index API** | Every screen (run selector) | `store/retention.list_runs()` exists; add `describe_run(path)` returning run_id, doc count, stages present, timestamps | Small |
| **No worker telemetry** | Dashboard CPU/memory/fleet status | `womblex_jobs` has no heartbeat or metrics columns. Add `heartbeat_at`, `worker_meta JSONB`; workers update on claim and completion | Medium — schema migration |
| **No local-run progress stream** | Dashboard, for `womblex run` | Checkpoint JSON updates once per batch, so local progress is batch-granular and only readable *after* a batch. Accept this: label local runs "batch-granular" rather than faking a stream | None (scope decision) |
| **Throughput is not recorded** | Dashboard | Derive from `updated_at` deltas on queue rows; for local runs, derive from checkpoint mtimes. No new schema | Small |
| **Config round-trip** | Pipeline Composer save | Pydantic gives load + validate; needs a YAML writer preserving comments, or an accepted lossy re-emit | Small–medium |
| **No auth** | Anything beyond localhost | Bind localhost-only by default; document that remote exposure is the deployer's problem | None (scope decision) |

Two things the brief implies that I recommend **against** building:

- **"Scale worker fleets (scale-to-zero)" from the Resources Console.** Womblex
  workers are processes against a Postgres queue; it has no scheduler and should
  not grow one. The console can show fleet state and link out to whatever runs
  the containers. Owning autoscaling here would be exactly the "corpus hosting
  custom code" anti-pattern, one level up.
- **Editing chunks or PII masks in the Chunk Inspector.** The brief asks for
  *review* of masking effectiveness and boundary accuracy — read plus flag, not
  write. Masking is terminal by design; a UI that rewrites `clean_text.parquet`
  would put a second, unversioned producer on an output the pipeline owns.
  Flags should land as an audit sidecar, if they land at all.

## 5. Delivery sequence

Sized to the 500-changed-line merge cap. Each merge stands alone and leaves the
tree green.

| # | Merge | Contents |
|---|---|---|
| 1 | **Design system** | `DESIGN.md` + this plan *(this change)* |
| 2 | **Read API skeleton** | `[ui]` extra, `src/womblex/ui/app.py`, `womblex ui` command, `/api/runs` + `/api/runs/{id}/manifest`, tests against a fixture shard dir |
| 3 | **Frontend shell** | `ui/` SvelteKit workspace, tokens from `DESIGN.md`, top bar + side nav, run selector, theme + density toggles |
| 4 | **Corpus Inspector** | Documents grid, lifecycle-checkpoint switcher, failure filter, `verify-shards` action |
| 5 | **Chunk Inspector** | Chunk reader endpoints, `ChunkCard`, entity/PII/money overlays |
| 6 | **Dashboard (queue)** | Queue stats, job list, stale detection, KPI tiles and throughput |
| 7 | **Worker telemetry** | Queue schema migration + worker heartbeat *(library change; lands before or after 6, independent)* |
| 8 | **Pipeline Composer** | `STAGE_CONTRACTS` graph endpoint, config form, validation, YAML round-trip |
| 9 | **Resources Console** | Connection cards, credential masking, test actions |
| 10 | **Execution controls** | `--allow-execute`, job submission, log streaming |

Merges 2–5 deliver a genuinely useful auditing console. Everything from 6 on is
additive; the sequence can stop at any point without leaving a half-built screen.

## 6. Open decisions

These change the work materially and are the user's call, not mine:

1. **Ship the built SPA in the wheel, or require a Node build?** Vendoring built
   assets makes `pip install womblex[ui]` sufficient and keeps the console
   installable in an air-gapped environment; it also puts build output in the
   repo. Recommendation: vendor it, in its own commit, excluded from the merge
   cap the way `fixtures/` is.
2. **Separate repo for `ui/`, or a workspace in this one?** Same-repo keeps the
   API and its client honest with each other. Recommendation: same repo.
3. **Does the console need to read remote (S3) runs, or only local paths?**
   `store/remote.py` makes remote reads possible; doing it in a request path
   raises latency and credential questions. Recommendation: local paths for
   merges 2–5, remote behind a flag afterwards.
4. **Multi-user, or single-operator?** The plan assumes single-operator on
   localhost. Multi-user means auth, per-user preferences and an audit log —
   a substantially larger project.
