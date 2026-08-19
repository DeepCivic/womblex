# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Entries are terse by design; rationale lives in the PR/commit history.

## [Unreleased]

## [0.5.9] - 2026-08-18
Minor, additive. Headline fix: the SageMaker/MinIO credential conflict that
403'd Isaacus-on-SageMaker `/invocations` on an EC2 instance role — the object
store now takes its own credential env vars so the ambient `AWS_ACCESS_KEY_ID`
can stay unset. Also completes `docs/ui-ingest-plan.md` merges 4–5 (Composer
owns dispatch; run logs) and adds an operator-editable S3 credential override
to the Resources Console. No parquet schema changed.

### Added
- **Store-specific S3 credentials (`WOMBLEX_S3_ACCESS_KEY_ID` / `WOMBLEX_S3_SECRET_ACCESS_KEY`) — fixes the SageMaker/MinIO 403.** `AWS_ACCESS_KEY_ID` is process-global, and boto3 checks env vars *before* the EC2 instance role — so setting it for MinIO/s3fs disabled the instance role for the `isaacus-sagemaker` SigV4 signer, which then signed the MinIO key against real SageMaker and got a 403 (`security token … is invalid`); the reverse broke MinIO with `InvalidAccessKeyId`. `store/remote.storage_options_from_env()` now reads the store-specific `WOMBLEX_S3_ACCESS_KEY_ID` / `WOMBLEX_S3_SECRET_ACCESS_KEY` first, falling back to `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` for back-compatibility — so s3fs gets MinIO's key explicitly while `AWS_ACCESS_KEY_ID` stays unset and boto3's chain reaches the instance role for SageMaker. `docker-compose.yml` sets `WOMBLEX_S3_ACCESS_KEY_ID`/`_SECRET` (defaulting to MinIO's static key) and defaults `AWS_ACCESS_KEY_ID`/`_SECRET` to **empty**; a cloud deployment on an instance role can leave both unset (s3fs then uses the role for S3 too). New README "SageMaker credentials and the MinIO conflict" section and updated cloud-deployment env block. `AWS_ENDPOINT_URL` is now documented as the `WOMBLEX_S3_ENDPOINT` fallback. Back-compatible: an existing deployment that only sets `AWS_*` keeps working unchanged.
- **Console Resources Console — operator-editable S3 credential override.** When the baked-in `WOMBLEX_S3_*` key is rotated, an operator can save the new pair through the Run store card (→ S3 credentials) rather than rebuilding the container: `RemoteStore.from_uri(..., credentials=(key, secret))` threads an explicit override that wins over the env, persisted to the settings volume (`--settings-dir` / `$WOMBLEX_UI_SETTINGS_DIR`) via `SavedLocations`. It overrides only the console's own store reads (and its enqueue/preflight against the ingest store); pipeline **workers** still read their keys from the env at start-up. The saved secret never appears in a response or log — the card reports only `credentials_source` (`saved`/`env`) and the access-key id masked to its last four. `PUT /api/resources/locations` gains `s3_access_key_id`/`s3_secret_access_key`/`clear_credentials`; because the response masks the secret, omitting both fields *keeps* the saved pair (preserve-on-omit), passing both sets a new pair, `clear_credentials` reverts to the env, and a half-set pair is refused (400). Because the settings volume now holds a credential, treat it as sensitive.
- **Console run logs — readable and downloadable (`docs/ui-ingest-plan.md` merge 5).** Batch logs the pipeline already tees to `runs/<run_id>/logs/batch-NNNN.log` (worker + `cmd_run`, via `utils/run_log.capture_batch_log`) are now listable and readable through the console, with the same local/remote fork every reader uses. A run predating this change lists empty rather than 404-ing; log names are validated (`batch-NNNN.log`) before any path join so the endpoint cannot probe outside the `logs/` prefix.

### Changed
- **Console: the Pipeline Composer owns dispatch; the Execution Controls screen is removed.** `docs/ui-ingest-plan.md` merge 4. The composer is now the console's one dispatch surface — it enqueues over the deployment's whole configured ingest location, so it re-inputs no prefix (ingest/output are Resources-Console settings). Its enqueue section gains a read-only deployment-locations strip (ingest / output / queue + "N documents ready" from `GET /api/execute/ingest`) and the batching controls moved across from the deleted screen: batch size is now a 10/50/100 select (server stays `ge=1`, so `womblex enqueue --batch-size` keeps full range) and max attempts carries over. The `blocker` banner moves here too, extended with the `no_ingest` case. `ui/src/routes/execute` and its `NAV_ITEMS` entry are deleted (`/execute` falls through to the SPA index); `api.ts`'s `ExecutionStatus` gains `has_ingest`/`ingest_uri`/`output_uri` and `EnqueueRequest.input_prefix` becomes optional. Net-negative frontend change.

### Fixed
- **Ingest listing reaches nested prefixes.** Object stores have a flat keyspace, so `inbox/2026-08/foo.pdf` is one key, not a file in a folder. `RemoteStore.list_files` gains `recursive=`, and the three sites that enumerate *documents* (`womblex enqueue`, the console's enqueue, `GET /api/execute/ingest`) use it — a dated or per-agency upload layout previously reported zero documents ready, with no prefix field on any screen to reach them.
- **Locations are parsed before they are saved.** `store/remote.validate_location_uri` refuses `s3:/bucket` (one slash, which fsspec reads as a *relative local path*), `S3://`, an unsupported scheme and a bucket-less `s3://`; `PUT /api/resources/locations` returns 400 instead of persisting a location that silently becomes a folder named `s3:`. Validation is string-first, so it never resolves a hostname.
- **Ingest-root refusal no longer trips on spelling, or burns the retry budget.** The worker compares normalised `store_root()` tuples (`same_location`), so an enqueue flag and a compose env var differing by a trailing slash do not refuse every job. A genuine mismatch calls the new `JobQueue.release()` — back to `pending`, reason recorded, attempt not consumed — and backs off by `poll_interval`, instead of `fail()`ing the batch and re-claiming it in a tight loop until it died.
- **`enqueue` checks disjointness against the prefix shards actually land under**, not a hardcoded `runs/`: `--output-prefix inbox/out` alongside `--ingest .../inbox` passed the guard and then wrote shards into the ingest.
- **A saved location override that stops validating degrades to the flag/env defaults** with a warning rather than 500ing every console request, matching the skip-and-continue an unparseable file already had. Start-up still fails hard, now naming the file to delete. `save_locations` also checks an ingest nested inside a local `output_root`, which had no overlap check at all.


## [0.5.8] - 2026-08-18
Library + CLI, back-compatible. `docs/ui-ingest-plan.md` merge 1: source
documents can now live at their own object-store/local location, separate
from the output store. No parquet schema changed; `womblex_jobs` gains one
nullable column.

### Added
- **Ingest as a distinct store (library + CLI).** `--ingest` / `$WOMBLEX_INGEST_URI` on `womblex enqueue` and `womblex worker`, back-compatible (omitted ⇒ today's single-store behaviour). `store/remote.py` gains `store_root()` and `assert_disjoint_locations()` — the single enforcement point requiring the ingest location and the store's effective `runs/` output to be on disjoint paths (same bucket, different folders is fine; either containing the other is a hard fail, checked at enqueue). `womblex_jobs` gains a nullable `ingest_root` column (`JobSpec`/`Job` carry it); a worker refuses a claimed job whose `ingest_root` differs from its own, naming both roots, rather than failing per file. Job failures now record the exception type, message, and the worker's ingest root instead of a bare `repr(e)`. `enqueue --input-prefix` is now optional — the whole ingest root is the default scope. `docker-compose.yml` and the README document the bundled stack's `inbox/` (ingest) vs `runs/` (output) prefixes of the one `womblex` bucket.

## [0.5.7] - 2026-08-18
Compose-only, additive. `docker-compose.yml` now serves two deployments from
one file: the self-contained local stack (unchanged when no env is set) and a
cloud deployment pointed at externally-provided Postgres + externally-provided
S3, with no code change. No application code or parquet schema changed (only
the version bump in `pyproject.toml`).

### Added
- **`docker-compose.yml` cloud profile — external Postgres + external S3 with no code change.** Every connection value in the compose file is now `${VAR:-<bundled default>}` (`WOMBLEX_DB_DSN`, `WOMBLEX_STORE_URI`, `WOMBLEX_S3_ENDPOINT`, `AWS_*`): unset env resolves to today's self-contained local stack byte-for-byte; set env points every service at external services. The bundled `postgres`/`minio`/`createbuckets` moved behind a `local` compose profile, so a plain `docker compose up worker` does not start them, and `init`/`worker`/`womblex`/`seed-demo`/`ui` declare their dependency on those bundled backends as `required: false` — so a cloud deployment (external services, bundled backends never started) is not blocked by a missing dependency, while the local stack still health-gates as before. The compose header documents both invocations and the one mechanical edit for engines older than Compose 2.20.0. `WOMBLEX_STORE_URI` accepts a path prefix (`s3://<bucket>/<prefix>`), so Womblex keeps its output in its own folder of a shared bucket; embeddings are published as `*.embeddings.parquet` in the object store and never written to Postgres, so `pgvector` is a property of the shared database for other consumers, not a Womblex requirement. New README "Cloud deployment" section walks the three-env-var invocation and the `init` / `psql -f sql/womblex_jobs.sql` table creation; `TestSidecarImage` gains coverage that the bundled backends sit behind the `local` profile, that cloud services do not hard-depend on them, and that the connection env is overridable.

## [0.5.6] - 2026-08-18
Minor, additive. Console reliability on a fresh/shared cluster: a reachable
queue whose `womblex_jobs` table does not exist yet reads as *empty*, not
"unreachable", and the compose `ui` service waits for Postgres/MinIO to be
healthy before it comes up. `DEFAULT-Isaacus` now enables embeddings so the
preset, config file and vendored demo run are the one shape. Ships
`sql/womblex_jobs.sql`, the DBA-reviewable schema artefact for provisioning the
one table Womblex owns into a shared or externally-managed database. No parquet
schema changed.

### Added
- **`sql/womblex_jobs.sql` — the DBA-reviewable schema artefact.** The one table Womblex owns (`womblex_jobs` + its claim index) now has a checked-in `.sql` file, so an operator provisioning a shared or externally-managed database can `psql "$WOMBLEX_DB_DSN" -f sql/womblex_jobs.sql` (and review the DDL / apply grants) instead of reading it out of Python or running `womblex jobs --create-schema`. It is byte-for-byte the DDL `ensure_schema` runs, pinned by `tests/test_cloud.py`. Every statement is `IF NOT EXISTS` and scoped to this one table (no DROP/TRUNCATE, no CREATE DATABASE/SCHEMA, no `search_path`), so Womblex coexists with another system's tables in the same database — it does not need a database of its own.

### Fixed
- **Console Dashboard/Execution: a reachable-but-schemaless queue now reads as empty, not "unreachable".** A fresh Postgres in which `init` (or the first enqueue) has not yet created the `womblex_jobs` table now shows as an *empty* queue instead of a fault. Previously every dashboard read (`stats`/`list_jobs`/`workers`/…) raised `UndefinedTable`, which `queue_section` reported as `queue_error` — and because the whole dashboard/execution surface is gated on the queue, a fresh cluster read as broken (with the intermittent Composer 500s / failed-to-fetch that came from the same racing-startup window) until the table happened to appear. `dashboard.queue_section` now maps `psycopg.errors.UndefinedTable` (SQLSTATE 42P01) to the empty-queue payload and never creates the table (the console is read-only; `init`/enqueue own creation); a genuine failure (connection, permissions) still surfaces as `queue_error`.
- **`docker compose up ui` no longer races Postgres/MinIO startup.** The `ui` service advertises `WOMBLEX_DB_DSN` and `WOMBLEX_STORE_URI` but depended only on `createbuckets`, so it came up before Postgres/MinIO were answering — the reported "queue unreachable" and intermittent Composer 500s. It now `depends_on` both `postgres` and `minio` being `service_healthy`. The sample-corpus quickstart comment now brings up `postgres`/`init` too, so the console sees a reachable (empty) queue rather than an unreachable one.
- **`DEFAULT-Isaacus` preset and `configs/default-isaacus.yaml` now enable embeddings.** The reference shape is `extract → chunk → enrich → build_graph → embed → money`, and the vendored demo run (`run-throsby-demo`) already carries `*.embeddings.parquet` — but the preset/config left `embedding.enabled: false`, so the composer's DEFAULT-Isaacus disagreed with the sample corpus it is meant to mirror. `embedding.enabled: true` in both, with `task`/`dimensions` left at the model defaults. Preset, config file and demo run are now the one shape (pinned by `test_config_file_and_ui_preset_agree` and a new demo-shape test).

## [0.5.5] - 2026-08-18
Build-only. Bakes the known-good `pip install --upgrade pip` step into both
Dockerfiles so images build reproducibly from the tag, ending the practice of
patching the Dockerfile on the instance (which produced a v0.5.4 image that
corresponded to no committed source). No behaviour, dependency or schema
change.

### Fixed
- **Docker builds no longer need an instance-local pip patch.** The stock pip on `python:3.11-slim` is old enough that its resolver stalls/fails on the boto stack — `s3fs` -> `aiobotocore` pins a narrow `botocore` range that must co-resolve with `boto3`'s own `botocore` pin. Both `Dockerfile` and `Dockerfile.ui` now run `pip install --upgrade pip` before the main install, so the newer resolver finds the compatible set. Previously an engineer added this line by hand on the instance, so the built image bypassed the clean Dockerfile from the tag.

### Changed
- **`docker-compose.yml` names both images explicitly.** The pipeline/worker services (`init`, `womblex`, `worker`, `seed-demo`) now spell out `build: {context: ., dockerfile: Dockerfile}` instead of the bare `build: .`, so the two-image split (plain `Dockerfile` for workers, `Dockerfile.ui` for the console) reads at a glance. No build behaviour change.

## [0.5.4] - 2026-08-18
Tooling-only: clears pre-existing lint/type noise on the base tree (files this
change never touched otherwise). One `chmod`, a few mypy per-module overrides,
one stale `type: ignore` code corrected. No behaviour, dependency or schema
change; `ruff check src/` and `mypy src/` both clean.

### Fixed
- **Ruff EXE002 on `src/womblex/utils/__init__.py`.** The package `__init__.py` carried the executable bit with no shebang — it is an importable module, not a script, so the bit is dropped (`chmod -x`) rather than the rule ignored.
- **Mypy `import-not-found` for third-party libraries without stubs.** `psycopg` (+ `psycopg.types.json`), `torch` and `ultralytics` ship no type stubs, so mypy errored on their imports (`cloud/queue.py`, `ingest/paddle_ocr.py`). Added `ignore_missing_imports = true` per-module overrides for `psycopg.*`, `torch.*` and `ultralytics.*` in `pyproject.toml`, matching the existing `boto3.*`/`fitz.*` pattern.
- **Stale `type: ignore` code on `ingest/paddle_ocr.py`.** The `from ultralytics import YOLO` line carried `# type: ignore[import-untyped]`; mypy reports the import as `import-not-found`, so the code is corrected to match.

## [0.5.3] - 2026-08-18
Dependency-only. `isaacus`, `isaacus-sagemaker` and the AWS SDK (`boto3`) move
into the base install; the `[isaacus]`, `[bedrock]` and `[cloud-ocr]` extras are
removed. Also fixes the console Resources card misreading a misspelled SageMaker
endpoints var as a bare "No API key". No parquet schema changed.

### Changed
- **Isaacus SDK, hosted-VLM (Bedrock) OCR and the AWS SDK are now core dependencies.** `isaacus`, `isaacus-sagemaker` and `boto3` move out of the `[isaacus]`/`[bedrock]`/`[cloud-ocr]` extras into the base install — every real deployment uses enrichment/embeddings and (often) hosted OCR or SageMaker, they are tiny next to the vision/ML stack already shipped, and gating them behind extras only produced misconfiguration (a missing SDK surfaced in the console as a bare "No API key"). They stay dormant until configured (no key / no `ISAACUS_SAGEMAKER_ENDPOINTS` / no `mistral-ocr` engine). The `[isaacus]`, `[bedrock]` and `[cloud-ocr]` extras are removed; remaining extras are `[local]` (empty), `[cloud]`, `[ui]`, `[dev]`. Dead `pip install womblex[…]` ImportError guards around first-party modules (enrich/persist/pii) simplified to direct imports; the stale `[pii]` extra reference is gone (presidio/sentence-transformers were already core).

### Fixed
- **Console Resources card: a misspelled SageMaker endpoints var now reads as a fixable cause, not "No API key".** Setting the singular `ISAACUS_SAGEMAKER_ENDPOINT` (or other near-misses) silently fell back to the hosted API, so the Isaacus card showed a bare "No API key" with no hint at the cause. `misconfigured_endpoints_var()` (network-free) detects the near-miss; the card gains an `endpoints_typo` field and the screen names the offending variable and the canonical `ISAACUS_SAGEMAKER_ENDPOINTS` (plural).

## [0.5.2] - 2026-08-18
Minor, additive. Console-focused: the Pipeline Composer gains save/enqueue of operator presets (store-backed in cloud mode), the Dashboard screen lands, and execution is on by default (`--audit-only` is the new opt-out). Also fixes console stage-presence reporting `enrich` empty after enrich ran. No parquet schema changed.

### Fixed
- **Console stage-presence read `enrich` empty even after enrich ran.** `_scan_stage_presence` (`ui/readers.py`) scanned a `source_hash` column, but the sharded enrichment sidecar carries the source_hash in `document_id` (as the Chunk Inspector's own path already handles). Presence now reads the column each sidecar actually carries, derived from the same `_CHUNK_DETAIL_SIDECARS` table so the two cannot drift; values still returned under `source_hash`. `GET /stage-presence/enrich` (and the Corpus Inspector's checkpoint overlay) now reflect enrich.

### Added
- **Console Pipeline Composer — save/enqueue presets (frontend).** The composer can now save the composed config as a named preset, delete one it saved (never a built-in), and hand a composed run off to the queue — `paths.input_root` → `input_prefix` (confirmed, since it may be absolute/local while the prefix is store-relative) and `dataset.run_id` → `run_id`, through the same enqueue the Execution Controls use (shown only when the console can dispatch). `savePreset`/`deletePreset` raise `SavePresetRefused` carrying the HTTP status (409 no writable location, 400 bad name/overlay) so the UI hides saving where it cannot write rather than failing repeatedly.
- **Console Pipeline Composer — operator presets are store-backed in cloud mode.** Saved presets now take the same local-vs-store shape as feedback (`ui/readers.py` owns the split): store-backed consoles write/list/delete under the object store's own `presets/` prefix (a sibling of `runs/`/`feedback/`), so a `read_only` compose `ui` service can save presets with no writable mount. Local mode still uses `--presets-dir` (`$WOMBLEX_UI_PRESETS_DIR`). `UISettings.presets_writable` (remote, or a local presets dir) gates the 409; `presets.py` keeps only the format (filename/bytes/parse), `RemoteStore` gains `read_text`/`delete`.
- **Console Pipeline Composer — save/delete operator presets (backend; merge 9).** `POST`/`DELETE /api/composer/presets` file one JSON per preset; `GET /presets` merges built-ins with saved (`source: builtin|saved`, a saved name shadows). `dataset`/`paths` stripped on save (a preset is an overlay), overlay validated via the same `WomblexConfig(**raw)` the built-ins use; 409 where the console cannot write, 400 on an unsafe name / non-loadable overlay.

### Changed
- **Console: execution is on by default; `--audit-only` is the opt-out.** Inverts the merge-11 switch — the console can dispatch runs into the queue without a flag, and `womblex ui --audit-only` gives a pure read/inspect console (the old `--allow-execute` is removed). Still queue-only, so dispatch needs both a `--store` and a `--dsn`; the `/api/execute/status` payload renames `allow_execute` → `audit_only` and `ExecutionCapability.can_execute` is now `not audit_only and has_store and has_queue`.

### Added
- **Console Dashboard — screen (`docs/ui-plan.md` merge 8).** Run-scoped, self-refreshing screen over `/api/dashboard`; completes merge 8, leaving only the `ReportIssue` control (7).
  - Queue half (needs a DSN): KPI tiles over exact status counts, total and throughput; `locked_by` worker fleet; stale-job detection naming what `--stale-timeout` recovers; the `womblex_jobs` list with stale rows flagged inline.
  - Checkpoint half (always): per-stage progress from inside the selected run, batch-granular bar + lifetime-average rate.
  - Renders in both deployments — with no queue configured the checkpoint half still shows; read-only (names a stalled job, never requeues it).

## [0.5.1] - 2026-08-17

Minor, additive. Headline: the optional console (`womblex ui`) — a read-only sidecar over pipeline artefacts. Also: Isaacus on Amazon SageMaker (`ISAACUS_SAGEMAKER_ENDPOINTS`), a nullable `elem_order` on `CHUNKS_SCHEMA` (back-filled on read), and deployment extras (`[local]`, `[cloud-ocr]`). No breaking/schema break: `CHUNKS_SCHEMA` gains a nullable column, every other schema byte-identical to `0.4.0`.

### Added
- **Console Pipeline Composer — presets (`DEFAULT-Isaacus`).** Named pre-configured pipelines from a dropdown; a preset is a *partial* `WomblexConfig` (never `dataset`/`paths`) deep-merged onto the form, served as data (`ui/presets.py`, `/api/composer/presets`) and validated at import. `configs/default-isaacus.yaml` ships the same shape for the CLI, pinned against the preset.
- **Console Execution Controls — screen (merge 11).** `ui/src/routes/execute` over `/api/execute/status` + `/enqueue`: a configure-and-run form, and a banner naming the one missing piece (audit-only / no store / no queue) when dispatch is unavailable. `EnqueueRefused` carries the HTTP status (403/409/400) so the client distinguishes the three failure shapes.
- **Console Execution Controls — backend (merge 11).** `GET /api/execute/status` + `POST /api/execute/enqueue` (`ui/execute.py`), the console's first writable-to-a-run surface. Dispatch is always the queue (thin wrapper over `cmd_enqueue`'s batching); `--allow-execute` is the switch, enforced in one `_guard`; needs both a store and a DSN or refuses 403/409.
- **Console Pipeline Composer — config form (merge 9).** `SchemaForm.svelte` renders `WomblexConfig`'s JSON Schema recursively (no hand-typed mirror); Validate and YAML download go to the endpoints (server-rendered, byte-identical to `run --config`). Node `enabled` toggles write the config section's `enabled` via a served `config_section` map.
- **Console Pipeline Composer — stage graph (merge 9).** The pipeline DAG from `/api/composer/graph` plus a per-stage detail panel; nodes laid out by longest path from `extract`, ordering derived from `required_inputs` edges (not hand-coded). Also fixed the frontend CI job (svelte-check `$state(null)` narrowing; a `Map`-in-`$derived` eslint flag).
- **Console Resources Console (merge 10).** `GET /api/resources` (three connection cards) + `POST /test/store` and `/test/queue` live checks. No new detection logic — reuses `is_remote_uri`/`storage_options_from_env`, `unserved_models()`, `dashboard.queue_section`. The `GET` makes no network call; each card's test pays its own timeout.
  - **Credentials do not leave the process:** store keys reported as configured/not, DSN masked. Fixed a full DSN leak — the masker missed libpq's keyword form (`host=… password=…`), now masked along with `?password=` and short secrets.
- **Console Pipeline Composer read API (merge 9).** `/api/composer/graph`, `/schema`, `POST /validate`, `/yaml`. Graph renders `STAGE_CONTRACTS` (edges from `required_inputs` via `PRODUCER_OF`, acyclic, all reachable from `extract`); `/schema` is `model_json_schema()` verbatim; `/validate` + `/yaml` build `WomblexConfig(**raw)` like `load_config`.
  - **Fixed a typo'd-key silent-drop:** Pydantic ignores unknown keys, so `chunkng:` validated clean and vanished on render. `/validate` now reports `unknown_keys`; `/yaml` names dropped keys in a header comment. Warnings not errors (the CLI loads such a file too); walk skips free-form `dict` fields.
- **Console dashboard read API (merge 8).** `GET /api/dashboard` serves queue state + per-stage progress, no new schema. `JobQueue` gains `list_jobs()`, `workers()`, `stale_jobs()` (read-only twin of `requeue_stale`), `throughput()`; `store/checkpoint.py` gains `read_checkpoints()` reading dot-dir checkpoints inside the run (map from `STAGE_CONTRACTS`). Queue optional and orthogonal to run source; an unreachable queue reports `queue_error` not 500 (`JobQueue` gained `connect_timeout`). `run_id` join contained via `is_safe_run_id`.
- **Console report action (merge 7).** `POST /api/runs/{run_id}/feedback` — the console's only write path — files one JSON file per report (no append, no lost update) under a `feedback/<run_id>/` root that is always a *sibling* of runs. `reported_by` from a trusted header/env, advisory. `store/feedback_output.py` owns and contains the join (`is_safe_run_id`, fixing a `..` escape).
- **Console frontend shell (merge 4).** `ui/` SvelteKit workspace, built by a Node stage in `Dockerfile.ui` and served by `create_app` (no JS runtime in the image; `/api/*` still served without the SPA build). Top bar + collapsible side nav over the five domains; theme/density persisted to `localStorage`; tokens per `DESIGN.md`; self-hosted fonts. SPA catch-all serves `index.html` but 404s `/api/`, with traversal containment. Independent Node CI job (lint/check/build).
  - Accessibility measured, not inspected: two `DESIGN.md` rules don't survive its own light theme (lime as active-nav label 1.32:1 there), so lime stays a fill and the label carries state in weight/`--foreground`; recorded in `docs/decisions.md`.
- **Console sidecar image (merge 3).** `Dockerfile.ui` + a `ui` compose service — its own container beside the workers, `womblex[ui,cloud]` (no boto3), hardened `read_only` with a `tmpfs /tmp`. SvelteKit build stage deferred to merge 4 (a stage copying a not-yet-existing dir can't build). Drift-guard tests parse the `ENTRYPOINT` against the real CLI.
- **Console read API skeleton (`womblex ui`).** `[ui]` extra (fastapi+uvicorn), a `womblex ui` command, and `/api/runs` + `/api/runs/{id}/manifest`. Remote reads in scope from the start (a store-backed request stages the manifest to a temp dir and reuses the local reader). Binds one run source; loopback by default.
- **Run index: `describe_run()` + `RemoteStore.list_dirs()`.** Summarises a run as run_id/document-count/stages-present/timestamps from existing artefacts (`STAGE_SUFFIXES` maps stage→sidecar suffix); `list_dirs()` enumerates `runs/<id>/` in object storage. Groundwork for the run selector; the CLI benefits too.
- **Isaacus on Amazon SageMaker (private, air-gapped).** Every Isaacus call routes to SageMaker endpoints in the user's AWS account when `ISAACUS_SAGEMAKER_ENDPOINTS` is set (AWS-signed, no API key). Comma-separated `name[@region][=model|…]` subscriptions, per-model plus universal; undeployed model fails at client construction naming what's served. `[isaacus]` gains `isaacus-sagemaker` (pulls boto3 — why the boto3-free rule stays scoped to `[cloud]`).
- **`elem_order` document-order anchor on table chunks.** `CHUNKS_SCHEMA` gains a nullable `elem_order`, set only for `content_type='table'` chunks, so consumers recover narrative↔table order. Not a coordinate-space change (offsets untouched, no re-enrichment); `read_chunks` back-fills nulls for older shards.
- **Deployment-shaped install extras.** `[local]` (empty — base install *is* local CPU) and `[cloud-ocr]` (alias of `[bedrock]`) join `[cloud]`/`[isaacus]`/`[bedrock]`. No package versions changed.
- **README installation matrix** (deployment → extra → what it adds) plus a backend-selection table.

### Changed
- `[cloud]` documented as explicitly *not* implying `[cloud-ocr]`: s3fs reaches S3 via aiobotocore→botocore, so object-storage staging needs no boto3; boto3 stays confined to `ingest/llm_ocr.py`.

## [0.4.0] - 2026-08-06

Minor, additive (the `run-stage` command + stage contracts). One observable change: the `money` op's narrative output changes for text it already read — space-grouped thousands (`$10 000`) were stored wrong by 10³ and are now correct; `$US`-marked/worded/restated amounts now resolve. Re-run `womblex money` over any 0.3.0 shard dir (regenerated in place, nothing depends on it). No parquet schema changed; `money_spans.evidence` carries one new value (`p11`).

### Added
- **Money: financial values in narrative structure** (`docs/money-extraction.md`). Two values corrected, three recovered, no span lost. `process/money.py` split (number reading → `money_numbers.py`); re-exported, no import site changed.
  - Worded amounts (`two million dollars`, `fifty cents`; `money_words.py`), currency word required; declines ranges/years/unit-declarations.
  - Space-grouped thousands (`$10 000`, incl. NBSP/thin space); a group is exactly three digits.
  - `$US`/`$A` symbol order (`$US655.5m`); fixes the metre-pattern blocker that also hit `US$655.5m`.
  - Restatement `one million dollars ($1,000,000)` no longer double-counted or read negative; equal bracketed digit amounts left alone.
  - Signs/brackets (true minus vs en-dash range, accounting `$(1,234.50)`, `50¢`); declines a second dotted group (`$3.219.3m`).
- **`womblex run-stage` — remote per-batch shard-stage runner.** Runs a downstream `*_shards()` stage directly against object storage (generalises `finalize`; no `*_shards()` signature changed). Covers normalise/spellfix/chunk/money/enrich/embed/link/pii/graph-refresh/quality.
  - Declarative stage contracts (`cloud/stage_contracts.py`); conditional inputs/outputs resolved from config, not stage name.
  - Every declared output verified before any is uploaded; skip only when all present; idempotent; exits 1 on nothing-to-do.
  - `graph-refresh` modelled as in-place mutator (never skipped); `quality` run-scoped (single pass for cross-batch dedup); Isaacus-needing stages fail non-zero.
  - Stage *ordering* is the caller's; a base with absent required inputs is not-ready, all-absent is an ordering error (exit 1).

## [0.3.0] - 2026-07-29

Minor, additive (the `money` op + two sidecars; the shared table-grid algorithm). Two observable changes: a scanned page with a clean table now emits a `kind='table'` element (was a `[TABLE]` placeholder), and the unreachable `ImageExtractor` was removed. No extraction schema changed.

### Added
- **`money` annotation op** (`womblex money --shards`). Recovers amounts to `*.money_spans.parquet` + `*.money_columns.parquet`; offline, no ordering dep, never rewrites text. Exact `decimal128(38,4)`; three loci in two coordinate spaces (`narrative`/`table_cell`/`sheet_cell`), never mixed. First real run: all 42 marked ANAO narrative amounts recovered, `Approved Budget $m` reconciles three ways.
  - Self-evidencing (`money.py`): symbol/ISO/word beside the number; magnitude expansion, range linking, gated accounting negatives; AU false-positive classes rejected.
  - Column-evidenced (`money_columns.py`): number format or money-header + numeric cells; whole-word vetoes; header supplies scale/currency; continuation-row header folding.
  - Fixes: `'000` no longer matched inside any number; tier-3 ISO codes context-gated (`TOP 10`≠paʻanga); count columns `(#)` not read as money; own-currency headers survive a veto term.
- **Table-cell reconstruction on OCR'd pages (#17).** Cells reconstructed inside a layout-detected table rect on OCR pages; one shared grid algorithm, precision-gated (refuse over partial).
  - A0 — plumbing: OCR regions + render dims passed to the layout pass; region↔rect intersection + a coordinate-space guard; region-based engines only (LLM/VLM deferred).
  - A1 — shared `ingest/table_grid.py` (point-space tolerances parameterised for pixel callers) + `ingest/ocr_tables.py` `reconstruct_table` (returns `None` below gates).
  - A2/A3 — OCR-PDF path emits `kind="table"` with cells; narrative rebuilt from regions *outside* the rect (no double-count); deskewed pages refuse rather than mis-bin.
  - A4 — images already reached reconstruction via `extract_text`; the dead `ImageExtractor` deleted (breaking for direct importers; nothing internal used it). `get_extractor` signature now `(profile)`.
  - B0/B2 — GT aggregation fix (recall 25%→50% by artefact); `MIN_ROW_FILL_RATIO=0.75` density gate closes three false positives.
  - B3/B4/B5 — rendered-table benchmark (`tests/test_table_benchmark.py`) + `EXTRACTION.md`/`evaluation.md §2b` wiring; sanity asserts become build-failing gates (exact row/col counts + `MIN_CELL_MATCH`, false-table count == 0). #17 round 1 complete.

### Fixed
- **Declined continental number no longer leaks its decimal tail.** `1.234,56 EUR` came back as `56 EUR` (wrong by 10³); ambiguous numeric runs are now blocked whole. Only suffix-pattern forms leaked; international mode unaffected.
- **CI runs type-check + tests again.** Unpinned `ruff` resolved 0.16.0 whose new defaults reported 297 errors, skipping mypy/pytest (the `money` op merged untested). `ruff` bounded `>=0.16,<0.17`; tree clean under its defaults (`BLE001`/`S110`/`S112` suppressed at isolation boundaries).
- **Isaacus test suites run in CI.** CI omitted the `isaacus` extra, skipping 66 no-key tests; now installed. Unmasked and fixed a real typing error in `process/chunker.py`.
- **`mypy` passes with `openpyxl` installed** (missing `ignore_missing_imports` entry).
- **Spreadsheet extraction preserves `number_format` + numeric `value_type`.** A read-only openpyxl pass supplies both (pandas discarded them); values untouched. Matters because a register's money column is often identifiable *only* from its format (`$#,##0.00` on bare `50000`).

## [0.2.0] - 2026-07-19

### Added
- **Pre-extracted records ingest (`ingest/records.py`).** Turns clean text records (JSONL; Open Australian Legal Corpus) into the standard shard layout so the enrich→chunk→embed→graph pipeline runs over them; content-addressed `source_hash`, corpus-agnostic `RecordFieldMapping`, provenance sidecar.
- **Token-budget request packer (`utils/token_packer.py`).** Packs Isaacus requests by exact local kanon-2 token counts (limits bind on tokens, not request count); over-budget item solo, over-ceiling doc split on blank lines. Cached offline `TokenCounter`.
- **Enrichment — token-aware batching + long-doc split (`enrich_stage.py`).** Packer-driven requests (8× fewer for small docs); over-`split_ceiling` docs split and offset-merged; honours `Retry-After`. New `EnrichmentConfig` knobs (`tokenizer`, `max_texts_per_request`, `token_budget`, `split_ceiling`).
- **Graph-edge refresh stage (`analyse/graph_refresh.py`, `womblex graph-refresh`).** Offline rebuild of mention→chunk edges from entity+chunk sidecars (AI chunking runs after enrichment, so the enrich-time graph lacks them); idempotent.
- **Offline kanon-2 tokenizer** vendored under `_models/kanon-2-tokenizer`, resolved locally (no HF round-trip).
- **Distributed / cloud execution (`womblex[cloud]`).** Optional scale-out; local CPU default unchanged.
  - `store/remote.py` — fsspec stage-in/out object-storage adapter, confining remote knowledge (Path-based stages untouched).
  - `cloud/queue.py` — Postgres `FOR UPDATE SKIP LOCKED` job queue; row `status` *is* the distributed checkpoint (idempotent re-enqueue, retry, stale requeue).
  - `cloud/worker.py` + CLI (`enqueue`/`worker`/`jobs`/`finalize`); ordinary shard layout, so downstream consumes a distributed run like a local one. `process_batch` is the single shared body behind `womblex run`.
- **Container image + compose stack.** `Dockerfile` + `docker-compose.yml` bundling Postgres, MinIO and scalable workers (`--scale worker=N`).
- **CI security job.** Semgrep SAST (Python + OWASP, blocking) + `pip-audit` (informational); test job installs the `cloud` extra.
- **ABN Lookup bulk extract ingest (`ingest/abn_bulk.py`).** Stream-parses the ABR XML (~6 GB) at constant memory to `<stem>.parquet` (records) + `<stem>_names.parquet`; verbatim strings, provenance in parquet metadata, per-file failure isolation. `womblex ingest-abn`; shared MD5 helper → `utils/checksum.py`.
- **Spreadsheet preamble/header detection.** Reads `header=None` and `split_preamble` finds the header as the row starting the longest run of table-consistent rows below it; preamble kept verbatim on `sheet_meta`. Fixes fabricated `Unnamed: N` columns and ragged-CSV failures; header-first/narrow sheets unaffected.
- **Run-level document manifest.** `womblex run` consolidates per-batch manifests into `<run>/manifest.parquet` (source_hash → doc_id/filename/method/counts/status); `womblex manifest --shards` regenerates it.
- **Shippable enrichment graph.** `enrich_shards` writes `*.graph_edges.parquet` (Kanon-2 document graph flattened to `GRAPH_EDGE_SCHEMA`, `document_id`=source_hash), mapping in chunk edges when chunks exist; resume re-enriches a batch missing it.
- **`womblex chunk --shards` + `--config` combinable** (were mutually exclusive, dead-ending per-stage AI chunking); `--shards`+`--config` sources chunking settings from YAML.
- **Single-enrichment reuse for AI chunking.** With AI chunking + enrich both on, enrich persists the raw ILGS Document to `*.enrichment_doc.parquet` and chunk reuses it (no double Kanon-2 call), gated by a byte-identity guard (mismatch → self-enrich, offsets never desync). Requires enrich before chunk.
- **AI chunking pass-through (semchunk 4).** `ChunkingConfig.chunking_model` (default null) follows the enricher's structure spans; opt-in, forwards params straight to `chunkerify`. Bumps `semchunk>=4.0`.
- **`spellfix` stage — dictionary-gated OCR glyph repair (`womblex spellfix`).** Opt-in; rewrites a token only on three gates (out-of-dict trigger, single-char in-dict candidate, unique) against bundled en_AU Hunspell (`spylls`). Element-layer overlay + audit; raw untouched. Tier A digit→letter default, Tier B opt-in.
- **Composable element-text overlays via one `processing.text_source`** (`elements`|`normalised`|`spellfix`), applied before reassembly at both chunk and enrich sites (one knob — enricher input and chunk source must match).
- **Enricher `overflow_strategy` (default `auto`)** passed to `enrichments.create` (vs upstream null, which errored >16k tokens); offsets still index the full source. Fixes long FOI bundles.
- **`score --text-source={elements,normalised}`** — measure how normalisation changes CER against the same GT.
- **Benchmark: ACT-ECI labelled-pages raw-vs-normalised CER** (`-m benchmark`); regression guard asserts normalisation never worsens CER. Cohort 7→19 pages.
- **`quality` stage — chunk-quality sidecar (`womblex quality`).** `*.chunk_quality.parquet` with ML-readiness flags + duplicate cluster ids (self-contained MinHash+LSH, no datasketch); annotation only, single global pass; config-driven `boilerplate_patterns`.
- **`normalise` — `unicode_hygiene` transform** (default on): folds unicode whitespace to ASCII, strips zero-width/BOM/control; smart quotes and dashes preserved.
- **Entity-link sidecars: `womblex enrich` + `womblex link` per-stage CLIs.** `enrich` reassembles narrative and writes entity/meta sidecars (per-doc failure isolation, and does *not* checkpoint an errored doc so resume retries); `link` resolves candidates to a reference register (`*.entity_links.parquet`). Generic by design (`entity_type` discriminator, stdlib-`difflib` matcher, config-declared register roles). New `isaacus` extra. Artemis smoke 16/17.
- **`womblex embed --shards` — chunk embeddings (Kanon-2 embedder).** `*.embeddings.parquet` (one vector/chunk, 128-text batching, 429 retry, task-aware); substrate for search/clustering + a no-extraction attribution backstop.
- **`womblex pii --shards` — graph-driven detection + `<PERSON_n>` masking.** `*.pii_spans.parquet` (audit) + masked `*.clean_text.parquet`; Kanon-2 graph is the primary source, spans mapped onto chunks via `start_char`. Masking terminal (after enrich+embed, never rewriting raw chunks); regex/cosine backstop opt-in.
- **`womblex redact --shards` per-stage CLI** (dual-mode like `chunk`): `--shards --pdfs` writes `*.redactions.parquet` (detection rasterises pages, so `--pdfs` required); `--config` runs the E2E path.
- **Shard integrity scan on `--resume` (E1).** `reconcile_checkpoint_with_shards` walks each batch's four sidecars (present/non-empty/readable + manifest count sums), drops+`.corrupt`-renames failures for re-extraction; unreadable manifests logged loudly. Default on (`--no-verify-resume`).
- **`womblex verify-shards` CLI (E2).** Audits a run/shard dir for corruption + cross-batch consistency; `--compare-to` diffs two runs, `--input-dir` surfaces source-vs-manifest drift; exits 2 on corruption.
- **`run_id` + retention plumbing (I1).** Runs write to `<output_root>/<run_id>/documents/`; id resolves `--run-id` → `dataset.run_id` → auto `run-<ts>`. New `processing.retention` block (`rolling`|`keep_all`).
- **Document-layout YOLO model (DocLayNet), K7(b).** New default `yolo11n_doc_layout.pt` (MIT) replaces COCO `yolov8n.pt`; taxonomy auto-detected from class names, 11 classes mapped to `ElementKind`. Closes the 1,587-element `figure` mis-classification; per-taxonomy imgsz default.
- **`footnote` ElementKind** (added to `TEXT_KINDS`; DocLayNet `Footnote` producer; downstream stages pick it up automatically).
- **OCR form-pair bboxes, K2′.** `_extract_form_pairs_from_regions` produces `FormField`s with real positions from per-region OCR; legacy line-based path is the LLM-OCR fallback. Closes silent-zero-bbox on 4,184/5,183 form elements.

### Changed
- **`operations.py` split into an `operations/` package** (over the 750-line cap); flat import surface preserved by re-exports, behaviour-neutral.
- **Resume-integrity self-heal generalised** — `reconcile_stage_checkpoint_with_shards` now backs chunk/enrich/link/embed identically (+ `--no-verify-resume`).
- **SemChunk wrapper audit (I5).** Exposed every `chunkerify`/`__call__` param; removed the dead `ChunkingConfig.batch` flag; widened `chunk_size` to `int | None` (auto-derive), default 480 unchanged. Byte-identical output.
- **`process/chunker.py` collapsed against semchunk v3+.** Single `chunk_batch` entry point (semchunk already batches/parallelises); `TextChunk` gains nullable `page_start`/`page_end`; new reassembly/collection helpers shared by both invocation paths.
- **`operations.run_chunking` rewired through `chunk_batch`**, building `ChunkInput` from `extraction.elements` (canonical). Behaviour change: in-memory PII/redact mutations to `pages[i].text` no longer flow to chunks under `womblex run`.
- **`annotate-redactions` is now a deprecated alias** for `redact --shards --pdfs` (positional surface preserved, byte-identical output).
- **`@pytest.mark.slow` tests run by default** (backend moved to rapidocr; cohort now ~7s). Marker retained for `-m 'not slow'`.
- **Manifest schema gains `doc_id` column** (removes the implicit `stem==doc_id` coincidence); `read_manifest` back-compat derives it from `filename` for older manifests. Parser-version bump deferred (additive).

### Fixed
- **OpenCV 5 compatibility in skew detection.** `HoughLinesP` segments unpacked as `line[0]` crashed every OCR extraction under OpenCV 5's `(N,4)` layout; reshaped to `(-1,4)`, accepting both. Direct unit tests pin both shapes.
- **`mypy` no longer pins `python_version="3.11"`** (broke the 3.12 leg on numpy PEP 695 stubs); each leg checks at its own version.
- **Register manifest covers `ingest-geo`, roles from footer metadata.** `cmd_ingest_geo` never wrote a manifest and the namespace said `geo` vs `geospatial.*`; namespace now taken from the `<ns>.source_file` key, ABN roles from an `abn.role` footer key (not filename suffix). Re-run `ingest-abn` to restore the role distinction. Constant renamed `REGISTER_MANIFEST_FILENAME`.
- **`RemoteStore` no longer leaks s3fs options into non-S3 backends** — `storage_options_from_env()` now takes the URI and returns options only for `s3://`. Also: `enqueue` batch-size fallback reads `ProcessingConfig()`; worker derives its upload glob from `BatchOutcome.shard_path`.
- **Full-page scans no longer dropped from chunking as `figure` (K9-fig).** The dominant-region fallback tagged a whole page `figure` (∉ `TEXT_KINDS`), silently losing it; `_ocr_region_block_type` promotes to `paragraph` at ≥5 words. ACT-ECI: `figure` 1,200→154, all 16 zero-chunk docs now chunk.
