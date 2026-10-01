# Womblex as a backend subsystem — plan

*Status: in progress (2026-09; A1–A5, U1, U2, B1, B2, B3 and B4a-1 shipped). Sequenced as 13 merges; each lands on its own and updates this document's merge list as it ships.*

## Context
Womblex is to serve other software in two modes:
- **(A)** a producer of a versioned on-disk data contract that consumers read directly;
- **(B)** a callable service that consumers submit work to and read results from.

The review confirmed these gaps:
- No contract version on any pipeline Parquet or on `egress_manifest.json`. Only the register ingests carry `schema_version`.
- Nothing marks raw-PII files apart from masked ones.
- No public Python API: `womblex/__init__` exports only `__version__`.
- The FR section 8 determinism TO-DO was open (closed by A2).
- The only HTTP surface is an unauthenticated operator console. It has no upload path, no caller identity, and no owner on `womblex_jobs`.
- Logging is plain stdlib text, with no metrics.

**Decisions taken:**
- The API is a separate `womblex serve` app. It reuses the console's scaffolding.
- The console is reframed as an admin/debug developer utility. It retires the location-override `PUT` and the report/feedback action.
- Callers authenticate with static service tokens. This is a trusted-subsystem deployment, not a public cloud service.
- Runs are owner-scoped.
- Determinism is an accuracy target, not a blocker. The contract guarantees *content* stability given the same version, config and models. File bytes are not guaranteed. Drift is detectable, not prevented.

Each merge stays under the 500-line cap and passes ruff, mypy and pytest on its own. `ui/readers.py` is already at the 750-line cap, so any new reader logic goes in a new module.

---

## Track A — data contract (serves A and B)

### A1. Contract version + sensitivity footer keys
- Add `CONTRACT_VERSION = "1.0"` and `SENSITIVITY_KEY` to `store/run_stamp.py`, and emit both from `RunStamp.footer_metadata()`. `test_the_package_version_is_not_the_schema_version` already anticipates keeping the contract version separate from the package version.
- Give `footer_metadata` and `sidecar_footer` a `sensitivity: Literal["raw","masked","none"]` argument:
  - `raw`: elements, table_cells, form_fields, chunks, pii_spans, enrichment*, normalised, spellfix.
  - `masked`: clean_text.
  - `none`: embeddings, manifest, money, quality.
- Route the footer-bypass writers through `_write_rows` + `sidecar_footer`:
  - the three bare `pq.write_table` calls in `store/enrichment_output.py`, and its duplicate `_write_enrichment_rows`;
  - `store/provenance_output.py`;
  - `redact/batch.py`.
- Add `contract_version` to `egress_manifest.json` (`store/egress.py` `_write_egress_manifest`).
- Test: a parametrised test over every sidecar writer asserts both keys are present.

### A2. Determinism contract (closes FR section 8 TO-DO)
- Add a `content_digest` column to `MANIFEST_SCHEMA`. It is the sha256 over each document's ordered elements (kind, order, text, plus cells and fields).
  - Compute it in `write_results` (`store/output.py`).
  - Backfill it as null through `_MANIFEST_BACKFILL`.
  - Carry it into `store/run_manifest.py`.
- Contract text: for a given `source_hash` + `womblex.version` + `config_digest` + `womblex.models`, extraction content and row order are stable, so `content_digest` matches. File bytes and `extracted_at_iso` are not stable.
  - A consumer re-running a year later compares `content_digest`. A mismatch is explained by the stamped version, config and model digests.
  - A mismatch never blocks output.
- Test: extract one vendored fixture twice and assert equal `content_digest` and equal rows. Byte differences are allowed.
- Update FR section 8: replace the TO-DO with this contract.

### A3. Consumer contract doc
New `docs/contract.md`:
- Lists every shard and sidecar, its join keys and its sensitivity.
- States the contract-version bump rules: additive column → minor; rename or removal → major, which needs a reader shim like the existing `_CHUNKS_BACKFILL`.
- States the determinism guarantee from A2.
- States what "safe to hand onward" means: `sensitivity=masked|none` only.

Cross-link it from `docs/egress.md` and `docs/extraction.md`.

### A4. Stable Python API + deprecation policy
- `womblex/__init__.py` re-exports a declared `__all__`, drawn from what exists today:
  - `extract_text`;
  - the `operations.run_*` functions;
  - the `*_shards` stage functions;
  - `build_bundle`, `write_run_manifest`, `read_results` (an existing alias of `read_elements`);
  - `CONTRACT_VERSION`.
  Use lazy imports so `import womblex` stays cheap.
- Add a test pinning `__all__`.
- Add a deprecation policy section to `docs/contract.md`: one minor release with a warning before removal. Add a CHANGELOG note.

### A5. Structured logging
`cli/_shared.setup_logging` gains `--log-format json` (and env `WOMBLEX_LOG_FORMAT`). Shipped as a top-level `womblex --log-format` flag; the worker and `womblex run` attach context via `utils/log_format.log_context`:
- It uses a stdlib-only JSON formatter, with no new dependency.
- It carries `run_id`, `job_id`, `stage` and `source_hash` through `extra=`.
- The worker (`cloud/worker.py`) and `capture_batch_log` (`utils/run_log.py`) pass that context.

---

## Track B — service API

### B1. Extract shared dispatch (mechanical)
Move `enqueue_extraction` and `enqueue_downstream_stages`, with their guard and result dataclasses, from `ui/execute.py` to a new `cloud/dispatch.py`. `ui/execute.py` keeps thin re-imports, so there is no behaviour change and existing UI tests pass unchanged. It also adds `connect_timeout`, matching the dashboard's `QUEUE_CONNECT_TIMEOUT`, which now lives in `cloud/dispatch.py` and is imported by the dashboard.

### B2. Run ownership in the queue
*Shipped.* Refusal is raised as `RunOwnedError`. An unscoped enqueue (`owner=None`: CLI, console) is not checked and its new rows inherit the run's existing owner, so a run never has mixed ownership. A named owner is also refused a run that has no owner. The CLI and console dispatch do not name an owner yet; the service API passes one from B4.

- `sql/womblex_jobs.sql` and `_SCHEMA` in `cloud/queue.py` gain a nullable `owner text` column and an index on `(owner, run_id)`. `ensure_schema` applies `ALTER TABLE … ADD COLUMN IF NOT EXISTS`, so existing queues migrate in place.
- `enqueue` and `enqueue_stages` take an `owner=` argument.
  - `enqueue` refuses a `run_id` already owned by a different owner.
  - `stats`, `list_jobs`, `workers` and `throughput` take an `owner=` filter.
  - A null owner means CLI or console-submitted, visible to admin only.
- Add a `JobQueue.runs(owner=)` listing: distinct `run_id` with status rollup.
- Update the byte-for-byte schema test.

### B3. Service-token auth
*Shipped.* The startup refusal on an empty registry (`ClientRegistry.empty`) and `--insecure-no-auth` (`caller_dependency(None)`) are provided here but enforced by `womblex serve` in B4. `admin` implies every scope; `Caller.owner` is `None` for admin and the client id otherwise, the value B4 passes to the queue's `owner=` filters.

New `api/auth.py`:
- It loads a client registry from `WOMBLEX_API_CLIENTS`, a YAML path. Each entry holds `client_id`, `token_sha256` and `scopes`. Scopes are `submit`, `read`, `read_raw` and `admin`.
- It compares tokens in constant time with `hmac.compare_digest`.
- It provides a FastAPI dependency returning a `Caller(client_id, scopes)`.
- The API refuses to start with an empty registry unless given `--insecure-no-auth`, which is for local development only.

Add `womblex api-token --client X` to print a new token and its hash for the registry. Document the registry in `docs/deployment-images.md` as private-network only.

### B4. `womblex serve` — `/v1` API
*B4a split in two to stay under the cap. B4a-1 shipped: `womblex serve`, health/ready, and the owner-scoped reads (run list, detail, manifest, metrics). Runs are known to the queue, so CLI- and console-submitted runs (no owner) are visible to `admin` only. B4a-2 remains: `POST /v1/runs` and `/files`. For B4a-2: `enqueue_extraction` and `enqueue_downstream_stages` need `owner=`; the run id is minted server-side; a non-admin caller's `input_prefix` is confined under its own client id; and a shared stage-list gate in `cloud/dispatch.py` should reject a bad config before any row is written. The OpenAPI pin is a test over the operation and model surface, not a snapshot file.*

- New `api/app.py` `create_api_app(...)`. It reuses `UISettings` binding from `ui/deps.py`, `RemoteStore` and `cloud/dispatch.py`.
- New `cli/serve.py`. It binds to loopback by default.
- Pydantic response models give a generated OpenAPI spec. A snapshot test pins `openapi.json`, so breaking API changes are visible in review.

| Method | Path | Scope | Reuses |
|---|---|---|---|
| GET | `/v1/health`, `/v1/ready` | none | store/queue probes in `ui/resources.py` |
| POST | `/v1/uploads` (multipart) | submit | `RemoteStore` → `<ingest>/<client_id>/<upload_id>/`, size-capped, supported-type check via `select_supported` |
| POST | `/v1/runs` | submit | `cloud/dispatch.enqueue_extraction` + `enqueue_downstream_stages`; body takes `input_prefix` (from upload), `preset` or `config`, `batch_size` |
| GET | `/v1/runs`, `/v1/runs/{id}` | read | `JobQueue.runs/stats/list_jobs(owner=)` |
| GET | `/v1/runs/{id}/manifest` | read | run manifest reader in `ui/readers.py` |
| GET | `/v1/runs/{id}/files` | read | object keys + footer (contract_version, sensitivity) via `read_parquet_footer`; lets mode-A consumers fetch Parquet directly |
| GET | `/v1/runs/{id}/documents/{hash}/text` | read (masked) / read_raw (raw) | new `api/readers.py`; serves `clean_text` by default, and `elements`/`chunks` only with `read_raw`, gated by the A1 sensitivity key |
| GET | `/v1/runs/{id}/metrics` | read | `JobQueue.stats/throughput/workers` as JSON (no Prometheus dependency) |

- A run owned by another client returns 404, and `admin` sees all.
- Out of scope for v1: callbacks and webhooks (consumers poll), cancellation, per-client quotas.

### B5. Packaging + deploy
- The API needs fastapi, uvicorn and python-multipart. Either reuse the `[ui]` extra or add an `[api]` extra. **This is a `pyproject.toml` change and needs your approval.** python-multipart is a new dependency.
- Add an `api` service to `docker-compose.yml`, on the same image as the workers, with the registry mounted read-only.

---

## Track U — console reframe (admin/debug utility)
- **U1. Retire the location override:**
  - Remove `PUT /api/resources/locations`.
  - Remove the `settings_dir` overlay in `ui/deps.py` (`get_settings` override) and `settings_store.py`.
  - Remove the `--settings-dir` flag and the SPA location cards.
  - Locations become deploy-time config only.
- **U2. Retire the report/feedback action:**
  - Remove `routes/feedback.py` and `store/feedback_output.py`.
  - Remove the `X-Womblex-Reported-By` handling. (The SPA never gained report buttons, so there was nothing to remove there.) `is_safe_run_id` moved to `store/retention.py`, as other modules depend on it.
  - Update the CLAUDE.md module table and the CHANGELOG.
- **U3. Reframe the docs:**
  - Rewrite FR sections 15–21 as developer/admin stories.
  - Add to the README and `cli/ui.py` docstring: "admin and debugging utility; integrations use `womblex serve`".
  - Update `docs/architecture.md` and `docs/project-structure.md`.
- Kept as-is: run inspection, audit, chunk inspector, logs, dashboard, resource probes, composer (schema, validate, YAML, presets), and admin dispatch.

---

## Merge order
Each merge must be under 500 lines. Split further if needed.

1. A1 contract version + sensitivity footers — **shipped (#114)**
2. A2 `content_digest` + FR section 8 contract — **shipped**
3. A3 + A4 contract doc + public `__all__` — **shipped**
4. A5 JSON logging — **shipped**
5. U1 retire the location override — **shipped**
6. U2 retire the feedback action — **shipped**
7. B1 extract `cloud/dispatch.py` — **shipped**
8. B2 queue `owner` column + migration — **shipped**
9. B3 service-token auth + `api-token` verb — **shipped**
10. B5 dependency extra — waits on your approval
11. B4a-1 `serve` app: health/ready, run list/detail, manifest, metrics — **shipped**
12. B4a-2 `POST /v1/runs` (owner threaded through dispatch) and `/files`
13. B4b uploads + document text endpoints with sensitivity gating
14. U3 docs reframe, plus architecture and project-structure updates

## Verification
- Each merge: `uv run ruff check src/ tests/`, `uv run mypy src/`, `uv run python -m pytest tests/ -v -m "not slow and not benchmark"`, plus `git diff --stat` against the merge base, which must be under 500 lines.
- A1/A2: a parametrised footer-key test over every writer, and the double-extraction `content_digest` equality test on a vendored fixture.
- B2: queue tests against the compose Postgres (`docker compose --profile local up postgres`); migrate an existing table in place.
- B3/B4: FastAPI `TestClient` tests covering:
  - missing or wrong token → 401;
  - wrong owner → 404;
  - raw text without `read_raw` → 403;
  - an OpenAPI snapshot.
- End to end on the local compose stack:
  1. `womblex serve`;
  2. `curl -H "Authorization: Bearer …" -F file=@fixture.pdf /v1/uploads`;
  3. `POST /v1/runs`;
  4. a worker drains the queue;
  5. `GET /v1/runs/{id}` reaches done;
  6. `GET …/documents/{hash}/text` returns masked text;
  7. `/files` shows `contract_version` and sensitivity.
