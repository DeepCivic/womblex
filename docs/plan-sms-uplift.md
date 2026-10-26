# SMS uplift — plan

*Status: proposed (2026-10); nothing built. Phases A (collections) and B (inbound webhooks) are the priority and stand alone; C to G may follow later. This document is deleted when its last phase lands.*

## Context

SMS-validate reads Womblex output and assigns kinds itself. Womblex's job is to emit
general-purpose atoms, and to give other systems enough capability to manage data
around it. Womblex does not become a warehouse. No SMS-named stage goes in Womblex.

**Priority:** collections (Requirement 34) and inbound webhooks (Requirement 35) come first.
Everything after them can be delayed without leaving the tree half-built.

**In scope:**
- collections
- inbound webhooks
- progress webhooks
- write-time timestamps
- withheld and omitted documents recorded as atoms
- failed-document manifest rows
- a spike on element evidence and fingerprints, with clean-up
- a spike on statement identity across versions, with clean-up
- the rest of the draft: location fingerprints, evidence fallback, Requirement 32 payload, records hash

**Out of scope:**
- kind assignment
- SMS assembly and validation (Requirement 33)
- derivation records
- classification scheme and version
- dropping confidence scores

**Merge rules (CLAUDE.md):**
- Each merge is ≤500 changed lines and passes ruff, mypy and pytest on its own.
- A file near the 750-line cap gets no new responsibility; new code goes in a new module beside it:
  - `store/output.py`: 707 lines
  - `cloud/jobs.py`: 673
  - `store/run_manifest.py`: 635
  - `store/shard_audit.py`: 709
- A new module lands in the CLAUDE.md table, `docs/architecture.md` and `docs/project-structure.md`
  in the same merge (`tests/test_doc_currency.py`).
- A merge that satisfies a requirement updates `docs/functional_requirements.md`.

## Phase A — Collections (Requirement 34)

Design: **a collection is a long-lived run whose id is the collection name.** This reuses
existing machinery as-is:
- `fire_workflow` adds new and changed documents, deduplicated by fingerprint and content hash, and keeps earlier versions
- `next_batch_num` numbers new batches after the run's last
- finalize consolidates the run manifest

The only identity change is that the run id stops being the workflow name.

**A1. `collection` field on the workflow spec.**
- Add an optional safe-name field to `StoredWorkflowSpec` (`cloud/workflow_store.py`).
- Drop it from `workflow_digest` and from `config_digest`, the same way an empty `destinations` list is dropped today.
- `run_scheduled` / `_fire` (`cloud/workflows.py`) use `spec.collection or stored.id` as the run id.
- `_extract` declares `IngestProvenance` with the collection, not `run_config.dataset.name`. `dataset.name` stays the preset name and keeps feeding `womblex.preset`.
- `workflow_runner.config_from_overlay` is unchanged.
- Record the digest exception in `docs/decisions-operations.md` beside "A workflow is a file".

**A2. Add to a collection on demand.**
- New CLI `womblex add-to-collection FILE --store --dsn`. It enqueues `fire_trigger` on `dbos_app.firing_queue(store)` with firing label `manual-<UTC>`, the same path a schedule takes. That gives it the same checks (`check_firing`), deduplication and replay safety.
- `WorkflowStore` currently keeps only workflows that have triggers (`workflow_schedules.schedule_workflow` deletes the rest). Widen this: store the copy when the workflow has triggers, a collection or inbound hooks. That way the API and inbound events can find it by name.

**A3. Workflow changes and rerunning a collection.**
- A dispatch outside a firing (`dispatch_workflow`, where `bases=None` means every batch) applies the current steps to the whole collection. Unit receipts (`cloud/unit_receipt.py`) skip units whose version, stage config and inputs are unchanged. So adding `pii` reruns only `pii`, and a changed setting reruns only the stages it touches.
- Expose this as `womblex rerun-collection FILE`, a thin wrapper over `dispatch-workflow` without `--firing`.
- Re-extraction is out: batch ids are deterministic and attach rather than rerun. Record this as a limit in the requirement.

**A4. API.**
- New `api/routes_collections.py`:
  - `GET /v1/collections`: from `WorkflowStore` plus the store's runs.
  - `GET /v1/collections/{name}`: documents, current-version counts, workflows that wrote to it (read from the run record's `workflows`).
  - `POST /v1/collections/{name}/add` and `POST /v1/collections/{name}/rerun` (submit scope).
- Add a `collection` field to every outbound webhook `event_body` (`cloud/webhook.py`).
- Update `test_openapi_surface_is_pinned`.

**A5. Document versions (the Requirement 31 need, document level).**
- New `store/document_versions.py`. At consolidation (`cloud/finalize.py`, `store/run_manifest.write_run_manifest`) it adds three columns to the run-root `manifest.parquet`:
  - `replaces`: the earlier version's `source_hash`
  - `replaced_by`
  - `is_current`

  Versions are grouped by `source_relpath` and ordered by batch number, never by time.
- Per-batch manifests are never rewritten.
- Additive columns, so contract 3.0 → 3.1, with a backfill entry in `store/output.py` `_BACKFILL`.

**A6. Redelivery trace.**
- `store/delivery_record.py` keeps the earlier records as `<destination>.<seq>.json`.
- Each record carries `seq`, `replaces` and `replaced_by`, plus whether `content_digest` changed per document.
- Run record version 8 → 9.

## Phase B — Inbound webhooks (Requirement 35)

**B1. Spec and storage.**
- Add `inbound: [{name, secret_env, events}]` to the workflow spec, validated like `WebhookDestination`.
- New `cloud/inbound_store.py` creates the `womblex_inbound_events` table (`event_id` PK, `workflow_id`, `hook`, `event`, `received_at`, `outcome_json`). It follows the `womblex_workflows` / `worker_registry` DDL pattern: insert first, and an `IntegrityError` means the event is a duplicate.

**B2. Route.** New `api/routes_inbound.py` serves `POST /v1/hooks/{workflow}/{hook}`.
- Authentication is HMAC only, not the bearer token:
  - read the raw body with `await request.body()`
  - verify with `webhook.sign` + `hmac.compare_digest` against `X-Womblex-Signature`
  - `--insecure-no-auth` still checks the signature
- Every check runs before any row is written:
  1. signature
  2. workflow is stored
  3. hook declares the event
  4. `event_id` not seen before
  5. `check_firing` / `require_served`
- Actions:
  - `documents_added` and `run_requested`: enqueue `fire_trigger` with firing `event-<sha8(event_id)>`, so the DBOS id is idempotent too.
  - `rerun_requested`: the A3 dispatch.
  - `delivery_received`: append to that destination's delivery record. The run's status is never touched.
- Refusals return `{code, message}` using the `RefusalCode` values plus `bad_signature`, `unknown_hook`, `event_not_declared` and `duplicate`.

**B3. Listing and docs.**
- `GET /v1/workflows/{id}/inbound-events`.
- `delivery_received` reopens Requirement 27's "write-back from consumers" exclusion. Record that in `docs/decisions-operations.md`.

## Phase C — Progress events and timestamps

**C1. Progress webhooks.**
- Add `stage_started` and `batch_progress` events. `batch_progress` fires once per extraction batch settling and once per stage unit, carrying `stage`, `done`, `failed`, `total`.
- `WebhookDestination` gets an optional `events:` filter, so a consumer can subscribe to coarse events only.
- Reopen the "no per-batch events" rule in the `cloud/webhook.py` docstring and the decisions doc.
- Emitters go in a new `cloud/run_events.py`, because `cloud/workflows.py` is at 540 lines. Each emit is a DBOS step with an idempotent id. `run_stage` emits per unit, and the coordinator emits `stage_started`.
- Bug fix: `run_finished` is currently sent only when the workflow has destinations.

**C2. Write time.**
- `sidecar_footer` / `contract_footer` add a `womblex.written_at` footer key. It is never a column, so `content_digest` and the rows don't change.
- Reword the byte-identical claims ("identical aside from the write time"):
  - `docs/decisions-operations.md` (two places)
  - `cloud/unit_receipt.py`
  - the `batch.process_batch` docstring
- None of the existing byte-identity tests compare Parquet files that both get rewritten. Confirm with a full run.

## Phase D — Withheld, omitted and failed documents

**D0. Make room.** Move manifest row building out of `store/output.py` (707 lines) into a new `store/manifest_rows.py`. This merge is mechanical only.

**D1. Failed documents get manifest rows.**
- `operations/persist.py` and `cli/pipeline.py` pass errored results through, so the existing `status: error` branch is finally reached.
- A source that can't be read is recorded as failed. Its `source_hash` is SHA-256 over `unreadable:` + `source_relpath`, with hash basis `unreadable`, so join keys never collide and are never empty. This replaces the empty string `_source_hash` returns today.

**D2. Omitted files.**
- `select_supported` and `dispatch.supported_keys` return the files they dropped.
- New `store/omitted_output.py` writes a run-root `omitted.parquet` (`source_relpath`, `reason`: `unsupported_type`, `nested`) at enqueue or at the start of a local run.
- Enrichment's `skip_short_documents` writes one `enrichment_meta` row per skipped document with `status = skipped_short`, so the document is recorded as "not looked for". This is an additive column.

**D3. Withheld regions.**
- New sidecar `*.redaction_regions.parquet` (`source_hash`, `page`, `bbox` in PDF points converted from detection DPI, `area`), written by the in-batch path (`operations/redact.py`) and by `redact/batch.py`.
- Today region boxes are never persisted; only page-level `has_redaction` is.

## Phase E — Spikes, then clean-up

**E1. Element evidence and fingerprint spike.**
- Prototype, on a throwaway branch, two new columns on `ELEMENT_SCHEMA`, `TABLE_CELLS` and `FORM_FIELDS`: SHA-256 over `content_digest._element_payload`, and the element's own evidence reference.
- Measure over the synthetic fixtures:
  - shard size growth
  - write time
  - every reader that breaks: `conform_shard`, which has no `elements` backfill entry today, and the second writer `ingest/records.py`
  - how the fingerprint interacts with the normalise and spellfix overlays (which layer it is taken over)
  - cell-level and form-field coverage
  - contract bump size
- Write the findings up as a decision.

**E2. Statement identity spike.**
- Prototype a deterministic key (stage + collection + `source_relpath` + locus + verbatim text) over money, entity-link and PII sidecars for two versions of a changed synthetic document.
- Measure match, false-match and miss rates.

**Clean-up after each spike:**
- delete the spike branch and any prototype code
- record the outcome in `docs/decisions.md`, or in `docs/decisions-dead-ends.md` if it is rejected
- open the build requirement text if it goes ahead
- confirm no spike columns, fixtures or tests reached `main`
- check `git log origin/main` for the spike commits

## Phase F — The rest of the draft

Phase F waits on E1, which settles the fingerprint basis.
- Location fingerprints on the evidence reference (Requirement 30), in `store/evidence.py` `EVIDENCE_FIELDS` with `backfill_evidence` null.
- Evidence fallback order across `link/stage.py` and `process/money_stage.py`. PII's deliberate document-before-chunk order is kept and documented.
- Requirement 32: quotes written as located rows, and pinpoints plus jurisdiction carried on external-document rows.
- Records-ingest hash with length-prefixing and a new hash basis. This needs a contract major bump and a reopening entry for "Pre-extracted corpora".

## Phase G — Stocktake

A docs-only merge per CLAUDE.md "Stocktake":
- module tables
- README command table (new CLI commands)
- Outstanding index
- CHANGELOG
- `functional_requirements.md` Requirements 30, 31, 34 and 35 written in that file's conventions

## Verification (each merge)

- `uv run ruff check src/ tests/` && `uv run mypy src/` && `uv run python -m pytest tests/ -v -m "not benchmark"`
- `git diff --stat $(git merge-base HEAD origin/main)..HEAD` ≤500 lines; `wc -l` on every touched file.
- **A:** a `fleet`-fixture test (SQLite + Postgres):
  - two firings of one workflow into collection `c1` give one run `c1`
  - changing a synthetic file gives two linked versions, the later one `is_current`
  - renaming the workflow file keeps the collection
  - `rerun-collection` after adding `pii` runs only `pii`
  - the API lists `c1`
- **B:** `TestClient` tests:
  - a signed `documents_added` adds one file
  - a resend adds nothing
  - a bad signature, a wrong hook and no live worker are each refused with their code and no queue row
  - `delivery_received` appears on the delivery record with the run status unchanged
- **C:** the `test_webhook.py` receiver sees `stage_started` and `batch_progress` with counts summing to the total. The `events` filter drops them.
- **D:** a fixture with an unreadable file, an unsupported file and a redacted PDF gives an error manifest row, an `omitted.parquet` row and region rows with page and bbox.
