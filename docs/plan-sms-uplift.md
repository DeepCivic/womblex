# SMS uplift — plan

*Status: proposed (2026-10); a piece is removed from this plan when it is built. Phases A (collections) and B (inbound webhooks) are the priority and stand alone; C to G may follow later. This document is deleted when its last phase lands.*

## Context

SMS-validate reads Womblex output and assigns kinds itself. Womblex's job is to emit
general-purpose atoms, and to give other systems enough capability to manage data
around it. Womblex does not become a warehouse. No SMS-named stage goes in Womblex.

**Priority:** collections (Requirement 34) and inbound webhooks (Requirement 35) come first.
Everything after them can be delayed without leaving the tree half-built.

**Requirement numbers:** Requirements 30 to 35 are working labels for this plan only.
`docs/functional_requirements.md` stops at Requirement 29. As each piece is delivered, its
requirement is written there as built, in that file's conventions and numbering. The piece is then
removed from this plan in the same merge.

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
- A merge that delivers a piece of this plan writes its requirement as built into `docs/functional_requirements.md` and removes that piece from this plan.

## Phase A — Collections (Requirement 34)

Design: **a collection is a long-lived run whose id is the collection name.** This reuses
existing machinery as-is:
- `fire_workflow` adds new and changed documents, deduplicated by fingerprint and content hash, and keeps earlier versions
- `next_batch_num` numbers new batches after the run's last
- finalize consolidates the run manifest

The only identity change is that the run id stops being the workflow name. That change is built (Requirement 34): a workflow's `collection` names its run, and the schedule context carries the run id.

**A2. Add to a collection on demand.** Built (Requirement 34), except that the stored copy is also kept for inbound hooks once B1 adds them.

**A2a. Workflow owner.** Built (Requirement 34): the store holds an owner and firings run under it. Nothing yet stores a workflow under a non-admin owner (the CLI stores none, and A4 does not create workflows); A4 must settle how an owned workflow comes to exist.

**A4. API.**
- New `api/routes_collections.py`:
  - Owner-scoped like `/v1/runs`: a caller sees and acts on its own collections (A2a), another owner's is 404, admin sees all.
  - `GET /v1/collections`: from `WorkflowStore` plus the store's runs.
  - `GET /v1/collections/{name}`: documents, current-version counts, workflows that wrote to it (read from the run record's `workflows`).
  - `POST /v1/collections/{name}/add` and `POST /v1/collections/{name}/rerun` (submit scope).
  - These two routes reopen the "managing workflows through the API" exclusion in Requirement 27's out-of-scope list. They act on a stored workflow but neither creates, edits nor removes one. Record the narrowed exclusion in `docs/decisions-operations.md`.
- Add a `collection` field to every outbound webhook `event_body` (`cloud/webhook.py`).
- Update `test_openapi_surface_is_pinned`.

**A5. Document versions (the Requirement 31 need, document level).**
- New `store/document_versions.py`. At consolidation (`cloud/finalize.py`, `store/run_manifest.write_run_manifest`) it adds three columns to the run-root `manifest.parquet`:
  - `replaces`: the earlier version's `source_hash`
  - `replaced_by`
  - `is_current`

  Versions are grouped by `source_relpath` and ordered by batch number, never by time.
- Limits, recorded in the requirement beside "re-extraction is out":
  - a moved or renamed file starts a new chain
  - identical bytes at two paths form two chains
- Unreadable rows (D1) chain like any other version; their batch-qualified hash keeps each occurrence distinct.
- Per-batch manifests are never rewritten.
- Additive columns, so contract 3.0 → 3.1, with a backfill entry in `store/output.py` `_BACKFILL`.

**A6. Redelivery trace.**
- `store/delivery_record.py` keeps the earlier records as `<destination>.<seq>.json`.
- Each record carries `seq`, `replaces` and `replaced_by`, plus whether `content_digest` changed per document.
- A redelivery writes a new `seq`; earlier records are never rewritten, except that `delivery_received` (B2) appends acknowledgements to the record it names.
- Run record version 8 → 9.

## Phase B — Inbound webhooks (Requirement 35)

**B1. Spec and storage.**
- Add `inbound: [{name, secret_env, events}]` to the workflow spec, validated like `WebhookDestination`.
- Each hook needs its own secret: validation refuses two hooks (in this workflow or another stored one) naming the same `secret_env`. The signature stays body-only, matching outbound `webhook.sign`, so a sender uses a standard HMAC.
- Development switch: `WOMBLEX_INSECURE_INBOUND=1` turns the inbound checks off for a proof of concept:
  - validation warns, rather than refuses, a shared or unset `secret_env`
  - the route skips signature verification and logs a warning on every event it accepts unsigned
  - it is read only from the environment, never from the spec, so a stored workflow cannot turn it on
- New `cloud/inbound_store.py` creates the `womblex_inbound_events` table (`event_id` PK, `workflow_id`, `hook`, `event`, `received_at`, `outcome_json`). It follows the `womblex_workflows` / `worker_registry` DDL pattern: insert first, and an `IntegrityError` means the event was seen before.

**B2. Route.** New `api/routes_inbound.py` serves `POST /v1/hooks/{workflow}/{hook}`.
- Authentication is HMAC only, not the bearer token:
  - read the raw body with `await request.body()`
  - verify with `webhook.sign` + `hmac.compare_digest` against `X-Womblex-Signature`
  - `--insecure-no-auth` still checks the signature; only `WOMBLEX_INSECURE_INBOUND` (B1) skips it
- Checks, in order:
  1. signature
  2. workflow is stored
  3. hook declares the event — a refusal at 1 to 3 writes nothing, so unauthenticated or malformed requests never fill the table
  4. insert the `event_id` row. A conflict means the event was seen before: the route returns the stored `outcome_json`, whether that was an acceptance or a refusal, and does nothing else.
  5. `check_firing` / `require_served`, before any queue row. The outcome, accepted or refused with its code, is written to `outcome_json`. A sender that wants to retry a refused event sends a new `event_id`.
- The firing runs under the workflow's owner (A2a).
- Actions:
  - `documents_added` and `run_requested`: enqueue `fire_trigger` with firing `event-<sha8(event_id)>`, so the DBOS id is idempotent too.
  - `rerun_requested`: the `rerun-collection` dispatch (Requirement 34).
  - `delivery_received`: the event names the destination and the `seq` (A6) it acknowledges, and is appended to that record. A `seq` with no record is refused (`unknown_delivery`). The run's status is never touched.
- Refusals return `{code, message}` using the `RefusalCode` values plus `bad_signature`, `unknown_hook`, `event_not_declared` and `unknown_delivery`. A resend returns the first outcome, not a separate `duplicate` code.

**B3. Listing and docs.**
- `GET /v1/workflows/{id}/inbound-events`, owner-scoped like A4.
- `delivery_received` reopens Requirement 27's "write-back from consumers" exclusion. Record that in `docs/decisions-operations.md`.

## Phase C — Progress events and timestamps

**C1. Progress webhooks.**
- Add `stage_started` and `batch_progress` events. `batch_progress` fires once per extraction batch settling and once per stage unit, carrying `stage`, `done`, `failed`, `total`.
- `WebhookDestination` gets an optional `events:` filter, so a consumer can subscribe to coarse events only.
- Reopen the "no per-batch events" rule in the `cloud/webhook.py` docstring and the decisions doc.
- Emitters go in a new `cloud/run_events.py`, because `cloud/workflows.py` is at 540 lines. Each emit is a DBOS step with an idempotent id. `run_stage` emits per unit, and the coordinator emits `stage_started`.

**C2. Write time.**
- `sidecar_footer` / `contract_footer` add a `womblex.written_at` footer key. It is never a column, so `content_digest` and the rows don't change.
- Reword the byte-identical claims ("identical aside from the write time"):
  - `docs/decisions-operations.md` (two places)
  - `docs/composable-design.md` (money output before or after `enrich`)
  - `cloud/unit_receipt.py`
  - the `batch.process_batch` docstring
- The footer is part of the file's bytes, so every rewrite changes a file's ETag / checksum even when its rows don't. Each check that compares file bytes, and how it copes:
  - `RemoteStore.fingerprints` and receipt output fingerprints (`cloud/unit_receipt.py`): a receipt compares the published output with the fingerprint it recorded when it wrote that output, so a file that was not rewritten still matches. A rerun writes a new receipt. No change.
  - `published_checksums` (`cloud/finalize.py`) and the run record's checksums: they record what was published, so they change on every rewrite, and that is correct. The docs must stop calling them stable across reruns.
  - Local-vs-fleet and rerun comparisons in the tests that assert byte identity (`tests/test_records.py`, `tests/test_stage_runner.py` (the local-vs-remote parity check), `tests/test_enrich_stage.py`, `tests/test_enrichment_output.py`, `tests/test_source_provenance.py`, `tests/test_ground_truth.py`, `tests/test_workflow_plan.py`): compare rows plus the footer minus `womblex.written_at`. Add a shared test helper for that comparison rather than editing each assertion.
  - Determinism stays `content_digest`, which never covers footers.

## Phase D — Withheld, omitted and failed documents

**D0. Make room.** Move manifest row building out of `store/output.py` (707 lines) into a new `store/manifest_rows.py`. This merge is mechanical only.

**D1. Failed documents get manifest rows.**
- `operations/persist.py` and `cli/pipeline.py` pass errored results through, so the existing `status: error` branch is finally reached.
- A source that can't be read is recorded as failed. Its `source_hash` is SHA-256 over `unreadable:` + `source_relpath` + `:` + the batch number, so join keys are never empty and an unreadable file in two firings of one collection gets two distinct hashes (A5). This replaces the empty string `_source_hash` returns today.
- The row's `extraction_method` is `unreadable`. A row that failed after its bytes were read keeps its file-bytes hash and its method, so `status: error` alone does not mark this case.
- Hash basis (`store/source_resolver.py`):
  - add `HASH_BASIS_UNREADABLE`. `hash_basis_for` maps `extraction_method == "unreadable"` to it, the way `_RECORD_METHODS` maps to `record_id+text`; today it would read as `file_bytes`
  - `SourceResolver` declines the row as `unsupported_basis`, with a reason naming its `source_relpath`. It never resolves one: `resolved` always means the bytes matched `source_hash` ("Path first, then hash", `docs/decisions-operations.md`), and this hash is not over bytes
  - egress already reports `unsupported_basis` with no entry under `sources/` (`docs/egress.md`). Add the unreadable case beside the records-ingest one there.

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
- if it goes ahead, add the build to Phase F
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
- `functional_requirements.md` holds every delivered piece's requirement, and this plan is deleted

## Verification (each merge)

- `uv run ruff check src/ tests/` && `uv run mypy src/` && `uv run python -m pytest tests/ -v -m "not benchmark"`
- `git diff --stat $(git merge-base HEAD origin/main)..HEAD` ≤500 lines; `wc -l` on every touched file.
- **A:** a `fleet`-fixture test (SQLite + Postgres):
  - two firings of one workflow into collection `c1` give one run `c1`
  - changing a synthetic file gives two linked versions, the later one `is_current`
  - an unreadable file in two firings gives two distinct `source_hash`es in one chain
  - a schedule synced before the `collection` field (no `run_id` in its context) still fires into the run named for the workflow
  - changing the workflow's `name` while keeping `collection: c1` fires into run `c1`. The old name's stored copy and schedules stay until `remove-workflow` removes them
  - `rerun-collection` after adding a step runs only that step among the sidecar stages (built, `tests/test_workflow_schedules.py`)
  - the API lists `c1` for its owner and admin, and 404s for another owner
  - `remove-workflow c1` removes the copy and schedules and leaves the run
- **B:** `TestClient` tests:
  - a signed `documents_added` adds one file
  - a resend adds nothing
  - a bad signature, a wrong hook and no live worker are each refused with their code and no queue row
  - a bad signature or wrong hook writes no event row; a no-live-worker refusal writes one, and its resend returns the same refusal
  - two hooks sharing a `secret_env` are refused at validation, and only warned with `WOMBLEX_INSECURE_INBOUND=1`
  - an unsigned event is accepted only with `WOMBLEX_INSECURE_INBOUND=1`
  - `delivery_received` appears on the named `seq` record with the run status unchanged; an unknown `seq` is refused
- **C:** the `test_webhook.py` receiver sees `stage_started` and `batch_progress` with counts summing to the total. The `events` filter drops them.
- **D:** a fixture with an unreadable file, an unsupported file and a redacted PDF gives an error manifest row (`extraction_method` `unreadable`, which `SourceResolver` declines as `unsupported_basis`), an `omitted.parquet` row and region rows with page and bbox.
