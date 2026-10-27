# SMS uplift — plan

*Status: Phases A (collections) and B (inbound webhooks) are built (Requirements 34 and 35); C to G are proposed and may follow later. A piece is removed from this plan when it is built, and this document is deleted when its last phase lands.*

## Context

SMS-validate reads Womblex output and assigns kinds itself. Womblex's job is to emit
general-purpose atoms, and to give other systems enough capability to manage data
around it. Womblex does not become a warehouse. No SMS-named stage goes in Womblex.

**Priority:** collections (Requirement 34) and inbound webhooks (Requirement 35) came first and are built.
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

Items A1 to A6, including A2a, are built (Requirement 34). One question is open: the plan
says a workflow's owner is the API caller that stored it, but the CLI stores none and the API
routes do not create workflows, so nothing yet stores a workflow under a non-admin owner and
only an admin sees collections until that is settled (`docs/decisions-operations.md`, "Acting on
a collection through the API").

## Phase B — Inbound webhooks (Requirement 35)

Items B1 to B3 are built (Requirement 35).

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
- **A and B:** built; each requirement's **Verification** line (34 and 35 in `functional_requirements.md`) names its tests. One check waits on D1: an unreadable file in two firings gives two distinct `source_hash`es in one chain.
- **C:** the `test_webhook.py` receiver sees `stage_started` and `batch_progress` with counts summing to the total. The `events` filter drops them.
- **D:** a fixture with an unreadable file, an unsupported file and a redacted PDF gives an error manifest row (`extraction_method` `unreadable`, which `SourceResolver` declines as `unsupported_basis`), an `omitted.parquet` row and region rows with page and bbox.
