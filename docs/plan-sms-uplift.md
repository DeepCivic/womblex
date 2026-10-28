# SMS uplift — plan

*Status: Phases D to G are proposed and may follow later. A piece is removed from this plan when it is built, and this document is deleted when its last phase lands.*

## Context

SMS-validate reads Womblex output and assigns kinds itself. Womblex's job is to emit
general-purpose atoms, and to give other systems enough capability to manage data
around it. Womblex does not become a warehouse. No SMS-named stage goes in Womblex.

**Priority:** each phase can be delayed without leaving the tree half-built.

**Requirement numbers:** Requirements 30 to 33 are working labels for this plan only.
`docs/functional_requirements.md` holds none of them yet. As each piece is delivered, its
requirement is written there as built, in that file's conventions and numbering. The piece is then
removed from this plan in the same merge.

**In scope:**
- withheld and omitted documents recorded as atoms
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
  - `store/output.py`: 684 lines
  - `cloud/jobs.py`: 673
  - `store/run_manifest.py`: 635
  - `store/shard_audit.py`: 709
- A new module lands in the CLAUDE.md table, `docs/architecture.md` and `docs/project-structure.md`
  in the same merge (`tests/test_doc_currency.py`).
- A merge that delivers a piece of this plan writes its requirement as built into `docs/functional_requirements.md` and removes that piece from this plan.

## Phase D — Withheld, omitted and failed documents

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
- **D:** a fixture with an unreadable file, an unsupported file and a redacted PDF gives an error manifest row (`extraction_method` `unreadable`, which `SourceResolver` declines as `unsupported_basis`), an `omitted.parquet` row and region rows with page and bbox. An unreadable file in two firings of one collection gives two distinct `source_hash`es in one chain.
