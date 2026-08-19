# Fix: strict stage separation — extraction does only extraction

A suite PR covering a run of related defects found in a live SageMaker/MinIO
deployment. Each issue is resolved in its own commit; the next developer picks
up the next open issue and updates this comment.

## Status

| # | Issue | Owner | Status |
|---|-------|-------|--------|
| 1 | Failed runs — `ValueError: No AWS region for SageMaker endpoint(s)` | this thread | **✅ Fixed (software half)** — deployment must still set `ISAACUS_SAGEMAKER_REGION` |
| 6 | `process_batch` runs chunk + PII inside extraction | this thread | **✅ Fixed** (same commit — shared root cause) |
| 5 | Console composes a pipeline it cannot execute (downstream stages never run) | next | ⏳ Open — own merge; PM wants ordering business-logic vetted |
| 2 | MinIO shows only extraction parquets; no chunks/embeddings | next | ⏳ Open — resolves once #5 runs downstream stages + region set |
| 3 | Chunk Inspector "0 chunks / chunk stage may not have run yet" | next | ⏳ Open — correct reporting; resolves with #2 |
| 4 | Chunk Inspector "This run has no documents" for a 140-doc run | next | ⏳ Open — needs frontend trace of `ui/src/routes/chunks/+page.svelte`; may self-resolve with #2 |
| 7 | Light-mode wrong colour scheme | next | ⏳ Open — PM confirmed accurate background is `#C0FF00`; correct `--surface-sunken` (and DESIGN.md) from the lime `#b0d820` |

## ✅ Fixed in this thread — Issues 1 & 6 (commit "Extraction does only extraction")

**Root cause (shared).** `womblex.batch.process_batch` sequenced
`extract → redact → chunk → pii`, gating chunk on `chunking.enabled` and PII on
`pii.enabled`. But those flags now mean *"this stage is in the pipeline"*, not
*"run it inside extraction"*. Consequences:

- The in-batch chunk/PII output was **discarded** — `write_results` has no
  chunk/`clean_text` path, so nothing was persisted anyway (this is why MinIO
  only ever showed extraction parquets — see #2).
- Chunk ran **before enrich** (wrong order for AI-chunking reuse).
- With `chunking_model: kanon-2-enricher`, the in-batch chunk built a SageMaker
  client via `make_ai_chunking_client → _require_region`, which raised
  `ValueError: No AWS region for SageMaker endpoint(s) kanon-2-bundle-001`
  **inside the batch, killing all 3 attempts** (#1).

**Fix.** `process_batch` is now **extract → optional redaction detection** only.
Chunk, PII, enrich, embed, money are downstream `run-stage` contracts,
dispatched separately. Redaction *detection* stays — flagging where the source
itself redacted content is representing the document true-to-form, not a
transform on top of it (per CLAUDE.md "Redaction is a post-extraction concern"
and the PM's clarification).

- `batch.py` — dropped `run_chunking` / `run_pii_cleaning`; extraction-only
  docstring.
- `cli/pipeline.py` — `cmd_run` reporting reconciled; removed the now-moot
  `post_enrichment`-PII pre-flight guard and the `chunk_will_skip` logic; logs a
  one-line "dispatch these downstream stages per-stage after extraction" note
  when the config enables them.
- `operations/redact.py` — log now reads `RedactionDetected: …` (flag mode, no
  text change) and `RedactionApplied [blackout|delete]: …` (mutating modes), so
  a batch log no longer reads as though PII cleaning ran inside extraction.
- Tests + `CLAUDE.md` / `README.md` / `configs/default-isaacus.yaml` /
  `docs/*` reconciled to extraction-only.

**PII paper trail.** No PII ever ran in-batch — both live runs had
`pii.enabled: false`, the in-batch call was gated off, and its output would have
been discarded regardless. No masking occurred, no data was lost. The `pii`
stage (`pii/pii_stage.py`, `run-stage --stage pii`) is untouched.

**Deployment action still required (cloud engineer, not code):** set
`ISAACUS_SAGEMAKER_REGION=ap-southeast-2` (or `AWS_REGION`, or
`ISAACUS_SAGEMAKER_ENDPOINTS=kanon-2-bundle-001@ap-southeast-2`). The real
enrich/chunk/embed stages call SageMaker; the software surfaces the missing
region correctly.

## Handoff note for the next developer

Pick up **Issue 5** next (the natural next step — it makes #2/#3 resolvable).
PM guidance: *"Auto-order is fine, but vet the business logic with me."* Correct
order is enrich → chunk → graph-refresh → embed → money → pii (enrich before
chunk for AI-chunking reuse; money/pii last). Scope it as its own merge under
the 500-line cap.
