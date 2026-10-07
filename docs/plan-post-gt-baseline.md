# Post ground-truth revision baseline — plan

*Status: waiting on the ground-truth revision in womblex-benchmark (2026-10). One pass, run by a maintainer once the revision lands; this document is deleted when the baseline reports are merged.*

## Context
Womblex, womblex-benchmark and the ground truth have all changed substantially: layout became its own stage, extraction moved behind the PDF seam, the test suite moved to synthetic fixtures, and the ground truth is being revised. Comparing the next benchmark run with earlier reports would mostly measure those changes, not quality. The next run is therefore a **fresh baseline**: everything runs, with every dependency present, and the reports it produces are the reference from then on.

## Preconditions
- The ground-truth revision is merged in womblex-benchmark.
- Womblex `main` and womblex-benchmark `main`, checked out side by side.
- The full benchmark fixture set, also copied to `fixtures/fixtures/` in the Womblex checkout for `test_bench_ocr_accuracy.py` (see [THIRD_PARTY_DATA.md](../THIRD_PARTY_DATA.md)).
- An `ISAACUS_API_KEY`, AWS credentials with Bedrock Pixtral Large access, and a Postgres DSN (`WOMBLEX_DB_DSN`), so nothing skips for want of a service.

## Steps
1. **Womblex suite, nothing deselected.** `uv run ruff check src/ tests/`, `uv run mypy src/`, then `uv run python -m pytest tests/ -v -rs`, which includes the `benchmark` module. Every skip left in the `-rs` summary is either explained (the geospatial extra) or fixed before going on.
2. **Every benchmark suite.** From the womblex-benchmark checkout, with Womblex's venv: `uv run python -m pytest accuracy/ -v`, regenerating every `docs/accuracy/*.md` report.
3. **Read the reports, don't diff them.** Check each report by hand for failures of the harness itself (a suite that scored nothing, a fixture that did not load, a metric outside its range) and fix those before taking the baseline. Prior numbers are not a gate.
4. **Land the baseline.** One Womblex merge with the regenerated reports (they do not count towards the merge cap), stating in the PR body that they are a new baseline, not comparable with earlier reports.

## What this run closes
- The benchmark items in [plan-permissive-deps.md](plan-permissive-deps.md) that only regenerate reports on the revised ground truth: L3c-B (`REDACTION_HANDLING`), the regeneration half of L1-B, and the transcript-CER half of H-B. L1-B's code changes in womblex-benchmark (DocLayNet `--model`, the label-map wording) land before step 2.
- Absolute thresholds in that plan's gates that were set from earlier reports (the DocLayNet layout F1, the Auditor-General transcript CER) are reset from this baseline in the same merge. Between-backend gates (fitz against pdfium) are unaffected.
