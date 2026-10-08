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

## Benchmark-side actions
Moved here from [plan-permissive-deps.md](plan-permissive-deps.md). All are womblex-benchmark merges (B), paired with the Womblex merge that lands the baseline reports.

- **L1-B code changes (before step 2).**
  - Make the DocLayNet layout-F1 test honour `--model` (it calls `get_layout_analyzer()` with no arguments today).
  - Bring the `DOCLAYNET_TO_WOMBLEX` comment and any stale wording for the previous layout model in `accuracy_reports.py` in line with `LABEL_MAP`.
  - The table benchmark and the false-table cohort feed ground-truth or whole-page rects to `reconstruct_table` and never run layout, so they are not layout gates. The layout gate on tables is end-to-end: tables emitted by `extract_text` on the FUNSD and DocLayNet scanned fixtures.
- **H-B, remaining half (before step 2).** CER between backends, now that P6 gives pdfium text to compare with fitz's. Until the harness runs pdfium as its second side, `BACKEND_PARITY.md` is a repeatability baseline of fitz against itself. PDFs are capped to 20 pages, and the DOCX/XLSX/CSV/XML/TXT fixtures never go through the PDF seam, so `tests/test_default_digest.py` gates them instead.
- **Report regeneration (step 2).** L3c-B: `REDACTION_HANDLING`, whose exclusion zones now come from the 200 dpi layout render. L1-B: `EXTRACTION` and `REDACTION_HANDLING` on the revised ground truth. H-B: CER against transcripts.
- **Hand checks (step 3).**
  - `REDACTION_HANDLING` read by hand; earlier reports are not a comparison.
  - Exclusion area spot-checked on the 02737-class scanned forms.
  - The swap's findings (recorded in `decisions.md`) re-checked: table-class recall fell from 50% to 25%, and `dense_text_548` gave three table regions where the ground truth has one, with a `chart` box (mapped to `figure`) almost identical to the `table` box. `LABEL_MAP` and the 0.3 threshold are the knobs. The `dense_text_548` table must be found.
  - End-to-end tables emitted on the scanned fixtures checked by hand.
  - DocLayNet F1 recorded as the floor later merges must not drop below.
- **F1-B (after F1 (W) is ready).** Regenerate `EXTRACTION`, `REDACTION_HANDLING`, `PII_CLEANING`, `READING_ORDER` and `CHUNKING`, plus the table and false-table suites, on the permissive backend. These are compared against this baseline under the F1 gates in [plan-permissive-deps.md](plan-permissive-deps.md).

## What this run closes
- L3c-B, L1-B and the transcript-CER half of H-B, as above.
- Absolute thresholds in the F1 gates that were set from earlier reports (the DocLayNet layout F1, the Auditor-General transcript CER) are reset from this baseline in the same merge. Between-backend gates (fitz against pdfium) are unaffected.
