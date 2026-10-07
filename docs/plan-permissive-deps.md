# Permissive dependencies — outstanding work

*Status: in progress (2026-10). Outstanding: P7, P8, F1, F2 (PyMuPDF) and L3c-B, L1-B (layout). Everything shipped is recorded in [`CHANGELOG.md`](../CHANGELOG.md), [`architecture.md`](architecture.md), [`models.md`](models.md) and [`decisions.md`](decisions.md), not here. The ground-truth revision gates only L1-B, F1-B's regeneration and H-B's CER-against-transcripts half; their report regeneration runs once, as the fresh baseline in [`plan-post-gt-baseline.md`](plan-post-gt-baseline.md). Everything else is deliverable now. Each merge updates this list as it lands, and the document is deleted once F2 ships.*

## Context
Womblex is Apache-2.0, so its dependencies must be licence-compatible. The remaining incompatible one is `pymupdf` (`import fitz`; opens every PDF and standalone image) and it is being replaced: every extractor already reads through the `ingest/pdf/` seam, `fitz` is imported only in `ingest/pdf/_fitz.py`, and the pdfium backend (`backend="pdfium"`) opens PDFs and images with geometry, rendering, images, drawings, widgets and text. Its `find_tables` raises `NotImplementedError` until P7. The default backend is still fitz.

A merge is W (womblex) or B (womblex-benchmark, paired). An approval tag means the merge edits `pyproject.toml` and needs human sign-off.

## Layout
The layout stage (L3) has shipped; [`layout.md`](layout.md) documents it and [`decisions.md`](decisions.md) records why it is shaped as it is. What remains is benchmark-side.

- **L3c-B (B).** Regenerate `REDACTION_HANDLING`: exclusion zones now come from the 200 dpi layout render. Runs in the post-GT baseline.

- **L1-B (B). Waits on the ground-truth revision.** Scores PP-DocLayout-M against the revised ground truth.
  - Make the DocLayNet layout-F1 test honour `--model` (it calls `get_layout_analyzer()` with no arguments today).
  - Bring the `DOCLAYNET_TO_WOMBLEX` comment and any stale wording for the previous layout model in `accuracy_reports.py` in line with `LABEL_MAP`.
  - The table benchmark and the false-table cohort feed ground-truth or whole-page rects to `reconstruct_table` and never run layout, so they are not layout gates. The layout gate on tables is end-to-end: tables emitted by `extract_text` on the FUNSD and DocLayNet scanned fixtures.
  - Check the findings from the swap (recorded in `decisions.md`): table-class recall fell from 50% to 25%, and `dense_text_548` gave three table regions where the ground truth has one, with a `chart` box (mapped to `figure`) almost identical to the `table` box. `LABEL_MAP` and the 0.3 threshold are the knobs.
  - Regenerate `EXTRACTION.md` and `REDACTION_HANDLING.md` (in the post-GT baseline), and spot-check exclusion area on the 02737-class scanned forms.

## PyMuPDF
- **H-B, remaining half (B).** CER between backends, now that P6 gives pdfium text to compare with fitz's; and CER against transcripts, which waits on the ground-truth revision. Until the harness runs pdfium as its second side, `BACKEND_PARITY.md` is a repeatability baseline of fitz against itself. PDFs are capped to 20 pages, and the DOCX/XLSX/CSV/XML/TXT fixtures never go through the PDF seam, so `tests/test_default_digest.py` gates them instead.
- **P7 (W).** `ingest/pdf/_tables.py`, an adapter onto pdfplumber's `TableFinder` (lines and text strategies). `page_profile` calls `find_tables` on every page, so time per page matters.
- **P8… (W).** Fidelity fixes driven by `BACKEND_PARITY.md`, repeated until the gates below hold. Known inputs:
  - Multi-column reading order in the pdfium text engine: Phase 0's divergence tail (see `decisions.md`), left to P8 by P6.
  - MuPDF's `get_drawings` also reports annotation and widget appearance streams, which pdfium's page objects do not hold (`FPDFAnnot_GetObject` reaches them, in appearance space). No vendored fixture is affected.
  - Rotated spreadsheet-print pages: `Rect.transform` is double precision where fitz rounds to float32 (up to 1.5e-5 pt). The womblex-collection run through H-B is still owed.
  - A JPEG 2000 or PSD that declares a resolution is not yet checked against MuPDF's page rect.
- **F1 (W) + F1-B (B).**
  - Flip `open_document`'s default to the permissive backend.
  - Regenerate `EXTRACTION`, `REDACTION_HANDLING`, `PII_CLEANING`, `READING_ORDER` and `CHUNKING`, plus the table and false-table suites.
  - Re-pin `tests/test_default_digest.py` deliberately, and move its version guard from `fitz.VersionBind` to the pypdfium2 version.
- **F2 (W, approval).**
  - Delete `_fitz.py` and the backend argument, and drop `pymupdf`, its mypy override and the `pymupdf_layout` warning filter.
  - Simplify the CI fitz-notice workaround and the Dockerfile comments.
  - Rewrite `CLAUDE.md`: the PyMuPDF-import and dehyphenation pitfalls, and the rule that "everything fitz can open routes through `extract_text`".
  - Update `dataflow`, `heuristics_disambiguation` and the README.
  - Move the declared behaviour changes below into `decisions.md`, and delete this document.

**Out of scope, kept as-is for parity:** the unrotated-text versus rotated-`page.rect` mismatch outside `spreadsheet_print`.

## Open question
- **Record the PDF library version in the run stamp?** `content_digest` depends on the PDF library's version, but no footer records it. Options: add it to the stamp; raise it as a separate requirement; or leave it, since the womblex version and `uv.lock` already pin the library.

## Conventions this plan holds to
- **Merge size.** At most 500 changed lines per merge, green on ruff, mypy and pytest. `uv.lock` doesn't count. Any merge that grows splits again before review.
- **File size.** At most 750 lines per file. `ingest/pdf/` is split by concern.
- **Thin adapters.** Library-native behaviour is preferred (pdfplumber `TableFinder`). Womblex code covers only coordinates, typing and the fitz-compatible shapes its callers need.
- **No toggles.** Backend selection is a private `open_document` argument used by H-B, deleted at F2. The `Page` / `Document` protocols stay afterwards because PDF and image pages remain two implementations.
- **Docs move with the code.** Every merge that adds, moves or retires a module updates `architecture.md`, `project-structure.md` and the `CLAUDE.md` module table in the same PR, adds a CHANGELOG entry and ticks itself off here.
- **Verbatim text.** Dehyphenation and segmentation are extractor behaviour, so they live in the backend, never as a post-pass.
- **Unusual input warns and continues.** A format the new backend cannot open (fitz also opened XPS, EPUB, MOBI, CBZ and SVG) becomes a per-document error status with the document ID logged. It never aborts the batch.
- **Benchmark boundary.** All scoring against womblex-collection ground truth, and every `docs/accuracy/` report, is produced in womblex-benchmark.

## Declared behaviour changes (at F1)
- **Fewer formats.** Calling `extract_text` directly can no longer open XPS, EPUB, MOBI, CBZ or SVG, nor an image format MuPDF decodes and Pillow does not. PDF and the common image formats remain, and WebP and AVIF are new. None of the dropped formats is reachable through the CLI or API today.
- **New content digests.** `content_digest` changes at F1, and `tests/test_default_digest.py` is re-pinned in that merge. Text stays verbatim, but it comes from a different producer.
- **Small text differences.** Paragraph segmentation, dehyphenation and bold-based heading detection may shift slightly.
- **Image counts.** `image_count` may count drawn images only, which moves the sub-page OCR gate.
- **Rendering.** Anti-aliasing differs, so OCR output differs slightly.
- **Speed.** The table pass may be slower; the gate bounds it.

## Gates
Migration gates, not quality scores. They retire with this plan.

| Merge | Gate |
|---|---|
| Every merge | `uv run ruff check src/ tests/`, `uv run mypy src/` and `uv run python -m pytest tests/ -v` pass; `uv lock --check` passes on approval merges; touched files are under 750 lines (`wc -l`); `git diff --stat $(git merge-base HEAD origin/main)..HEAD` is within the cap |
| L3c-B | `REDACTION_HANDLING` checked by hand in the post-GT baseline; earlier reports are not a comparison |
| L1-B | The `dense_text_548` table found; DocLayNet F1 recorded by the post-GT baseline, which later merges must not drop below; end-to-end tables emitted on the scanned fixtures checked by hand |

F1 flips the default only when all of these hold:

| Measure | Required |
|---|---|
| New open failures | Zero |
| Doc type | ≥ 98% agreement |
| Per-page plan operation | ≥ 99% agreement |
| `has_text_layer` | 100% agreement |
| Native-text CER between backends | Median ≤ 0.01, p95 ≤ 0.03 |
| Auditor-General transcript CER | Within 0.005 of the post-GT baseline |
| ACT-ECI CER | No strategy worse by more than 0.01 |
| Table count | Equal on ≥ 95% of table pages |
| Table-benchmark F1 | Drops by at most 0.01 |
| False-table cohort | Does not grow |
| Vector-redaction counts | Identical |
| AcroForm and FUNSD field counts | Equal |
| `CHUNKING`, `READING_ORDER`, `PII_CLEANING`, `REDACTION_HANDLING` | No regression from the post-GT baseline |
| Native extraction | ≤ 1.5× the fitz wall time |
