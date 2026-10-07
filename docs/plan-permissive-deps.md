# Permissive dependencies — outstanding work

*Status: in progress (2026-10). Outstanding: P7, P8, F1, F2 (PyMuPDF) and L3c-B, L1-B (layout). Everything shipped is recorded in [`CHANGELOG.md`](../CHANGELOG.md), [`architecture.md`](architecture.md), [`models.md`](models.md) and [`decisions.md`](decisions.md), not here. The ground-truth revision gates only L1-B, F1-B's regeneration and H-B's CER-against-transcripts half; everything else is deliverable now. L3 is independent of the rest. Each merge updates this list as it lands, and the document is deleted once F2 ships.*

## Context
Womblex is Apache-2.0, so its dependencies must be licence-compatible. The remaining incompatible one is `pymupdf` (`import fitz`; opens every PDF and standalone image) and it is being replaced: every extractor already reads through the `ingest/pdf/` seam, `fitz` is imported only in `ingest/pdf/_fitz.py`, and the pdfium backend (`backend="pdfium"`) opens PDFs and images with geometry, rendering, images, drawings, widgets and text. Its `find_tables` raises `NotImplementedError` until P7. The default backend is still fitz.

A merge is W (womblex) or B (womblex-benchmark, paired). An approval tag means the merge edits `pyproject.toml` and needs human sign-off.

## Layout
- **L3. Layout as its own stage.** Design settled 2026-10; merges below.

  *Today.* Layout runs twice, unpersisted: inside the OCR page operation (`extraction.ocr.layout_model`, OCR dpi, paddleocr branch only) for table rects and the collapsed block's `block_type`, and inside redaction detection (`redaction.layout_model`, redaction dpi, raster-fallback pages) for `figure` / `table` exclusion zones. A page can get two models at two resolutions, and neither result can be inspected or rerun.

  *Decisions.*
  - **One model, one setting.** A top-level `layout:` section (`model`, `options`, `page_scope`) chosen through the registry's layout slot; a tuned model is a `womblex.models.layout` plugin. `extraction.ocr.layout_model` / `layout_options` and `redaction.layout_model` / `layout_options` are deleted outright (pre-1.0 breaking change, recorded in the CHANGELOG). `redaction.use_layout_filter` stays: it decides whether redaction consumes the regions, not which model runs. `model_check` gets a `layout` scope in place of the slot appearing under both `extract` and redaction.
  - **Runs in-batch, before OCR.** Order: profile → layout → OCR → redaction detection. The step lives in the orchestrator (it needs each page's route), carries its regions on `ExtractionResult`, and `write_results` writes the sidecar; redaction reads the same in-memory regions rather than re-running anything. Only documents the PDF seam opens (PDFs and images) get layout; DOCX, spreadsheets, text and the records ingest write no rows. OCR table reconstruction and redaction read the persisted regions instead of calling an analyser. The rejected alternative (persist OCR regions, reconstruct tables downstream) would make OCR-page table elements depend on a downstream stage, needing an element-structure overlay. Distributed workers already stage sources for a batch, so this needs nothing new there.
  - **Page scope.** `layout.page_scope`: `consumers` (default) or `all`. `consumers` is the OCR-routed pages (excluding engines whose registry entry carries the `markdown` trait, which bypass layout today) plus, when redaction and its layout filter are on, the pages without vector redactions. Both are known before layout runs (the vector test reads only drawings), so there is no circular dependency. `all` is the benchmark-coverage setting. Skipping pure-vector pages is a consequence of `consumers`, not a separate rule.
  - **Sidecar.** `store/layout_output.py`, `*.layout_regions.parquet`, one row per region keyed by `(source_hash, page)`: `region_order`, `bbox` (normalised 0–1, top-left, the element `BBox` convention, float32), `label`, `block_type`, `confidence` (always present: `check_layout_regions` requires it), `status` (`ok` / `empty` / `error`) and `error`. Pages that ran and found nothing, and pages where layout failed, get one status row each, so the two are never conflated. Strings dictionary-encoded. The footer carries a layout fingerprint (model name, model digest, options digest, render dpi, page scope, schema version), since a standalone rerun cannot inherit the extraction run's stamp, and whether redaction consumed the regions (`redaction.enabled` and `use_layout_filter`). Contract sensitivity `none` (no text); an additive minor bump to `CONTRACT_VERSION` (`1.1`) and `contract.md`. In `PRODUCER_OF` it maps to `layout`; it never makes a base discoverable.
  - **Geometry.** Layout renders at `extraction.ocr.dpi`, so its page and OCR's page are the same render size, as today. Normalised boxes let OCR and redaction (150 dpi) convert to their own render through `render_box`, which replaces today's OCR-versus-layout pixel-dimension check. Layout renders the raw page; the A2 deskew refusal stays with OCR, which alone knows it deskewed.
  - **Failure.** A page whose layout failed (`status=error`) falls back to whole-page OCR text, as today; the document continues. On such a page with `use_layout_filter` on, redaction runs without exclusion zones and logs a warning naming the document and page. Nothing new is stored: an unfiltered page is one whose layout row has `status=error` in a sidecar whose footer says redaction consumed the regions. This over-reports slightly (a page redaction resolved from vector drawings never needed the filter), which is the safe direction. `store/layout_output.py` exposes the one reader for it, and the help doc documents it.
  - **Reruns.** The elements footer records the layout fingerprint extraction consumed. `run-stage layout` replaces only the layout sidecar; elements, tables and redaction results stay as extracted, and a fingerprint mismatch between the two footers is the out-of-date signal for anything reading both (benchmark, console, shard verification). No stage reads layout after extraction, so no stale-output tracking is added. The skip rule compares fingerprints: same model and settings skip, a change reruns. Applying a new model to tables or redaction is a re-extract. A run extracted before L3 has no fingerprint in its elements footer; a rerun over it reads as "provenance unknown", not a mismatch. Layout stays out of `DOWNSTREAM_STAGES`: a rerun is deliberate, as `pii` is.
  - **Help doc.** A user-facing `docs/layout.md`, linked from the README's documentation table, grows with each merge: what layout does and its settings; the sidecar and how to read it; how to list unfiltered redaction pages reliably (the reader, and the equivalent Parquet query with the footer check it depends on); how to tell whether a shard's layout file matches what its elements were built from, including runs from before L3; and how to rerun layout locally and on a distributed run.
  - **Measurement without OCR.** `run-stage layout` renders pages from the source documents, the first downstream stage to need them: locally through `SourceResolver`, distributed through source staging in the stage runner.

  *Merges.*
  - **L3c-B (B).** Regenerate `REDACTION_HANDLING`: exclusion zones now come from the 200 dpi layout render (the W half, L3c, has shipped).

- **L1-B (B). Waits on the ground-truth revision.** Scores PP-DocLayout-M against the revised ground truth.
  - Make the DocLayNet layout-F1 test honour `--model` (it calls `get_layout_analyzer()` with no arguments today).
  - Bring the `DOCLAYNET_TO_WOMBLEX` comment and any stale wording for the previous layout model in `accuracy_reports.py` in line with `LABEL_MAP`.
  - The table benchmark and the false-table cohort feed ground-truth or whole-page rects to `reconstruct_table` and never run layout, so they are not layout gates. The layout gate on tables is end-to-end: tables emitted by `extract_text` on the FUNSD and DocLayNet scanned fixtures.
  - Check the findings from the swap (recorded in `decisions.md`): table-class recall fell from 50% to 25%, and `dense_text_548` gave three table regions where the ground truth has one, with a `chart` box (mapped to `figure`) almost identical to the `table` box. `LABEL_MAP` and the 0.3 threshold are the knobs.
  - Regenerate `EXTRACTION.md` and `REDACTION_HANDLING.md`, and spot-check exclusion area on the 02737-class scanned forms.

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
| L3c-B | No regression in `REDACTION_HANDLING` beyond the declared render-dpi change, reviewed by hand |
| L1-B | DocLayNet F1 ≥ 0.29 with the `dense_text_548` table found; end-to-end tables emitted on the scanned fixtures checked by hand; no regression in `REDACTION_HANDLING` |

F1 flips the default only when all of these hold:

| Measure | Required |
|---|---|
| New open failures | Zero |
| Doc type | ≥ 98% agreement |
| Per-page plan operation | ≥ 99% agreement |
| `has_text_layer` | 100% agreement |
| Native-text CER between backends | Median ≤ 0.01, p95 ≤ 0.03 |
| Auditor-General transcript CER | ≤ 0.216 (currently 0.211) |
| ACT-ECI CER | No strategy worse by more than 0.01 |
| Table count | Equal on ≥ 95% of table pages |
| Table-benchmark F1 | Drops by at most 0.01 |
| False-table cohort | Does not grow |
| Vector-redaction counts | Identical |
| AcroForm and FUNSD field counts | Equal |
| `CHUNKING`, `READING_ORDER`, `PII_CLEANING`, `REDACTION_HANDLING` | No regression |
| Native extraction | ≤ 1.5× the fitz wall time |
