# Improvement Steering

Where the pipeline is today, what to work on next, and why. Updated as changes land.

See `accuracy/` for current benchmark numbers. See `architecture.md` for how the system works.

## Priority List

> **Scope note.** This is the *extraction-quality* track. The per-stage
> pipeline (extract → chunk → redact → enrich → link → embed → PII) has landed;
> the durable design decisions, dead-ends, limitations and deferred backlog are
> in [decisions.md](decisions.md). The one remaining pipeline iteration is the
> downstream text-cleaning op (#B/#D in decisions.md). Items below are the older
> accuracy track; completed ones are marked.

| # | Change | Effort | Impact | Status |
|---|--------|--------|--------|--------|
| 1 | Add sorted CER to FUNSD evaluation | Low | Reveals 65% of CER was reading-order, not recognition | **Done** |
| 2 | Add per-class layout P/R/F1 to DocLayNet | Low | Makes layout failures actionable | **Done** |
| 3 | Replace mean threshold with histogram analysis | Medium | DocLayNet avg CER pp −15.5% | **Done** |
| 4 | Wire `STRUCTURED` detection into `_classify()` | Medium | Surfaces table-heavy documents as a doc-level summary type (per-page routing handles per-region structure) | **Done — superseded by per-page orchestrator** |
| 5 | Add strategy-selection log line | Low | Enables pipeline path tracing | **Done** |
| 6 | Integrate local models (all-MiniLM-L6-v2, yolov8n) | Low | No network access at inference time | **Done** |
| 7 | Programmatic accuracy doc generation | Low | Docs reflect actual last test run | **Done** |
| 8 | URL / phone / email PII regex | Low | Covers 6/12 GT Throsby entities (WEBSITE ×4, PHONE, EMAIL) at near-zero FP risk | |
| 9 | Adaptive binarisation second signal | Medium | CER-s shows binarisation hurts FUNSD by +39%; histogram alone is insufficient | |
| 10 | NER-based PII (Presidio Analyzer + spaCy) | Medium | Covers ORGANISATION (4 GT) + improves PERSON precision (currently 16.7%) | |
| 11 | Redaction threshold tuning for signature blocks | Low | 3/7 GT redactions missed on page 2 Throsby; aspect-ratio filter likely culprit | |
| 12 | Replace YOLO COCO model with document-specific layout model | High | YOLOv8n produces 0 predictions on all DocLayNet fixtures — general COCO model has no document layout classes | **Done — K7(b) DocLayNet `yolo11n_doc_layout.pt` swap, 2026-05-25** |
| 13 | ~~Layout class coverage (heading, footer, caption, figure)~~ | — | Subsumed by #12 — entire layout pipeline needs a document-trained model | **Merged into #12** |
| 14 | Per-document-type config overrides | High | Enables type-specific DPI, thresholds | |
| 15 | End-to-end task metrics (Isaacus integration) | High | Measures actual application success | **In progress — I6/I7 landed Isaacus enrich/link/embed stages; PII (I8) + coverage metrics next** |
| 16 | Handwriting via dedicated HTR model | High | Only if handwritten docs are in scope | |
| 17 | Table-cell reconstruction for OCR'd pages | Medium | Unblocks every structural consumer on scanned documents. Measured: a scanned money table yields 1 amount of ~35 today, 30 with cells | **Done** |

## Findings by Component

### Classification

Two `DocumentType` values are still unreachable:

- `IMAGE` — no detection path produces it, scanned photos fall to `SCANNED_MACHINEWRITTEN`
- Forms — `_has_form_structure()` exists in `detect.py` but is never called

`STRUCTURED` is reachable as a doc-level summary type — documents where ≥80% of sampled pages contain table signals classify as `STRUCTURED`. The doc-level strategy classes (`StructuredExtractor` etc.) have since been removed; the per-page orchestrator (`ingest/orchestrator.py`) dispatches operations page-by-page based on `PageProfile`, and the `spreadsheet_print` extractor runs behind a `qualify_for_spreadsheet_print` gate when the manifest signal fires.

### Preprocessing

**Resolved:** Histogram-based binarisation skip correctly handles digital vs scanned. Dead heuristic code removed.

**Open:** Binarisation hurts recognition on FUNSD forms (CER-s raw 0.189 → pp 0.262, +39%). The histogram correctly identifies these as scanned, but Otsu binarisation degrades character shapes. The preprocessing decision may need a second signal — contrast quality or sample OCR confidence — to decide whether binarisation helps a particular scanned image.

### Layout Detection

**Resolved (model): the document-specific swap landed** (#12, `yolo11n_doc_layout.pt`, 2026-05-25) — the earlier "0 predictions across all DocLayNet fixtures" finding described the general-purpose COCO YOLOv8n and is obsolete. The current model does detect document layout, including tables: on `dense_text_548` it returns a `table` region at 0.96 confidence.

**Resolved (metric): the reported 25% table recall was largely a GT-aggregation artefact** (B0, 2026-07-28). `_aggregate_doclaynet_blocks` groups *consecutive* same-label word spans, so two stray 1-word Table-labelled footnote lines in `dense_text_548` split the real 397-word table run into three GT blocks, each unmatched stray charged as a separate false negative. GT Table blocks are now filtered by a minimum span count (`MIN_TABLE_GT_SPANS = 3`) before matching. Note also `table_0` contains no Table-labelled GT at all (196 Text, 2 Section-header, 1 Page-footer) — despite the name it is not a table fixture and serves as a false-table (no-GT) fixture instead.

Per-class P/R/F1 is tracked in `accuracy/EXTRACTION.md`.

### Table-cell reconstruction on OCR'd pages (#17) — **Done**

**A detected table region now becomes cells.** Found while validating the
`money` op against real documents ([money-extraction.md](money-extraction.md),
"First real-document run"); the fix is entirely extraction-side and benefits
every structural consumer, not just money.

The original failure, on DocLayNet `dense_text_548` (a scanned
*Grants of Plan-Based Awards* page, four money columns, ~35 amounts): OCR read
the page well and the layout model found the table (0.96 confidence), but
`_layout_blocks_and_tables` turned that region into a
`TextBlock(text="[TABLE]", block_type="table")` and returned an empty `tables`
list — so the shard carried no `table` element and no `table_cells` rows, and
every downstream consumer of structure saw a single page-wide paragraph.

What shipped:

- **Reconstruct cells within a detected table region.** The grid algorithm was
  lifted from `ingest/spreadsheet_print.py` — the real prior art — into the
  shared `ingest/table_grid.py`, and `ingest/ocr_tables.py` feeds it OCR quads
  via `reconstruct_table(regions, table_rect, dpi, conf)`. The layout pass now
  emits a cellified `table` element on OCR'd PDF pages (the page narrative is
  rebuilt from the regions outside the rect, so the table isn't also chunked as
  prose). Deskewed pages refuse outright. The precision gates are calibrated
  (`MIN_ROW_FILL_RATIO=0.75`): the rendered-clean cohort reconstructs, and the
  hard and false-table shapes refuse. A hard-shape table (stacked spanning
  headers, hierarchical rows) **refuses** rather than producing a low-quality
  grid — no cells is better than wrong cells. The mechanism and refusal
  rationale are in [decisions.md](decisions.md) “Table-cell reconstruction on
  OCR pages”.
- **Standalone images need no separate route.** `extract_text` gates the legacy
  path-based dispatch on `(SPREADSHEET, DOCX, TEXT)`; `IMAGE` is not in it and
  falls through to `fitz.open()` + `extract_pdf_with_plan`, since PyMuPDF opens
  an image as a one-page document. Images have always gone through
  `_apply_ocr_page`, so the same wiring fixed them alongside scanned PDFs —
  verified by driving a real `.png` through `extract_text` and observing a
  cellified `table` element with `context_producer=table_grid`. The unreachable
  `ImageExtractor` and `get_extractor`'s dead `IMAGE` case were deleted, and the
  routing is pinned by `TestImageDocumentsRouteThroughTheOrchestrator`.

**Measured payoff.** Feeding the same page's real grid through the money op's
column classifier recovers 30 of 30 amounts, versus 1 today — the consuming
stage is already ready for the structure. More broadly this is the only route
to the column-evidenced path on scanned documents, which
[money-extraction.md](money-extraction.md) measures as where the overwhelming
majority of the corpus's monetary amounts live.

**Measurement.** The reconstructor is measured with `utils/tabular_metrics.py`
(structural fidelity, data integrity) over rendered-clean and DocLayNet
fixtures, conditioned on a correct table rect (detection is scored separately
as the per-class layout F1). The benchmark (`tests/test_table_benchmark.py`)
surfaces into `accuracy/EXTRACTION.md` via a `## Table Reconstruction` section,
and the rendered-clean / false-table asserts are build-failing gates. The
metric set, the two-stage decomposition and the ground-truth authoring spec
are documented in [evaluation.md](evaluation.md) §2b. Precision matters more
than coverage here: a wrongly-binned grid produces confidently wrong values
downstream, which is worse than honest silence.

### Reading Order

**Resolved.** CER-s (sorted CER) now separates recognition from reading-order accuracy. 65% of FUNSD sequential CER was ordering mismatch.

### Handwriting

PaddleOCR v4 cannot recognise handwriting (IAM WER 1.000). Not worth investing unless handwritten document support becomes a requirement. Add a dedicated HTR model behind `SCANNED_HANDWRITTEN` if needed.

### Pipeline Observability

**Resolved.** `extract_text()` now logs `strategy selected: doc=<name> type=<type> confidence=<conf> strategy=<class>` for every document. Visible at INFO level.

### PII Cleaning

Measured on Throsby fixture (12 GT entities across 6 types). Only PERSON is currently detected (regex + `all-MiniLM-L6-v2` context validation).

| Entity Type | GT | Supported | Notes |
|-------------|-----|-----------|-------|
| ORGANISATION | 4 | No | Needs NER |
| WEBSITE | 4 | No | URL regex — low effort |
| PHONE | 1 | No | Phone regex — low effort |
| EMAIL | 1 | No | Email regex — low effort |
| ADDRESS | 1 | No | Address regex or NER |
| PERSON | 1 | Yes | Recall 100%, precision 16.7% (5 FP) |

**Open issues:**
- 11/12 GT entities unsupported. URL/phone/email regex would close 6 of those for minimal effort.
- PERSON precision of 16.7% (1 TP, 5 FP) — false positives come from OCR artefacts, state abbreviations, and partial organisation name fragments that escape `_COMMON_WORDS` filtering. Uniform regulatory vocabulary makes cosine similarity poorly discriminative at the 0.35 threshold.
- NER via Presidio Analyzer + spaCy would handle ORGANISATION and improve PERSON precision, but adds a large dependency. Assess against real-document PII inventory before adding.

### Redaction Handling

Measured on Throsby fixture (7 GT `<REDACTED>` tags across 3 pages); vector-first detection is described in [decisions.md](decisions.md) "Redaction detection" and [CHANGELOG.md](../CHANGELOG.md).

- **Native cohort recall significantly improved post vector-first detection.** `redact/stage.py:detect_redactions` now tries `page.get_drawings()` for filled near-black rectangles before falling back to the raster CV2 contour detector. On the §1 residual pages (01093 / 01094 / 01349) recall jumped 6→14, 7→13, 3→68 without regressing FOI master (0 regions preserved).
- **Filters** (each surfaced during validation): near-black RGB/CMYK fill; `min_width ≥ 3pt` excludes narrow vertical separators in manifest tables; `min_height ≥ 8pt` excludes glyph-rendering small filled rects on PDFs that draw text as filled-path glyphs (01125-class regression: 14,184 false positives → 144 actual).
- **Open — scanned/raster cohort precision.** Direct-Complaint forms with dark form-field backgrounds (02737-class scanned_mixed docs) still trigger the area-threshold contour detector even with `max_area_ratio=0.05`. Higher precision on this cohort would need a different detection signal (e.g. layout-aware classes that distinguish form fields from redaction bars). See `stories/STATUS.md` §11.

## Changelog

### 2026-07-29: #17 complete — table reconstruction on OCR'd pages

Table-cell reconstruction (#17) is fully landed: the OCR paths emit cellified
`table` elements, the precision gates are calibrated, the benchmark surfaces
into `accuracy/EXTRACTION.md`, and the rendered-clean / false-table asserts are
build-failing CI gates (`dense_text_548` stays tracking, ungated). The
implementation plan (formerly `table-cell-reconstruction-plan.md`) has been
folded into the standard docs and removed: the mechanism and refusal rationale
live in [decisions.md](decisions.md) “Table-cell reconstruction on OCR pages”,
the metric set + two-stage decomposition + ground-truth authoring spec in
[evaluation.md](evaluation.md) §2b, and the component status above.

### 2026-07-29: B4 — table reconstruction report + docs wiring

Wired the table benchmark's results into the generated accuracy docs. A
`## Table Reconstruction` section (`tests/accuracy_reports.py`) now sits
under the DocLayNet per-class layout section it decomposes — detection is
stage 1 there, reconstruction (conditioned on a correct rect, B1.2) is this
section — rendering three cohorts: rendered-clean (gated), the
`dense_text_548` tracking row, and a separate false-table table headed by
the live false-positive count. `test_fixture_accuracy._results` gained a
`"tables"` key that `tests/test_table_benchmark.py` aliases, so its entries
flow into the existing session `write_report` finaliser with no duplicate
plumbing. Money recall is deliberately omitted (no labelled money GT — see
[money-extraction.md](money-extraction.md)) rather than fabricated;
CHUNKING.md's table knock-on is noted
(its generator is unwritten and numbers predate tables on OCR pages).
`evaluation.md` gained §2b (Document-Table Reconstruction Accuracy),
distinct from §2's spreadsheet→parquet. Only B5 (CI gates) remains. See
[evaluation.md](evaluation.md) §2b.

### 2026-07-29: B2 — precision-gate calibration + benchmark metric set

Calibrated the OCR table reconstructor's precision gates against the
rendered-clean cohort (must pass) and a false-table cohort (must refuse).
Added `MIN_ROW_FILL_RATIO = 0.75` to `ingest/ocr_tables.py` — mean cell
occupancy across the reconstructed body. Measured, the two populations do
not overlap: rendered-clean fill 0.98–1.00, false/hard shapes 0.375–0.49.
The gate is a clean sweep — all six rendered-clean fixtures reconstruct,
all eight false-table probes (3 non-table DocLayNet pages + 5 FUNSD forms)
refuse, and `dense_text_548` now **refuses** rather than emitting the
pre-B2 12×12 partial against its 39×11 GT. Three false positives the
provisional gates had let through (`diverse_layout_49`,
`funsd/82200067_0069`, `funsd/87528321`) are closed. The right-edge
overflow signal the plan asked about measured 0 everywhere
(`column_for_x` absorbs right content), so density — not assigned-ratio
symmetry — is the guardrail. Benchmark additions in
`tests/test_table_benchmark.py`: an alignment projection feeding
`utils/tabular_metrics.py` (`structural_fidelity` + `data_integrity`), the
false-table cohort (FP count gates), and the GT acceptance
checker. See [evaluation.md](evaluation.md) §2b.

### 2026-07-28: B0 — table metric fixed before measuring reconstruction against it

Corrected the stale Layout Detection finding (the "0 predictions" claim
described the pre-#12 COCO model) and fixed the DocLayNet table-class GT
aggregation: stray sub-`MIN_TABLE_GT_SPANS` Table-labelled runs (footnote
lines mislabelled Table in `dense_text_548`) are dropped from the GT instead
of each being charged as a false negative. Recorded that `table_0` carries no
Table-labelled GT and is a false-table fixture, not a table fixture.
Verified by a layout-only probe replicating the suite's computation: table
class TP1/FP0/FN3 (R 25%, F1 40%) → TP1/FP0/FN1 (R 50%, F1 66.7%), the
remaining FN being `sparse_text_344`'s genuinely undetected 8-word block; all
other classes unchanged. `docs/accuracy/EXTRACTION.md` still shows the
pre-fix numbers until the next full accuracy-suite run regenerates it. See
[evaluation.md](evaluation.md) §2b.

### 2026-03-22: Benchmark test performance + stale findings cleanup

Added `max_pages=30` to `extract_text()` for PDF extraction — tests now evaluate only the first 30 pages of large documents. Replaced `rapidfuzz` C backend with pure-Python Levenshtein for CER/WER (no external dependency). Ground truth is proportionally truncated when page-limited.

Updated layout detection findings: YOLOv8n (COCO) produces 0 predictions across all DocLayNet fixtures. Previous steering entries referencing per-fixture prediction counts and over/under-segmentation modes were stale. Merged items #12 and #13 — both require replacing the COCO model with a document-trained layout model.

### 2026-03-22: Programmatic accuracy doc generation + model integration

Rewrote `test_womblex_collection_accuracy.py` to accumulate measured values during test execution and write `REDACTION_HANDLING.md` and `PII_CLEANING.md` at session end via an autouse fixture finaliser. Docs are no longer manually maintained — running the test suite regenerates them.

Added local model path resolution (`utils/models.py`): `all-MiniLM-L6-v2` and `yolov8n.pt` load from `models/` directory without network access. `YOLOLayoutAnalyzer` added in `paddle_ocr.py` as the layout backend (replaced rapid-layout).

PII regex fixes: changed `\s+` to `[^\S\n]+` in `_TITLE_CASE_RE` and `_HONORIFIC_RE` to prevent multi-line span capture. Default context similarity threshold lowered from 0.5 to 0.35 after empirical calibration on Throsby fixture.

First measured PII baseline: PERSON recall 100% (1/1), precision 16.7% (1 TP, 5 FP). Entity-type coverage: 1/6 types supported, 1/12 GT entities.

### 2026-03-22: Strategy-selection logging and STRUCTURED detection

Added INFO-level log line in `extract_text()` recording `{doc, type, confidence, strategy}` for every document processed. Enables tracing which detection → strategy path each document takes.

Wired `STRUCTURED` detection into `_classify()`: documents where ≥80% of sampled pages contain table signals (regex patterns, PyMuPDF table finder) now route to `StructuredExtractor` instead of `NATIVE_WITH_STRUCTURED`. Table signal counting changed from boolean (first-page-wins) to per-page count, enabling table coverage ratio.

### 2026-03-22: Per-class layout P/R/F1

Replaced detection-rate (recall-only) and label-accuracy with proper precision, recall, and F1 per class. Current results show 0 predictions from YOLOv8n across all fixtures — a document-specific layout model is needed.

### 2026-03-22: Sorted CER for FUNSD

Added CER-s — spatially sorted CER that isolates recognition from reading-order accuracy. Average CER-s raw 0.189 vs sequential CER 0.536, confirming that most error was ordering mismatch.

Binarisation hurts FUNSD recognition: CER-s raw 0.189 → pp 0.262 (+39%).

### 2026-03-22: Histogram-based binarisation skip

Replaced `mean > 240` pixel threshold with `analyze_histogram()`. DocLayNet avg CER pp improved from 0.297 to 0.251 (−15.5%).
