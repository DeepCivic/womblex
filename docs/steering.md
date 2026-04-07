# Improvement Steering

Where the pipeline is today, what to work on next, and why. Updated as changes land.

See `accuracy/` for current benchmark numbers. See `architecture.md` for how the system works.

## Priority List

| # | Change | Effort | Impact | Status |
|---|--------|--------|--------|--------|
| 1 | Add sorted CER to FUNSD evaluation | Low | Reveals 65% of CER was reading-order, not recognition | **Done** |
| 2 | Add per-class layout P/R/F1 to DocLayNet | Low | Makes layout failures actionable | **Done** |
| 3 | Replace mean threshold with histogram analysis | Medium | DocLayNet avg CER pp −15.5% | **Done** |
| 4 | Wire `STRUCTURED` detection into `_classify()` | Medium | Routes table-heavy documents to `StructuredExtractor` | **Done** |
| 5 | Add strategy-selection log line | Low | Enables pipeline path tracing | **Done** |
| 6 | Integrate local models (all-MiniLM-L6-v2, yolov8n) | Low | No network access at inference time | **Done** |
| 7 | Programmatic accuracy doc generation | Low | Docs reflect actual last test run | **Done** |
| 8 | URL / phone / email PII regex | Low | Covers 6/12 GT Throsby entities (WEBSITE ×4, PHONE, EMAIL) at near-zero FP risk | |
| 9 | Adaptive binarisation second signal | Medium | CER-s shows binarisation hurts FUNSD by +39%; histogram alone is insufficient | |
| 10 | NER-based PII (Presidio Analyzer + spaCy) | Medium | Covers ORGANISATION (4 GT) + improves PERSON precision (currently 16.7%) | |
| 11 | Redaction threshold tuning for signature blocks | Low | 3/7 GT redactions missed on page 2 Throsby; aspect-ratio filter likely culprit | |
| 12 | Replace YOLO COCO model with document-specific layout model | High | YOLOv8n produces 0 predictions on all DocLayNet fixtures — general COCO model has no document layout classes | |
| 13 | ~~Layout class coverage (heading, footer, caption, figure)~~ | — | Subsumed by #12 — entire layout pipeline needs a document-trained model | **Merged into #12** |
| 14 | Per-document-type config overrides | High | Enables type-specific DPI, thresholds | |
| 15 | End-to-end task metrics (Isaacus integration) | High | Measures actual application success | |
| 16 | Handwriting via dedicated HTR model | High | Only if handwritten docs are in scope | |

## Findings by Component

### Classification

Two `DocumentType` values are still unreachable:

- `IMAGE` — no detection path produces it, scanned photos fall to `SCANNED_MACHINEWRITTEN`
- Forms — `_has_form_structure()` exists in `detect.py` but is never called

`STRUCTURED` is now reachable: documents where ≥80% of sampled pages contain table signals route to `StructuredExtractor`.

### Preprocessing

**Resolved:** Histogram-based binarisation skip correctly handles digital vs scanned. Dead heuristic code removed.

**Open:** Binarisation hurts recognition on FUNSD forms (CER-s raw 0.189 → pp 0.262, +39%). The histogram correctly identifies these as scanned, but Otsu binarisation degrades character shapes. The preprocessing decision may need a second signal — contrast quality or sample OCR confidence — to decide whether binarisation helps a particular scanned image.

### Layout Detection

Layout detection produces 0 predictions across all DocLayNet fixtures — 0% precision, recall, and F1 for every class. The YOLOv8n model (general-purpose COCO) produces no document-layout predictions. A document-specific layout model is needed for any layout analysis capability.

Per-class P/R/F1 is tracked in `accuracy/EXTRACTION.md`.

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

Measured on Throsby fixture (7 GT `<REDACTED>` tags across 3 pages).

- **Region recall: 57%** — 4/7 GT redactions detected. Page-level recall is 100% (the only affected page is found) but 3 redactions are missed.
- All 4 detected regions are on page 0 (header and inline name redactions). The 3 missed are on page 2 (signature block).
- Suspected cause: signature-block redaction has an unusual aspect ratio or falls outside the current area threshold (min 0.1%, max 90%). Worth loosening thresholds and re-testing.
- High false-positive risk on graphical documents — dark table borders and figure fills trigger the contour detector.

## Changelog

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
