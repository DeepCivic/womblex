# Improvement Steering

Where the pipeline is today, what to work on next, and why. Updated as changes land.

## Priority List

> **Scope note.** This is the *extraction-quality* track. The per-stage
> pipeline has landed, including the downstream text-cleaning op
> (`womblex normalise` + `spellfix`, selected by `processing.text_source`).
> Items below are the older accuracy track; completed ones are marked.

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
| 12 | Replace YOLO COCO model with document-specific layout model | High | YOLOv8n produces 0 predictions on all DocLayNet fixtures — general COCO model has no document layout classes | **Done — K7(b) DocLayNet `yolo11n_doc_layout.pt` swap, 2026-05-25; since replaced by PP-DocLayout-M (#130)** |
| 13 | ~~Layout class coverage (heading, footer, caption, figure)~~ | — | Subsumed by #12 — entire layout pipeline needs a document-trained model | **Merged into #12** |
| 14 | Per-document-type config overrides | High | Enables type-specific DPI, thresholds | |
| 15 | End-to-end task metrics (Isaacus integration) | High | Measures actual application success | **In progress — I6-I10 landed (enrich/link/embed + graph-driven PII); end-to-end coverage metrics still pending** |
| 16 | Handwriting via dedicated HTR model | High | Only if handwritten docs are in scope | |
| 17 | Table-cell reconstruction for OCR'd pages | Medium | Unblocks every structural consumer on scanned documents. Measured: a scanned money table yields 1 amount of ~35 today, 30 with cells | **Done** |

## Findings by Component

### Classification

Two `DocumentType` values are still unreachable:

- `IMAGE` — no detection path produces it, scanned photos fall to `SCANNED_MACHINEWRITTEN`
- Forms — no form `DocumentType`; `_has_form_structure()` (in `detect.py`) only sets the per-page `has_form_signal` in `page_profile.py`

`STRUCTURED` is reachable as a doc-level summary type — documents where ≥80% of sampled pages contain table signals classify as `STRUCTURED`. The doc-level strategy classes (`StructuredExtractor` etc.) have since been removed; the per-page orchestrator (`ingest/orchestrator.py`) dispatches operations page-by-page based on `PageProfile`, and the `spreadsheet_print` extractor runs behind a `qualify_for_spreadsheet_print` gate when the manifest signal fires.

### Preprocessing

**Resolved:** Histogram-based binarisation skip correctly handles digital vs scanned. Dead heuristic code removed.

**Open:** Binarisation hurts recognition on FUNSD forms (CER-s raw 0.189 → pp 0.262, +39%). The histogram correctly identifies these as scanned, but Otsu binarisation degrades character shapes. The preprocessing decision may need a second signal — contrast quality or sample OCR confidence — to decide whether binarisation helps a particular scanned image.

### Layout Detection

**Resolved (model): the document-specific swap landed** (#12, `yolo11n_doc_layout.pt`, 2026-05-25) — the earlier "0 predictions across all DocLayNet fixtures" finding described the general-purpose COCO YOLOv8n and is obsolete. That model detected document layout, including tables: on `dense_text_548` it returned a `table` region at 0.96 confidence.

**Open (model): PP-DocLayout-M replaced YOLO** (#130, `ingest/layout_onnx.py`, registered default `pp-doclayout-m`) for licensing, not accuracy. In an uncontrolled run table-class recall fell from 50% to 25%, and `dense_text_548` gave three table regions where the ground truth has one, plus a `chart` box (mapped to `figure`) almost identical to the `table` box. Tuning is tracked in [plan-permissive-deps.md](plan-permissive-deps.md).

**Resolved (metric): the reported 25% table recall was largely a GT-aggregation artefact** (B0, 2026-07-28). `_aggregate_doclaynet_blocks` groups *consecutive* same-label word spans, so two stray 1-word Table-labelled footnote lines in `dense_text_548` split the real 397-word table run into three GT blocks, each unmatched stray charged as a separate false negative. GT Table blocks are now filtered by a minimum span count (`MIN_TABLE_GT_SPANS = 3`) before matching. Note also `table_0` contains no Table-labelled GT at all (196 Text, 2 Section-header, 1 Page-footer) — despite the name it is not a table fixture and serves as a false-table (no-GT) fixture instead.

### Table-cell reconstruction on OCR'd pages (#17) — **Done**

A layout-detected table region on an OCR'd page (a scanned PDF page or a
standalone image) now becomes a cellified `table` element, behind precision
gates that refuse rather than emit a partial grid — a wrongly-binned grid
produces confidently wrong values downstream, which is worse than no cells.
Still open:

- **Hard shapes refuse.** Stacked spanning headers, hierarchical rows and
  deskewed pages get no cells; DocLayNet `dense_text_548`, the anchor fixture,
  refuses and tracks rather than gates. Repairing these shapes rather than
  refusing them is the deferred scan round.
- **Markdown OCR engines get none.** The Mistral and Ollama engines return page
  markdown with no regions, so they skip the layout pass and reconstruction.

### Money

Self-evidencing narrative amounts are the solved half; the open problems are
structural.

- **Scanned money tables are mostly unreachable.** On `dense_text_548` (four
  `($)` columns, about 35 amounts) the op recovers one — the footnote where OCR
  kept a `$` — because the page's hard-shape table refuses reconstruction,
  leaving bare numbers the narrative path rightly declines. Fed the page's real
  grid, the column path recovers 30 of 30, so the gap is extraction-side, not in
  the op.
- **OCR reads `$` as `s`.** Two further amounts on that page come out as
  `s15.37`. Accepting `s` as a symbol would collide with the `s15`
  legislative-reference blocker, so the fix belongs in OCR or a cleaning op.
- **Flattened transcripts lose the column path.** The ANAO transcript yields 0
  of the 27 `Approved Budget $m` amounts its PDF yields. This is the designed
  refusal; prefer the structural source when a corpus offers both.
- **Labelled coverage is thin.** The labelled money set scores narrative recall
  over tag-labelled transcripts. Table-cell money recall — and so the money
  payoff of #17 — is unmeasured, as is OCR money loss across the PDF set (11 of
  the 29 benchmark PDFs have no text layer, and none of those contains money).
  Next: a bounded labelled sample drawn from the parquet, a few hundred
  candidates across the three loci labelled money / not-money with expected
  value. It would settle the `quantulum3` decision by measurement and serve as
  a regression baseline.

### Reading Order

**Resolved.** CER-s (sorted CER) now separates recognition from reading-order accuracy. 65% of FUNSD sequential CER was ordering mismatch.

### Handwriting

PaddleOCR recognises handwriting poorly (IAM average WER 0.856, CER 0.429 in `docs/accuracy/EXTRACTION.md`, measured on the v4 models; the bundled models are now v5 under `_models/paddleocr-v5`, with the wheel's v4 as fallback). Not worth investing unless handwritten document support becomes a requirement. Add a dedicated HTR model behind `SCANNED_HANDWRITTEN` if needed.

### Pipeline Observability

**Resolved.** `extract_text()` logs one INFO line per document: `strategy selected: doc=<name> type=<type> confidence=<conf> strategy=<class>` for the path-based formats (DOCX, spreadsheet, text, markdown), and `plan-driven extract: doc=<name> type=<type> pages=<n>` for PDFs and images.

### PII Cleaning

PERSON and ADDRESS are now both detected, graph-driven: Kanon-2 enrichment entities (`natural` → PERSON, `address` → ADDRESS) are the candidates. The regex + `all-MiniLM-L6-v2` context backstop is opt-in (`pii.use_regex_backstop`, default off). The Throsby ground-truth recall/precision run below predates ADDRESS support and needs re-running; the GT counts against the 12-entity/6-type fixture are retained for the remaining gaps.

| Entity Type | GT | Supported | Notes |
|-------------|-----|-----------|-------|
| ORGANISATION | 4 | No | Needs NER |
| WEBSITE | 4 | No | URL regex — low effort |
| PHONE | 1 | No | Phone regex — low effort |
| EMAIL | 1 | No | Email regex — low effort |
| ADDRESS | 1 | Yes | Graph candidates (street-type regex in the opt-in backstop); not yet re-measured against this fixture |
| PERSON | 1 | Yes | Graph candidates (regex + context validation in the opt-in backstop); earlier measured run: recall 100%, precision 16.7% (5 FP), predates graph-derived candidates |

**Open issues:**
- ORGANISATION, WEBSITE, PHONE, EMAIL still unsupported. URL/phone/email regex would close 3 of those for minimal effort.
- PERSON precision of 16.7% (1 TP, 5 FP) in the last dedicated run — false positives come from OCR artefacts, state abbreviations, and partial organisation name fragments that escape `_COMMON_WORDS` filtering. Uniform regulatory vocabulary makes cosine similarity poorly discriminative at the 0.35 threshold. Needs re-running now that graph-derived candidates and ADDRESS are wired in.
- NER via Presidio Analyzer + spaCy would handle ORGANISATION and improve PERSON precision, but adds a large dependency. Assess against real-document PII inventory before adding.

### Redaction Handling

Measured on Throsby fixture (7 GT `<REDACTED>` tags across 3 pages).

- **Native cohort recall significantly improved post vector-first detection.** `redact/stage.py:detect_redactions` now tries `page.get_drawings()` for filled near-black rectangles before falling back to the raster CV2 contour detector. On the residual pages (01093 / 01094 / 01349) recall jumped 6→14, 7→13, 3→68 without regressing FOI master (0 regions preserved).
- **Filters** (each surfaced during validation): near-black RGB/CMYK fill; `min_width ≥ 3pt` excludes narrow vertical separators in manifest tables; `min_height ≥ 8pt` excludes glyph-rendering small filled rects on PDFs that draw text as filled-path glyphs (01125-class regression: 14,184 false positives → 144 actual).
- **Open — scanned/raster cohort precision.** Direct-Complaint forms with dark form-field backgrounds (02737-class scanned_mixed docs) still trigger the area-threshold contour detector even with `max_area_ratio=0.05`. Higher precision on this cohort would need a different detection signal (e.g. layout-aware classes that distinguish form fields from redaction bars).
