# Plan — #17 table-cell reconstruction, round 1: clean contemporary documents

Implementation plan for [steering.md](steering.md#table-cell-reconstruction-on-ocrd-pages-17)
item #17, **rewritten 2026-07-28 against `097c075`** to match the corpus
priority: round 1 targets digital-native extraction and flat, relatively
clean, contemporary PDFs. Feature-complete — a detected table region on an
OCR'd page yields a `kind="table"` element with cells, on **both** the
OCR-PDF path and the image path — but tuned and gated for clean tables.
Hard-scan shapes (skewed pages, stacked multi-line spanning headers,
hierarchical row groups) are *refused cleanly and tracked*, not solved, in
this round. Two principles carry over unchanged from the original plan:

- **Precision-first.** A wrongly-binned grid is worse than today's silence.
  `reconstruct_table` returns `None` (never a partial) below the precision
  gates, and the benchmark must be able to say which one we shipped. Refusal
  on a hard shape is a *correct* round-1 outcome, not a failure.
- **No duplicate algorithms.** One shared grid module; existing table-ish
  code is consumed or superseded, never paralleled (see A1).

## Status (updated 2026-07-28)

| Stage | Status |
|---|---|
| A0 — scope + region plumbing | **Landed** (`1f85f7e`) |
| A1 — shared `table_grid` + OCR feeder | **Landed** (`#25`, incl. QA fixes) |
| A2 — skew refusal | **Landed** (with A3) |
| A3 — wire the OCR-PDF path | **Landed** — the OCR-PDF path emits table elements; narrative subtracted, form pairs de-duplicated |
| A4 — route the image path | **Closed — no such path.** Images already route through the orchestrator (`extract_text` sends everything `fitz` opens there), so A3 covered them. The unreachable `ImageExtractor` was deleted rather than wired up; routing is now pinned by test |
| A5 — conventions + lineage | **Verified** — element projection, producer marker and single-markdown chunker view observed end-to-end on the OCR path (A3), which is the path images take too |
| B0 — fix the table metric | **Landed** (`#26`) — steering corrected; GT aggregation drops stray Table runs (`MIN_TABLE_GT_SPANS=3`); EXTRACTION.md refreshes on next full accuracy run |
| B1 — decompose the measurement | **B1.2 landed** (`#26`, `tests/test_table_benchmark.py`) — GT-rect-conditioned reconstruction, no detector in the loop. Stage 1 is the existing per-class F1 (fixed by B0); stage 3 (end-to-end, detector included) is now *possible* on the OCR-PDF path after A3 but is not yet written — it rides with B4 |
| B2 — metric set + gate calibration | Not started — inherits three specifics from A1 plus the `dense_text_548` partial-grid finding from B3 (see Open decisions) |
| B3 — rendered-clean GT harness | **Landed** (`#26`) — 6 rendered fixtures reconstruct with exact structure; `sparse_text_344` off-spec GT removed; the A.6 checker still rides with B2 |
| B4 — report + docs wiring | Not started |
| B5 — regression guard | Not started |

**Sequencing deviation, resolved:** A1 landed *before* B0/B3/B1.2, against
the planned order — the reconstructor existed before any measurement of it
did. The correction held: B0 → B3 → B1.2 all landed **before A3 wired the
reconstructor in**, so A3 went live against a measured baseline rather than
putting unmeasured grids into parquet shards. What remains true is that the
precision gates in `ocr_tables.py` are still provisional structural
constants, not calibrated thresholds — B2 owns that, and it now calibrates
against a live path rather than a dormant one. Remaining order:
`B2 → B4/B5`.

## Where tables come from today (why this scope is enough)

| Path | Mechanism | Status |
|---|---|---|
| Native PDF | `_find_native_tables` — PyMuPDF `find_tables` (lines + text strategies), cross-checked, cellified (`orchestrator.py:104`, `:124`) | **Works.** Covers the digital-native majority of the round-1 corpus |
| Spreadsheet-print PDF | `ingest/spreadsheet_print.py` behind its qualifier | **Works** |
| OCR'd PDF page | `_layout_blocks_and_tables` detects the region → `reconstruct_table` cellifies it | **Works** as of A3 (refuses below the gates, and on deskewed pages per A2) |
| OCR'd image file | Same route as an OCR'd PDF page — `fitz` opens the image as a one-page doc, the orchestrator dispatches `_apply_ocr_page` | **Works** as of A3. Not a separate path; the `ImageExtractor` this table once named was unreachable (see A4) |
| LLM-OCR (mistral-ocr, ollama) | Markdown, no regions | Out of scope (A0) |

Everything downstream of `TableData` already works: `table_to_element`
(`views.py`) cellifies it,
`_accum_to_elements` places it by (y, x), the writer denormalises to
`table_cells.parquet`. **The only missing piece was producing a
`TableData` on the OCR paths.** No schema change, no new element kind.

---

# Track A — the extraction fix

### A0 — scope: region-based engines only — **landed**

When `reading_order_native` is true (mistral-ocr, ollama), the engine
returns markdown and an **empty `regions` list**, and the orchestrator
bypasses `_layout_blocks_and_tables` entirely (`orchestrator.py:179-183`
sets `page_tables = []`). There is nothing to feed `reconstruct_table` — no
quads exist. Track A is therefore scoped to the region-based (paddleocr)
path, on both the OCR-PDF and image routes, and the benchmark must pin
`engine="paddleocr"` or its numbers measure nothing. A markdown-pipe-table →
`TableData` parser for the LLM path is deferred (see Deferred).

Plumbing note: `_ocr_page` already returns
`(text, conf, steps, reading_order_native, regions, pix_dims)` and the
orchestrator unpacks all six at `orchestrator.py:163` — what remains is
passing `regions` + `pix_dims` *into* `_layout_blocks_and_tables`.

**What landed.** `_layout_blocks_and_tables` gained `ocr_regions` +
`ocr_pix_dims` (both optional — callers without them keep today's exact
behaviour), and the orchestrator's region-based branch passes them. Three
things came with the seam:

- `_regions_in_rect(regions, rect)` — the OCR-quad → table-rect
  intersection by centroid containment, the primitive A1's feeder consumes.
- A coordinate-space guard: the OCR render and the layout render are the
  same page at the same dpi, so their pixel spaces coincide. Unless the OCR
  dimensions are supplied *and* match, the regions are dropped with a
  warning rather than binned as incomparable coordinates. (Deskew is *not*
  caught by this — dims survive `warpAffine`; that is A2's page-level
  refusal.)
- A per-table-region debug line carrying the intersecting region count, so
  the size of the gap is traceable per page before the reconstructor exists.

`tables` is still returned empty on every path — A0 produces no cells. The
accuracy suite now pins `engine="paddleocr"` on its extraction calls (the
DocLayNet harness was already pinned by construction, calling
`get_paddle_reader` directly). Tests: `tests/test_table_reconstruction.py`,
which also pins the A3 starting point — the fallback currently collapses the
whole page, table content included, onto one block.

### A1 — one algorithm: `ingest/table_grid.py` — **landed**

**Resolved: shared module, not a duplicate.** The repo already holds three
table-ish algorithms and must not gain a fourth:

1. `spreadsheet_print`'s `_bin_y_bands → _columns_from_data → _column_for_x
   → _bands_to_rows` over `_Span(y_top, y_bottom, x_left, x_right, text)`
   (:58) — this *is* cell reconstruction, and an OCR quad reduces exactly to
   a `_Span`. This is the algorithm to lift.
2. `_table_aware_text` (`strategies_scanned.py:91-212`) — clusters OCR
   regions into rows/columns (`_cluster_x_centroids`, `_find_table_end`,
   `_emit_columns`) and re-emits detected table runs column-major inside the
   page text. Its region→rows preamble also duplicates
   `_spatial_sort_regions` (:59-88 vs :113-138) nearly line-for-line.
3. `grid_projection` — page-level prose gutters and whitespace-aligned text,
   not cells. Weakest prior art; not a feeder and not consumed.

End-state after A1:

- Lift the binning/clustering helpers into `ingest/table_grid.py`;
  `spreadsheet_print` imports them (505 → ~385 lines), the new OCR feeder
  (`ingest/ocr_tables.py`) too. One algorithm, two feeders.
- `_table_aware_text` is **superseded inside layout-detected table rects**
  (the reconstructor owns those); it keeps covering table runs the layout
  model missed, and its row-clustering preamble is replaced by the shared
  `table_grid` helpers so the `_spatial_sort_regions` duplication goes too.
- `OCRRegionResult.bbox` is a four-point quadrilateral `[[x,y] × 4]`
  (`interfaces/protocols.py:26`, built at `paddle_ocr.py:202`) — a 3-line
  bounds conversion turns it into a `_Span`.
- `ROW_BAND_PX=3.0` / `DATA_CLUSTER_GAP_PX=12.0` / `COLUMN_X_TOLERANCE_PX=6.0`
  are PDF points and must become parameters scaled `dpi/72` — 2.8× off at
  200 dpi.

Signature: `reconstruct_table(regions, table_rect, dpi, conf) -> TableData |
None`; returns `None` (never a partial) below the precision gates in B2.

**What landed.** `ingest/table_grid.py` holds the lifted algorithm —
`Span` / `Column`, `bin_y_bands`, `columns_from_data`, `column_for_x`,
`bands_to_rows`, `drop_blank_rows` — with the point-space tolerances as
parameters (`*_PT` defaults; pixel-space callers scale by `dpi/72`), plus
`rows_from_spans` (the adaptive y-centroid row clustering) and
`cluster_x_centroids`. `spreadsheet_print` consumes it (505 → 356 lines,
behaviour unchanged); `_spatial_sort_regions` and `_table_aware_text` now
share the `rows_from_spans` preamble, so the near-line-for-line
region→rows duplication is gone. `ingest/ocr_tables.py` is the second
feeder: `span_from_region` (the quad→bounds reduction), `regions_in_rect`
(moved from `strategies_scanned` so the A3 import direction has no cycle)
and `reconstruct_table(regions, table_rect, dpi, conf, *, pix_dims)` —
`pix_dims` normalises the element position; `conf` is the layout
detector's table-region confidence, capping the mean constituent-region
confidence per A5, and `context["producer"] = "table_grid"` stamps the
lineage. Columns derive from body spans (headers are commonly centred and
would skew clusters).

The header band and the body bands are binned **separately**, because
`bands_to_rows`'s continuation rule folds a band with no leading-column
value into the row above — right for a wrapped body cell, silently wrong
for a first body row whose leading cell is blank (indented or grouped
rows), which was otherwise absorbed into the header and lost.

The precision gates are structural and **provisional until B2 calibrates
them**: `MIN_COLUMNS=3`, `MIN_BODY_ROWS=3`, `MIN_ASSIGNED_RATIO=0.9`,
plus a refusal when no header cell recovers text; each refusal is
debug-logged. Two measured properties B2 must calibrate around rather
than assume away:

- `MIN_BODY_ROWS` is 3, not 2, because `columns_from_data` independently
  drops any x-cluster holding fewer than 3 spans — every column of a
  2-body-row table is filtered out, so that shape can never reconstruct
  and a lower constant would be unreachable rather than permissive.
- `MIN_ASSIGNED_RATIO` is **asymmetric**: `column_for_x` assigns anything
  at or right of the first column, so the ratio gates left-edge overflow
  only. Content right of the last column either forms its own column or
  joins the last one — the right-edge guardrail has to come from B2's
  false-table fixtures, not from this ratio.

Known round-1 limitation, deliberately not fixed here: a two-line header's
second line becomes a spurious first body row (only `bands[0]` is the
header). Merging it means solving multi-line headers — deferred — and
refusing it means a proximity threshold; `spreadsheet_print`'s
`HEADER_MERGE_PX` scales to ~33 px at 200 dpi, which would falsely refuse
a dense clean table whose row pitch is ~28 px. B2's cell-F1 metric should
measure the cost against real fixtures before a threshold is picked.

Nothing is wired into the layout pass yet — `tables` is still returned
empty on every extraction path until A3.

### A2 — skew: refuse, don't solve (round-1 cut) — **landed**

The trap, verified: `_ocr_page` (`strategies_scanned.py:215`) calls
`preprocess_for_ocr` for region engines, which **deskews via `warpAffine`**
(`paddle_ocr.py:489-493`) before OCR — so region coords are in deskewed
space, while `_layout_blocks_and_tables` (:424-426) renders its own raw
pixmap for YOLO. Intersecting the table rect with OCR regions is then wrong
by the skew angle and fails silently at the edges.

Round-1 handling: **skip reconstruction on deskewed pages.** Deskew fires
only when |angle| > 0.5° with confidence > 0.3, and flags itself with
`"deskew" ∈ steps` — flat contemporary documents almost never trip it. When
it fires, keep today's behaviour exactly (no table element, full-text
narrative block). That is a page-level refusal consistent with
precision-first, costs three lines, and defers the coordinate plumbing
(exporting the rotation matrix and mapping the YOLO rect into deskewed
space) to the round that targets real scans. Do **not** "share one image"
between OCR and YOLO as a shortcut — sharing the preprocessed image feeds
the layout model binarised grayscale, a distribution shift that silently
moves B1's detection metric.

(The original plan text noted that `ImageExtractor` doesn't preprocess, so
"the image path has no mismatch and no refusal condition". A4 established
there is no separate image path: images go through `_apply_ocr_page`,
which *does* preprocess, so a skewed image page refuses exactly as a
skewed PDF page does.)

**What landed.** `_layout_blocks_and_tables` gained `page_deskewed: bool`,
which drops the cell source alongside the A0 dimension guard — the two
answers to the same question ("are these coordinates comparable to my
render?"), kept together. The orchestrator supplies it as
`"deskew" in steps`, reading the flag `preprocess_for_ocr` already sets
rather than inferring skew a second time. The refusal is debug-logged, and
the page keeps today's behaviour exactly: no table element, full-text
narrative block.

### A3 — wire the OCR-PDF path — **landed**

`_layout_blocks_and_tables` already takes `ocr_regions` + `ocr_pix_dims`
(landed in A0). What remains is populating the `tables` list declared at
:421 and returned empty at :467/:477.

**The double-count to prevent is the fallback narrative block, not the
`[TABLE]` placeholder.** Layout-derived non-table blocks always carry
`text=""` (`strategies_scanned.py:445`), so the "no block has text" fallback
at :452 fires on essentially every layout-successful page and returns **one
block containing the whole page's OCR text — table content included** —
already discarding the placeholder (`return [block], tables` at :467). So:
when a table element is emitted, **rebuild the page narrative from the
regions outside the table rect** (`_spatial_sort_regions` over the
complement) and emit that as the paragraph block; otherwise the chunker sees
the table twice — as narrative paragraph and as table markdown. On refusal
(A2 skew, B2 gates), keep today's full-text fallback exactly.

Two adjacent overlaps, both cheap to close in the same wiring:

- The orchestrator runs `_extract_form_pairs_from_regions` over *all* page
  regions (:173-175); exclude regions consumed by a successful table so
  colon-bearing cells don't land in both a form element and the table.
- Leave the `SCANNED_MACHINEWRITTEN` grid fallback (:205-214) alone;
  `not page_tables` already guards it.

**What landed.** `_layout_blocks_and_tables` calls `reconstruct_table` per
detected table region and now returns a third value —
`consumed_regions`, the OCR regions a successful table absorbed. On success
the `[TABLE]` placeholder block is dropped (the table element replaces it);
on refusal it stays, so the fallback's "no non-table block has text" test is
unchanged. The narrative rule is as planned: when anything reconstructed,
the fallback block's text is rebuilt from the complement regions and the
block is typed `paragraph` outright (the dominant region's kind described
the table, which is no longer part of that block). A page that is only a
table returns no narrative block rather than the empty layout placeholders.
The debug line now reports the reconstruction *outcome* per region rather
than the pre-reconstructor region count, and is no longer `isEnabledFor`-gated
because the intersection is computed either way.

Three deliberate calls beyond the plan text:

- **The complement is re-emitted with `_table_aware_text`, not
  `_spatial_sort_regions`.** A3's prose named the latter, but A1's end-state
  is explicit that `_table_aware_text` is superseded only *inside*
  layout-detected table rects and "keeps covering table runs the layout model
  missed". Using the row-major sorter on the complement would drop that
  coverage for no gain; the reconstructor already owns the rects it was given.
- **`PageResult.text` stays the verbatim full-page OCR text.** The
  subtraction is an element-stream concern. Page text feeds `_text_coverage`
  and the accuracy suite's CER, which compare against a transcript of the
  whole page — subtracting the table there would move those metrics without
  any downstream consumer benefiting, since chunking reads `elements`.
- **The exception handler resets `tables`/`consumed`.** The pass's
  catch-all previously wrapped a function that could only return empty
  tables. Now a throw partway through the region loop could surface tables
  whose text had not yet been subtracted from the narrative — precisely the
  double-count A3 exists to prevent — so the handler drops them and the page
  falls back to today's behaviour.

The form-pair exclusion landed as planned; the orchestrator's layout call now
runs *before* the form call so it has the consumed set to filter with (forms
are collected separately and appended per page, so element order is
unaffected). The `SCANNED_MACHINEWRITTEN` grid fallback was left alone.

### A4 — route the image path — **closed: the path did not exist**

**The premise was wrong, and QA caught it after the first implementation.**
A4 was written against `ImageExtractor.extract`, which never called
`_layout_blocks_and_tables` and emitted one page-wide paragraph per page.
That description of the *class* was accurate. What it got wrong is that
the class is reachable at all.

`extract_text` gates the legacy path-based dispatch on
`doc_type in (SPREADSHEET, DOCX, TEXT)` (`extract.py`). **`IMAGE` is not in
that tuple** — it falls through to `fitz.open()` +
`extract_pdf_with_plan`, because PyMuPDF opens a standalone image as a
one-page document. So `get_extractor`'s `DocumentType.IMAGE` case was
unreachable from the only call site that existed, and `ImageExtractor` was
dead on every production *and* measurement path (the accuracy suites call
`extract_text` or `get_paddle_reader` directly).

Measured, not inferred — a real `.png` through the real `extract_text()`
with OCR and layout stubbed:

```
ImageExtractor reached: False
element kinds: ['paragraph', 'table']
  paragraph text: 'Narrative line one\nNarrative line two'
  table meta: {'context_producer': 'table_grid'} extractor: ocr_paddle
  header row: ['H1', 'H2', 'H3', 'H4']
```

Table element, producer marker, narrative subtracted. **A3 closed the
image gap when it wired the orchestrator** — images had been going through
`_apply_ocr_page` all along.

**What landed instead.** The first A4 implementation (wiring the layout
pass into `ImageExtractor`, ~45 lines plus 9 tests) was reverted: it
changed no production behaviour and made a dead class a *more* faithful
parallel of the orchestrator, against this plan's own anti-duplication
principle and CLAUDE.md's "delete the Womblex code rather than carrying a
parallel implementation". Deleted instead:

- `ImageExtractor` itself, and the now-orphaned imports it held.
- `get_extractor`'s unreachable `DocumentType.IMAGE` case. Its `dpi` /
  `lang` / `engine` / `engine_options` parameters went with it — they
  existed only to construct `ImageExtractor`, and a function that silently
  ignores an `engine=` argument is a trap. The signature is now
  `get_extractor(profile)` and the return type narrows to
  `PathExtractionStrategy`, dropping a `type: ignore` at the call site.
- The `strategies.py` re-export.

`table_to_element` stays in `ingest/views.py` (moved out of the
orchestrator during the reverted attempt) — with the reverse projections
it is the whole view↔element mapping in one file. It has one caller.

Tests: `TestImageDocumentsRouteThroughTheOrchestrator` replaces the
deleted cohort — it drives a real `.png` through `extract_text` and
asserts the table element, its producer marker, the subtracted narrative,
and that `get_extractor` now refuses `IMAGE`. That pins the routing, so a
future change reintroducing an image bypass fails here rather than
silently losing table reconstruction on every image input.

**Stale claims this corrected**, all of which predated the attempt: the
"Where tables come from today" row calling the image path "the remaining
gap"; steering's "every image input (the whole DocLayNet/FUNSD fixture
shape) is still unchanged — this is now the largest remaining piece of
#17"; `money.md`'s "through `ImageExtractor`, which still never calls the
layout pass"; `get_extractor`'s docstring listing IMAGE among the types it
handles; and the generated `EXTRACTION.md` strategy-matrix row
`| IMAGE | ImageExtractor (legacy) | Direct PaddleOCR |`.

### A5 — post-processing conventions and lineage — **verified on both OCR paths**

The two mandated provenance fields below landed with A1
(`reconstruct_table` sets confidence from the constituent regions and
stamps `context["producer"] = "table_grid"`). A3 is the first path that
actually puts one through the pipeline, so the convention claims are now
*observed* rather than structural: `TestReconstructedTableDownstream`
drives an OCR page through the orchestrator and asserts the element is a
cellified `kind="table"` with `header_rows=[0]`, that
`meta["context_producer"] == "table_grid"` survives the existing
`context_* → meta` copy with no schema change, that the narrative
paragraph beside it holds no table text, and that
`collect_tables_from_elements` yields exactly one markdown table.
`TestImageDocumentsRouteThroughTheOrchestrator` observes the same
conventions from the `extract_text` entry point on an image input (A4).

Because Track A produces a `TableData` and reuses `table_to_element`, every
downstream composed stage consumes reconstructed tables through the existing
conventions — verified, not assumed:

- **Writer** (`store/output.py:275`): any `kind="table"` element with cells
  denormalises to `table_cells.parquet` keyed `(source_hash,
  parent_elem_order)`; the element row carries `source_hash`,
  `collection_id`, `elem_order`, `extractor`, `confidence`, `page`, `bbox`,
  `header_rows`, `meta`; the batch manifest's `table_cells_count` rises
  naturally. No writer or schema change.
- **Chunker**: `collect_tables_from_elements` (`chunker.py:491`) picks up
  any `kind='table'` element → one markdown table; chunks/embeddings join on
  `(source_hash, chunk_index, content_type)` unchanged.
- **Money stage**: the `table_cell` locus anchors `(source_hash,
  parent_elem_order, row, col)` on the sidecar (`money_stage.py:243`) —
  reconstructed tables become column-classifiable with zero money-stage
  changes; that is the measured payoff.
- **Cleaning overlays** (normalise / spellfix) rewrite `TEXT_KINDS` only —
  table cell values stay extraction-verbatim, exactly as native and
  spreadsheet-print tables do today.

Two provenance fields the reconstructor must populate deliberately rather
than inherit as defaults:

- **`confidence`** — set `TableData.confidence` from the constituent region
  confidences (and the gate score), not a constant; it is the element's
  lineage signal for downstream quality filtering.
- **Producer marker** — on OCR pages `_accum_to_elements` stamps every
  element `extractor="ocr_paddle"`, and the `SCANNED_MACHINEWRITTEN` grid
  fallback (:214) lands PyMuPDF tables through the *same* accumulator, so
  reconstructed and fallback tables would be indistinguishable in the
  parquet. Set `TableData.context["producer"] = "table_grid"` — the existing
  `context_* → meta` copy in `_table_to_element` (`orchestrator.py:267`)
  carries it to `meta["context_producer"]` with no schema change.

---

# Track B — benchmark + accuracy extension

Track B is half the work, and B0 lands first because the existing table
metric is currently misleading. The round-1 reprioritisation changes *which
fixtures gate*: the rendered clean-document GT (B3) is the primary target —
it is exactly the flat contemporary shape round 1 ships for — and the hard
scanned fixture tracks progress without gating.

### What exists to build on

| Piece | Where | Fit |
|---|---|---|
| `_results` accumulator + per-fixture dicts | `test_fixture_accuracy.py:169` | add a `tables` key |
| `generate_extraction_report(results)` | `accuracy_reports.py:252` | add a section emitter |
| `write_report` session fixture | `test_fixture_accuracy.py:725` | already regenerates EXTRACTION.md |
| `structural_fidelity` / `data_integrity` / `key_column_preservation` / `schema_conformance` | `tabular_metrics.py:31/80/160/195` | DataFrame-shaped; wired only to spreadsheet ingest today (`evaluation.md` §2) |
| DocLayNet layout harness incl. per-class table F1 | `test_fixture_accuracy.py:448` | B1's decomposition hangs off this |

### B0 — fix the table metric before measuring against it — **landed**

**(a) steering's Layout Detection paragraph is stale.** It says "0
predictions across all DocLayNet fixtures"; EXTRACTION.md's per-class table
reports `table | TP 1 | FP 0 | FN 3 | P 100% | R 25% | F1 40%`. The layout
model *does* detect tables. (steering.md:110-114 acknowledges the staleness;
the correction itself is still unapplied.)

**(b) the 25% recall is largely an annotation artefact.** Aggregating GT
spans per `_aggregate_doclaynet_blocks` (:218) across the vendored fixtures
gives exactly 4 GT `Table` blocks:

```
dense_text_548    table blocks=3   word runs = [397, 1, 1]
sparse_text_344   table blocks=1   word runs = [8]
diverse_layout_49 / formula_29 / table_0   table blocks=0
```

The 397-word run is the real table; the two 1-word runs are stray
Table-labelled words splitting it, each charged as a separate FN. Fix the
aggregation for the table class (or report table detection at region
granularity with a minimum-span filter) before anyone reads a reconstruction
number against it. Also: **`table_0` contains no Table-labelled GT at all**
(196 Text, 2 Section-header, 1 Page-footer) — despite the name it is not a
table fixture and must not be used as one.

**What landed.** Both corrections applied to steering.md (Layout Detection
rewritten; the `grid_projection` bullet corrected while there). The
aggregation fix took the min-span form: `_aggregate_doclaynet_blocks` now
counts constituent spans per block and drops Table blocks below
`MIN_TABLE_GT_SPANS = 3`. Drop, not merge, was verified as the right remedy:
the two strays are footnote lines *below* the table (y 725–734 against a
table ending at y 724) mislabelled Table — merging would have stretched the
GT table rect over footnote text, which would poison the B1.2 GT-rect
derivation. `dense_text_548` now contributes exactly 1 GT Table block;
`sparse_text_344`'s 8-span block is untouched. Verified by a layout-only
probe replicating the suite's per-class computation (all non-table classes
byte-identical to EXTRACTION.md): table TP1/FP0/FN3 → TP1/FP0/FN1
(R 25% → 50%, F1 40% → 66.7%); the remaining FN is `sparse_text_344`'s
genuinely undetected block. EXTRACTION.md itself refreshes on the next full
accuracy-suite run (hour-plus; not run for this change).

### B1 — decompose the measurement into two stages — **B1.2 landed**

Report reconstruction **conditioned on a correct region**, not blended with
detector recall:

1. **Detection** — does layout return a `table` region covering the GT
   table? (existing per-class F1, fixed per B0.)
2. **Reconstruction** — feed the *GT* table rect straight to
   `reconstruct_table` and measure the grid. This is the number that tracks
   A1, tunable without the detector in the loop.
3. **End-to-end** — real pipeline, detector included; the product of 1 × 2.

### B2 — the metric set

`tabular_metrics` assumes aligned DataFrames with matching column names,
which a reconstructed OCR grid will not have. Add a thin alignment
projection (`cells → DataFrame`, header row → column names), then:

- **Structure**: detected rows/cols vs GT rows/cols (exact, plus off-by-one
  rate).
- **Cell content F1** on `(row, col, text)` triples after alignment —
  catches column-shift errors a cell-count metric misses.
- `data_integrity` for exact-match cells; `key_column_preservation` where a
  key column exists (the rendered-CSV route in B3 gives one). The scorer
  declares its normalisation (NFKC + dash folding at minimum) — GT stays
  verbatim.
- **False-table rate (the precision guardrail)**: run the reconstructor over
  fixtures with *no* GT table (`diverse_layout_49`, `formula_29`, `table_0`,
  plus FUNSD forms — dense label/value pairs are the likeliest false
  positive). Any table emitted is an FP. This is what makes
  "precision over coverage" falsifiable.
- **Task metric**: money amounts recovered — the measured payoff in
  steering (1 of ~35 today on the dense fixture; 30 with cells). Reported,
  not gated, in round 1.

Not proposing TEDS — it needs a tree edit-distance dependency. Note it as
the upgrade path if the alignment metric proves too coarse.

### B3 — ground truth: rendered-clean is primary, dense scan tracks

- **Primary (gates): rendered clean-document GT.**
  `womblex-collection/_spreadsheets/Approved-providers-au-export_20260204.csv`
  (10,859 × 10, natural key column) and `mso-statistics-sept-qtr-2025.xlsx`
  already ship. Render → PDF → rasterise → OCR → reconstruct → compare
  against the **source DataFrame**. Arbitrary volume, zero annotation cost,
  exactly the shape `tabular_metrics` was built for, and — under the round-1
  scope — exactly the flat contemporary shape we are shipping for. Render a
  small deterministic subset (say 3 pages per source at fixed dpi), not the
  full 10k rows, so the suite stays fast.
- **Tracking (reported, no gate): `dense_text_548`** — GT landed
  (`7f756cc`): `dense_text_548_table.csv` + `.meta.json` pass the A.6 checks
  (39 × 11 rectangular, unique headers, no trailing whitespace,
  `n_header_rows: 3`). It is a skewless but otherwise hard scan (7-line
  stacked spanning header, hierarchical rows) — precisely the far edge round
  1 is not solving. Expected round-1 outcome is refusal or a partial score;
  either is fine, both are visible in the report. The A.6 checker still
  needs writing (the checks above were run by hand once).
- **`sparse_text_344`: declare non-GT and remove its CSV/meta** — done
  (`#26`). It landed off-spec (no `_table` suffix; a different meta
  schema than Appendix A) and its 8-word, 4-row single-column block
  exercises almost nothing. One GT convention, one loader — carrying a
  second format for a marginal fixture is exactly the duplication this
  round avoids.
- Later: more Table-labelled pages from the full DocLayNet clone (per
  THIRD_PARTY_DATA.md) when the scan round arrives.

**What landed** (`tests/test_table_benchmark.py`, marked `benchmark`, ~25 s
for the whole cohort). The rendered-GT route: pandas reads the source →
deterministic bounded-width column subsets (5 of the CSV's 10; 5 of the
fuel sheets' 8 by position) → fitz draws a left-aligned grid on a
content-sized page (helv 9 pt, 16 pt row pitch, 24 pt column gaps) → 200 dpi
rasterise → paddleocr → `reconstruct_table` with the rect known by
construction (B1.2: `conf=1.0`, no detector in the loop) → positional
scoring against the drawn strings, normalised NFKC + dash-fold + whitespace
collapse (the scorer declares its normalisation; GT stays verbatim). Six
fixtures: 3 pages × 30 rows of the Approved-providers CSV, plus one page
per fuel sheet (Diesel / Gasoline / Kerosene, 9 rows). B2's alignment
projection can later swap this positional scorer for the `tabular_metrics`
set without touching the render route.

Measured (round-1 baseline, 2026-07-28):

| Fixture | GT | Result | Headers | Cells |
|---|---|---|---|---|
| approved_providers_p1–p3 | 30×5 | 30×5 exact | 5/5 | 96.0–98.7% |
| mso_diesel / gasoline / kerosene | 9×5 | 9×5 exact | 5/5 | 84.4–95.6% |
| dense_text_548 (tracking, GT rect) | 39×11 | **partial 12×12** | — | — |

Every rendered-clean cell mismatch is glyph-level OCR recognition, not
binning: a lost space (`PO Box1213`), `0`→`o` on single-char cells,
`6`↔`9` at 9 pt. The grid itself is exact on all six. Two structural
asserts already gate (reconstruction happened; column count exact) — B5
formalises the rest.

**The tracking fixture did not refuse** — the provisional gates passed a
12×12 partial grid against the 39×11 GT (over-segmented columns,
under-merged rows). Round 1 said "refusal or a partial score; either is
fine, both are visible" — but this is now a *measured* input to B2:
precision-first wants the stacked-header hierarchical shape refused (or
scored low), so gate calibration must add a signal this shape trips
(e.g. assigned-ratio symmetry, header-band coherence) rather than assume
`MIN_*` constants catch it.

### B4 — report + docs wiring

- New `_results["tables"]` entries → a `## Table Reconstruction` section in
  `generate_extraction_report` (extend EXTRACTION.md, directly under the
  per-class layout section it decomposes). Columns: fixture, GT rows×cols,
  detected rows×cols, cell F1, data integrity, gate outcome, money recall.
  Rendered-clean fixtures and the tracking scan fixture appear in the same
  table with their gate/track status explicit.
- Separate **false-table** table so the guardrail is visible, not buried.
- `docs/evaluation.md`: §2 is "Tabular Extraction Accuracy" scoped to
  CSV/XLSX → parquet; add §2b for document-table reconstruction so the two
  capabilities aren't conflated.
- **Knock-on: CHUNKING.md.** Landing tables on OCR pages changes chunk
  counts and composition — regenerate and sanity-check in the same run.
- All regeneration via the test run (`write_report`, :725). Never hand-edit.

### B5 — regression guard

Assert-with-threshold, resolved for round 1 as: **gates fail the build on
the rendered-clean fixtures** (structure exact-match, cell F1 floor,
**false-table count == 0**) and **`dense_text_548` is reported without a
gate** until the scan round sets one. Today's benchmark tests end in
`assert True` (:530) and only report — the table gates should be the first
that actually fail. The benchmark pins `engine="paddleocr"` (per A0) so a
config default change to an LLM engine can't turn the gates into no-ops.

---

## Sequencing

Planned: `B0 → B3 rendered-GT harness + B1.2 → A1 → A3 → A4 → B2 breadth →
B4/B5`. Actual: A1 landed first (deviation recorded in Status above), and
A4 turned out to be a no-op — the path it targeted was unreachable, so it
closed by deleting dead code rather than by adding any. Remaining:
**`B2 breadth → B4/B5`** — Track A is complete for the region-based
engines it was scoped to (A0).

B0 went first because it was a live doc/metric defect. The rendered-GT
harness and B1.2 landed before A3 as planned — the measurement existed
before the reconstructor was wired in, and it surfaced one calibration
input immediately (the `dense_text_548` partial grid, above). A2 rode
along with A3's wiring as intended.

One consequence of the ordering to keep in view: A3 is live but B2 has not
calibrated the gates, so the constants in `ocr_tables.py` remain
provisional *in production*, not just in the abstract — and A4 established
that this covers image inputs too, not just PDFs. The rendered-clean
cohort says they are safe on
the target shape (6/6 exact structure); the `dense_text_548` partial says
they do not yet refuse the hard shape. Until B2 lands, a hard-shape table
on an OCR'd page can produce a low-quality grid rather than silence —
visible in the parquet via `meta["context_producer"] = "table_grid"` and
the element confidence.

## Deferred to a later round (recorded so they aren't relitigated)

- **Skewed scans**: export the rotation matrix from `preprocess_for_ocr`
  and map the YOLO table rect into deskewed space (A2 refuses instead).
- **Stacked spanning headers / hierarchical rows as a gated target**:
  `dense_text_548` graduates from tracking to gated when a scan round picks
  it up.
- **LLM-OCR path tables**: markdown pipe-table → `TableData` parser as a
  third feeder (mistral-ocr / ollama emit no regions).
- **TEDS** if the alignment metric proves too coarse.
- **More real-scan GT** from the full DocLayNet clone.

## Open decisions

1. Precision-gate thresholds — **now the highest-priority open item**, because
   A3 made the gates load-bearing in production rather than dormant. Set them
   from the rendered-clean fixtures + the false-table set (not from
   `dense_text_548`, which no longer gates).
   Three specifics inherited from A1: a right-edge overflow guardrail
   (`MIN_ASSIGNED_RATIO` only gates the left edge), whether the
   column-population floor of 3 should be a parameter rather than couple
   `MIN_BODY_ROWS` to it, and whether a two-line header should merge or
   refuse. Plus one measured by B3: the current gates pass a 12×12 partial
   grid on `dense_text_548` (39×11 GT) — the stacked-header hierarchical
   shape needs a refusal signal, not just tighter `MIN_*` constants.
Resolved this revision: shared `table_grid.py` (A1 — anti-duplication);
`_table_aware_text` end-state (A1); skew handling (A2 — refuse);
gate-vs-warn (B5 — gate clean, track dense); GT sign-off (landed in
`7f756cc`, conventions as recommended); `sparse_text_344` (B3 — non-GT);
B0 remedy (landed: min-span filter `MIN_TABLE_GT_SPANS=3` in
`_aggregate_doclaynet_blocks`, dropping — not merging — stray Table runs).

---

# Appendix A — GT authoring spec for `dense_text_548` (landed `7f756cc`)

Retained as the normative spec for any future table GT (the scan round will
add more). The landed `dense_text_548_table.csv` follows it; the A.6
checker remains to be built in B2.

The page is a *Grants of Plan-Based Awards* proxy table: **11 columns**, a
7-line stacked header, and hierarchical rows (participant name, then award
sub-rows). It is the hardest realistic shape we have — which is why, under
the round-1 scope, it tracks rather than gates.

### A.1 Files to produce

| File | Purpose |
|---|---|
| `fixtures/fixtures/doclaynet/<fixture>_table.csv` | the GT grid |
| `fixtures/fixtures/doclaynet/<fixture>_table.meta.json` | `{"n_header_rows": N, "key_column": null, "notes": "..."}` |

Sits beside the existing `_transcript.txt` sidecar, following that
convention. UTF-8, `\n` line endings, RFC4180 quoting. If a page has two
tables, use `_table1.csv` / `_table2.csv`.

**No bounding box needed** — the GT table rect derives automatically from
the `Table`-labelled bboxes in the fixture json. Don't hand-measure it.

### A.2 The governing rule: transcribe the page, not the intended table

GT measures extraction fidelity, so it records **what is printed**, verbatim:

- Keep thousands separators exactly as printed — `32,031`, not `32031`.
- Keep the em-dash null marker `—` (U+2014) as its own cell value; empty and
  `—` are different observations.
- Keep unit annotations where printed — `($)`, `(#)`, `($/share)`.
- Do not fix the source document's own oddities (repeating column letters,
  interleaved footnote markers). Transcribe; do not renumber.
- Do not correct spelling, spacing or alignment to what the table "should" be.

### A.3 Header handling

Emit **one flattened header row** as CSV row 1 with **unique** column names
(pandas silently mangles duplicates and `structural_fidelity` compares
column *sets*). Qualify duplicated names by their spanning group
(`Non-Equity Threshold`, `Equity Threshold`). Record the printed stacked
rows (units row, column-letter row) as ordinary data rows underneath, and
set `n_header_rows` in the meta json. The reconstructor is not expected to
flatten a spanning header; the scorer uses `n_header_rows` to compare like
with like.

### A.4 Row handling

Participant names occupy their own rows with the data columns empty;
sub-rows follow as their own rows. Do **not** forward-fill the name — that
describes a normalised relational view a geometric reconstructor cannot
produce. Names split across printed lines join with a single space. Every
CSV row has the full column count; empty cells are empty strings.

### A.5 Scoring implication

`data_integrity` is exact string match, so the comparison side declares a
normalisation — NFKC plus dash folding at minimum. **GT stays verbatim; the
scorer declares its normalisation.**

### A.6 Acceptance checks before a GT is used

A small checker (part of B2) asserts: rectangular, column names unique and
non-empty, UTF-8 with no BOM, no trailing whitespace in cells,
`n_header_rows` consistent with the file, and the GT cell count within a
sane band of the fixture json's Table-labelled word count. A GT that fails
these is a bug in the GT, not in extraction.
