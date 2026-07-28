# Plan — #17 table-cell reconstruction, with benchmark + accuracy extension

Implementation plan for [steering.md](steering.md#table-cell-reconstruction-on-ocrd-pages-17)
item #17. Written before implementation and not yet executed. **Revised
2026-07-28 against `097c075`** — the original `00b4a39` baseline predated two
commits that change the plan's ground: the Mistral-OCR engine swap (`097c075`,
which reshaped `_ocr_page` / the orchestrator's OCR branch and added an
LLM-OCR path with no regions at all — see A0), and the GT fixture commit
(`7f756cc`, which partially executed Appendix A — see B3). Line references
are updated to `097c075`; open decisions at the end are unresolved except
where marked.

Goal: a detected table region on an OCR'd page yields a `kind="table"` element
with cells, on **both** the OCR-PDF path and the image path — and the benchmark
can *prove* it, per fixture, per stage, with a regression guard. Precision-first:
a wrongly-binned grid is worse than today's silence, and the metrics must be able
to say which one we shipped.

Two tracks. **Track B is not "step 5" — it is half the work, and B0 lands first
because the existing table metric is currently misleading.**

---

# Track A — the extraction fix

Everything downstream of `TableData` already works: `_table_to_element`
(`orchestrator.py:250`) cellifies it, `_accum_to_elements` (:295) places it by
(y, x), the writer denormalises to `table_cells.parquet`. **The only missing
piece is producing a `TableData`.** No schema change, no new element kind.

### A0 — scope: region-based engines only (new since the engine swap)

The Mistral-OCR swap split the OCR branch in two, and reconstruction can only
run on one side. When `reading_order_native` is true (mistral-ocr, ollama),
the engine returns markdown and an **empty `regions` list**, and the
orchestrator bypasses `_layout_blocks_and_tables` entirely
(`orchestrator.py:179-183` sets `page_tables = []`). There is nothing to feed
`reconstruct_table` — no quads exist. So Track A is architecturally scoped to
the region-based (paddleocr) path, on both the OCR-PDF and image routes, and
the benchmark must pin `engine="paddleocr"` or its numbers measure nothing.

Pixtral markdown *does* contain pipe tables for table regions; a
markdown-table → `TableData` parser would be a cheap third feeder that gives
the LLM path table elements too. Not in scope for #17 — recorded as open
decision 6 so it isn't lost.

Also now true (simplifies A3): `_ocr_page` already returns
`(text, conf, steps, reading_order_native, regions, pix_dims)` and the
orchestrator already unpacks all six at `orchestrator.py:163` — the plumbing
the original plan assigned to A3 half-exists; what remains is passing
`regions` + `pix_dims` *into* `_layout_blocks_and_tables`.

### A1 — `ingest/table_grid.py` (shared) + `ingest/ocr_tables.py` (OCR feeder)

Three corrections to the write-up's framing:

- `OCRRegionResult.bbox` is a **four-point quadrilateral** `[[x,y] × 4]`
  (`interfaces/protocols.py:26`, built at `paddle_ocr.py:202`), not
  `(x0,y0,x1,y1)`. Not drop-in into `grid_projection`; needs a 3-line bounds
  conversion.
- **`spreadsheet_print` is the whole algorithm, not just "the row side".** Its
  `_Span(y_top, y_bottom, x_left, x_right, text)` (:58) is exactly what an OCR
  quad reduces to, and `_bin_y_bands → _columns_from_data → _column_for_x →
  _bands_to_rows` *is* cell reconstruction. `grid_projection` finds
  page-level prose gutters and renders whitespace-aligned text, not cells —
  weaker prior art than the doc implies.
- **There is already a third table algorithm in the target file, and the plan
  must state its end-state.** `_table_aware_text`
  (`strategies_scanned.py:91-212`) clusters OCR regions into rows and columns
  (`_cluster_x_centroids`, `_find_table_end`, `_emit_columns`) and re-emits
  detected table runs **column-major inside the page text**. It is the text
  the fallback narrative block carries today (see A3). Intended end-state:
  the reconstructor supersedes it *inside* layout-detected table rects;
  `_table_aware_text` keeps covering table runs the layout model missed.
  Without this decision the ingest package carries three table-ish
  algorithms (`spreadsheet_print` binning, `_table_aware_text` clustering,
  new `table_grid`) with no stated relationship.

Lift the shared helpers into `ingest/table_grid.py`; `spreadsheet_print` imports
them (505 → ~385 lines), `ocr_tables` too. One algorithm, two feeders.
`ROW_BAND_PX=3.0` / `DATA_CLUSTER_GAP_PX=12.0` / `COLUMN_X_TOLERANCE_PX=6.0` are
PDF points and must become parameters scaled `dpi/72` — 2.8× off at 200 dpi.

`reconstruct_table(regions, table_rect, dpi, conf) -> TableData | None`; returns
`None` (never a partial) below the precision gates in B2.

### A2 — the coordinate-space trap

`_ocr_page` (`strategies_scanned.py:215`) calls `preprocess_for_ocr` for
region-based engines (LLM engines now skip it via `is_llm_engine`, :252 —
irrelevant here since they yield no regions), which **deskews via
`warpAffine`** (`paddle_ocr.py:489-493`) before OCR — so region coords are in
deskewed space, while `_layout_blocks_and_tables` (:424-426) renders its
*own* raw pixmap for YOLO. Same dimensions, content rotated; intersecting the
table rect with OCR regions is wrong by the skew angle and fails silently at
the edges.

"Share one image out of `_ocr_page`" is the wrong fix as originally worded:
the raw render is the one with the wrong coords, and sharing the
*preprocessed* image with YOLO feeds the layout model binarised grayscale —
a distribution shift that silently moves B1's detection metric. Preferred
fix: have `preprocess_for_ocr` return the rotation matrix (deskew fires only
when |angle| > 0.5° and confidence > 0.3, so it is usually identity and
`"deskew" ∈ steps` flags when it isn't) and map the YOLO table rect into
deskewed space — an affine on 4 points, no change to either model's input.
The duplicate pixmap render can still be shared (both sides render the same
raw pixmap at the same dpi); only the *coordinate mapping* needs the matrix.
`ImageExtractor` doesn't preprocess (:518-520), so it has no mismatch —
another reason the fix belongs in `_ocr_page`, not the reconstructor.

### A3 — wire the OCR-PDF path

`_layout_blocks_and_tables` gains `regions` + `pix_dims` (the orchestrator
already holds both at :163, per A0). Populate the `tables` list declared at
:421 and returned empty at :467/:477.

**The real double-count is the fallback narrative block, not the `[TABLE]`
placeholder.** Layout-derived non-table blocks always carry `text=""`
(`strategies_scanned.py:445` — "layout region text not yet segmented"), so
the "no block has text" fallback at :452 fires on essentially every
layout-successful page and returns **one block containing the whole page's
OCR text — table content included, column-major via `_table_aware_text`** —
already discarding the `[TABLE]` placeholder in the process (`return [block],
tables` at :467). Dropping the placeholder on success is therefore mostly a
no-op; the actual requirement is: **when a table element is emitted, the
table region's text must be excluded from the narrative block**, or the
chunker sees the content twice — once as a narrative paragraph, once as
table markdown. Fix shape: on reconstruction success, rebuild the page
narrative from the regions *outside* the table rect (`_spatial_sort_regions`
over the complement) and emit that as the paragraph block; on failure, keep
today's full-text fallback (and its placeholder behaviour) exactly.
`_BLOCK_TYPE_TO_KIND` still maps `table → paragraph` (:231), so any surviving
placeholder would double-count — keep the drop, just don't mistake it for
the fix. Leave the `SCANNED_MACHINEWRITTEN` grid fallback (:205-214) alone;
`not page_tables` already guards it.

A second, smaller overlap: the orchestrator runs
`_extract_form_pairs_from_regions` over *all* page regions unconditionally
(:173-175). Colon-bearing cells in a reconstructed table can land in both a
form element and the table element. Cheap fix in the same wiring: exclude
regions consumed by a successful table from the form-pair candidates.

### A4 — route the image path

`ImageExtractor.extract` (:500) never calls `_layout_blocks_and_tables` — one
page-wide paragraph per page (:533). It already holds `page_result.regions` and
`pix`. Three wrinkles: it builds `Element`s directly rather than via
`_accum_to_elements` (so `_table_to_element` needs a shared home), tables
must interleave by y rather than append, and it too now has an LLM branch
(:521-524, markdown + no regions) that must pass through untouched per A0.
The A3 narrative-subtraction rule applies here identically — the page-wide
paragraph currently contains the table text.

---

# Track B — benchmark + accuracy extension

### What exists to build on

| Piece | Where | Fit |
|---|---|---|
| `_results` accumulator + per-fixture dicts | `test_fixture_accuracy.py:169` | add a `tables` key |
| `generate_extraction_report(results)` | `accuracy_reports.py:252` | add a section emitter |
| `write_report` session fixture | `test_fixture_accuracy.py:725` | already regenerates EXTRACTION.md |
| `structural_fidelity` / `data_integrity` / `key_column_preservation` / `schema_conformance` | `tabular_metrics.py:31/80/160/195` | DataFrame-shaped; wired **only** to spreadsheet ingest today (`evaluation.md` §2) |
| DocLayNet layout harness incl. per-class table F1 | `test_fixture_accuracy.py:448` | the decomposition in B1 hangs off this |

### B0 — the table metric is currently wrong; fix it before measuring against it

Two things surfaced reading the harness, and both change what "improvement" means.

**(a) steering's Layout Detection paragraph is stale.** It says "0 predictions
across all DocLayNet fixtures". EXTRACTION.md's Per-Class table already reports
`table | TP 1 | FP 0 | FN 3 | P 100% | R 25% | F1 40%`. The layout model *does*
detect tables. That paragraph needs correcting regardless of #17.

**(b) the 25% recall is largely an annotation artefact.** Aggregating GT spans
per `_aggregate_doclaynet_blocks` (:218) across the vendored fixtures gives
exactly 4 GT `Table` blocks:

```
dense_text_548    table blocks=3   word runs = [397, 1, 1]
sparse_text_344   table blocks=1   word runs = [8]
diverse_layout_49 / formula_29 / table_0   table blocks=0
```

The 397-word run is the real table; the two 1-word runs are single stray
Table-labelled words splitting the run, counted as separate GT blocks. So the
detector plausibly found the one real table and is charged 3 FNs for aggregation
artefacts and an 8-word fragment. **Fix the aggregation for the table class (or
report table detection at region granularity with a minimum-span filter) before
anyone reads a reconstruction number against it** — otherwise #17 inherits a
detector "ceiling" that isn't real.

Also worth flagging: **`table_0` contains no Table-labelled GT at all** (196
Text, 2 Section-header, 1 Page-footer). Despite the name it is not a table
fixture and must not be used as one.

Net: the vendored set supports **one** real table page. That drives B3.

### B1 — decompose the measurement into two stages

Report reconstruction **conditioned on a correct region**, not blended with
detector recall, or A1's quality is unreadable:

1. **Detection** — does layout return a `table` region covering the GT table?
   (existing per-class F1, fixed per B0.)
2. **Reconstruction** — feed the *GT* table rect straight to
   `reconstruct_table` and measure the grid. This is the number that tracks A1.
3. **End-to-end** — real pipeline, detector included. The number that matters
   for the corpus, and the product of 1 × 2.

### B2 — the metric set

`tabular_metrics` assumes aligned DataFrames with matching column names, which a
reconstructed OCR grid will not have. Add a thin alignment projection
(`cells → DataFrame`, header row → column names) then:

- **Structure**: detected rows/cols vs GT rows/cols (exact, plus off-by-one rate).
- **Cell content F1** on `(row, col, text)` triples after alignment — catches
  column-shift errors that a cell-count metric misses.
- `data_integrity` for exact-match cell score; `key_column_preservation` where a
  key column exists (the CSV route in B3 gives one).
- **False-table rate (the precision guardrail)**: run the reconstructor over the
  fixtures with *no* GT table (`diverse_layout_49`, `formula_29`, `table_0`,
  plus FUNSD forms — dense label/value pairs are the likeliest false positive).
  Any table emitted is an FP. This is what makes "precision over coverage"
  falsifiable rather than a stated intention.
- **Task metric**: money amounts recovered on `dense_text_548` — 1 today, 30
  predicted. Per CLAUDE.md's analysis checklist this is the metric that decides
  whether the structural gains matter downstream.

Not proposing TEDS (the standard table-recognition metric) — it needs a tree
edit-distance dependency. Note it as the upgrade path if the alignment metric
proves too coarse; don't add it speculatively.

### B3 — ground truth supply, given one real fixture

- **Real scanned GT — landed** (`7f756cc`). `dense_text_548_table.csv` +
  `.meta.json` now exist and pass the A.6 checks (39 × 11 rectangular, unique
  headers, no trailing whitespace, `n_header_rows: 3`); the A.4 name-own-row
  and A.2 em-dash conventions were applied as recommended. This unblocks
  B1.2 and B2. The A.6 checker still needs writing — the checks above were
  run by hand once, not wired into the suite.
- **`sparse_text_344` GT landed off-spec and must be normalised before the
  B2 loader exists.** `sparse_text_344.csv` lacks the `_table` suffix and its
  `.meta.json` uses a different schema entirely
  (`{"table": {header, data}, "context": {before, after}}` instead of
  `{"n_header_rows", "key_column", "notes"}`). Two GT files, two conventions
  — one loader cannot consume both. Either rename/rewrite it to the
  Appendix A spec or declare it non-GT (B0 already flagged its 8-word block
  as a fragment of marginal value; a 4-row single-column table exercises
  almost nothing).
- **Volume GT, free and exact** —
  `womblex-collection/_spreadsheets/Approved-providers-au-export_20260204.csv`
  (10,859 × 10, with a natural key column) and `mso-statistics-sept-qtr-2025.xlsx`
  already ship in the fixtures. Render → PDF → rasterise → OCR → reconstruct →
  compare against the **source DataFrame**. Arbitrary volume, zero annotation
  cost, and exactly the shape `tabular_metrics` was built for. Caveat to state in
  the report: a clean render is easier than a real scan, so this measures the
  reconstructor's ceiling, not scan robustness — it complements `dense_text_548`,
  it does not replace it.
- Optional later: pull more Table-labelled pages from the full DocLayNet clone
  (per THIRD_PARTY_DATA.md) so the real-scan side isn't n=1.

### B4 — report + docs wiring

- New `_results["tables"]` entries → new `## Table Reconstruction` section in
  `generate_extraction_report` (extend EXTRACTION.md rather than starting a
  `TABLES.md`; it sits directly under the per-class layout section it decomposes).
  Columns: fixture, GT rows×cols, detected rows×cols, cell F1, data integrity,
  gate outcome, money recall.
- Separate **false-table** table so the guardrail is visible, not buried.
- `docs/evaluation.md`: §2 is "Tabular Extraction Accuracy" scoped to
  CSV/XLSX → parquet. Document table reconstruction is a *different* capability
  against different GT — add §2b (or split the table) so the two aren't conflated.
- **Knock-on: CHUNKING.md.** Tables reach the chunker as one `TableData` →
  markdown each. Landing tables on scanned pages changes chunk counts and
  composition, so the chunking accuracy doc shifts too — regenerate and
  sanity-check it in the same run rather than being surprised later.
- All regeneration via the test run (`write_report`, :725). Never hand-edit.

### B5 — regression guard

Assert-with-threshold on the new metrics (structure exact-match, cell F1 floor,
**false-table count == 0**), so a later tuning change can't quietly trade
precision for coverage. Today's benchmark tests end in `assert True` (:530) and
only report — the table gates should be the first that actually fail. The
benchmark must pin `engine="paddleocr"` (per A0) so a config default change
to an LLM engine can't silently turn the gates into no-ops.

---

## Sequencing

`B0 → A1/A2 (behind the GT-rect harness in B1.2) → A3 → A4 → B2/B3 breadth → B4/B5`.
B0 first because it is a live doc/metric defect. B1.2 before A3 so the
reconstructor is tunable without the detector in the loop.

## Open decisions

1. Shared `table_grid.py` vs duplicating the algorithm in `ocr_tables.py`.
   Recommendation: share — `spreadsheet_print` is at 505 lines against the
   750 cap, and a duplicated binning algorithm is exactly the parallel
   implementation CLAUDE.md's thin-adapter rule tells us to delete.
2. Precision-gate thresholds — set from `dense_text_548` + the false-table set.
3. B0 remedy: fix `_aggregate_doclaynet_blocks` for tables, or report table
   detection at region granularity with a min-span filter?
4. Do the new gates fail the build, or warn until the numbers settle?
5. ~~GT spec sign-off~~ — **resolved by `7f756cc`**: A.4 (name-own-row) and
   A.2 (verbatim em-dash) were applied as recommended in the landed
   `dense_text_548_table.csv`. A.5 (scorer-side dash folding) remains a B2
   implementation detail, not a GT question. What replaced it: the
   `sparse_text_344` normalisation call in B3.
6. LLM-OCR path (mistral-ocr / ollama): parse Pixtral markdown pipe tables
   into `TableData` as a third feeder, or leave the LLM path table-less?
   Out of scope for #17 either way (see A0); decide before it's needed.
7. `_table_aware_text` end-state — accept the A1 recommendation (reconstructor
   supersedes it inside detected table rects, it keeps covering undetected
   runs) or fold it into `table_grid.py` wholesale?

---

# Appendix A — GT authoring spec for `dense_text_548` (human)

The page is a *Grants of Plan-Based Awards* proxy table: **11 columns**, a
7-line stacked header, and hierarchical rows (participant name, then award
sub-rows). It is the hardest realistic shape we have, which is why it is the
anchor fixture.

### A.1 Files to produce

| File | Purpose |
|---|---|
| `fixtures/fixtures/doclaynet/dense_text_548_table.csv` | the GT grid |
| `fixtures/fixtures/doclaynet/dense_text_548_table.meta.json` | `{"n_header_rows": N, "key_column": null, "notes": "..."}` |

Sits beside the existing `_transcript.txt` sidecar, following that convention.
UTF-8, `\n` line endings, RFC4180 quoting. If a page ever has two tables, use
`_table1.csv` / `_table2.csv`; this one has a single table.

**No bounding box needed** — the GT table rect is derived automatically from the
`Table`-labelled bboxes in `dense_text_548.json` (x 24–974, y 83–734). Don't
hand-measure it.

### A.2 The governing rule: transcribe the page, not the intended table

GT measures extraction fidelity, so it records **what is printed**, verbatim:

- Keep thousands separators exactly as printed — `32,031`, not `32031`.
- Keep the em-dash null marker `—` (U+2014) as its own cell value. Do **not**
  convert it to an empty cell; empty and `—` are different observations.
- Keep unit annotations where printed — `($)`, `(#)`, `($/share)`.
- **Do not fix the source document's own oddities.** The column-letter row reads
  `(a) (c) (d) (e) (c) (g) (e) (i) (1) (j) (k) (2)` — with letters repeating and
  footnote markers `(1)`/`(2)` interleaved. That is what the page says.
  Transcribe it; do not renumber.
- Do not correct spelling, spacing or alignment to what the table "should" be.

### A.3 Header handling — the one thing that will break the metrics if freelanced

The header stacks 7 printed lines, with `Estimated Future Payouts Under
Non-Equity Incentive Plan Awards` spanning three sub-columns and an identical
`…Equity Incentive Plan Awards` group spanning three more. So `Threshold`,
`Target` and `Maximum` each appear **twice**.

- Emit **one flattened header row** as CSV row 1, and make every column name
  **unique** — pandas silently mangles duplicates and `structural_fidelity`
  compares column *sets*, so duplicates corrupt the metric. Qualify with the
  spanning group: `Non-Equity Threshold`, `Equity Threshold`, etc.
- Record the *printed* stacked rows as ordinary data rows underneath (the units
  row `($) ($) (#)…` and the column-letter row), and set `n_header_rows` in the
  meta json to how many leading CSV rows are header. The reconstructor is not
  expected to flatten a spanning header; the scorer uses `n_header_rows` to
  compare like with like.
- Column 1 is the row-label column (`Name (a)`); columns 2–11 are the ten data
  columns.

### A.4 Row handling — needs a decision, then apply it consistently

Participant names occupy their own visual lines (`Craig A.` / `Rogerson`) above
their award sub-rows (`2019 ICP`, `2019 LTI`, `2019 MIP`, `RSUs`, `PSUs`).

- **Recommended:** the name is one row with column 1 populated and columns 2–11
  empty; sub-rows follow as their own rows. Do **not** forward-fill the name
  onto sub-rows. Forward-filling describes a normalised relational view, but a
  geometric reconstructor cannot produce it, so GT would be scoring an
  impossible target.
- Names split across two printed lines (`Craig A.` / `Rogerson`) join into one
  cell with a single space — that is a line wrap, not two rows.
- Every CSV row has exactly 11 fields. Empty cells are empty strings.

### A.5 What this implies for scoring (not for the author)

`data_integrity` is exact string match, so the comparison side must declare a
normalisation — NFKC plus dash folding at minimum, or every `—` cell fails when
OCR reads a hyphen. **GT stays verbatim; the scorer declares its normalisation.**
Flag it here so the two don't drift into normalising the GT instead.

### A.6 Acceptance checks before the GT is used

A small checker (part of B2, not the author's job) asserts: rectangular (every
row 11 fields), column names unique and non-empty, UTF-8 with no BOM, no
trailing whitespace in cells, `n_header_rows` consistent with the file, and the
GT cell count within a sane band of the 399 `Table`-labelled words in the
fixture json. A GT that fails these is a bug in the GT, not in extraction.
