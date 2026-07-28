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

## Where tables come from today (why this scope is enough)

| Path | Mechanism | Status |
|---|---|---|
| Native PDF | `_find_native_tables` — PyMuPDF `find_tables` (lines + text strategies), cross-checked, cellified (`orchestrator.py:104`, `:124`) | **Works.** Covers the digital-native majority of the round-1 corpus |
| Spreadsheet-print PDF | `ingest/spreadsheet_print.py` behind its qualifier | **Works** |
| OCR'd page (PDF or image) | `_layout_blocks_and_tables` detects the region, `tables` returned empty | **The gap.** #17 closes it |
| LLM-OCR (mistral-ocr, ollama) | Markdown, no regions | Out of scope (A0) |

Everything downstream of `TableData` already works: `_table_to_element`
(`orchestrator.py:250`) cellifies it, `_accum_to_elements` (:295) places it
by (y, x), the writer denormalises to `table_cells.parquet`. **The only
missing piece is producing a `TableData` on the OCR paths.** No schema
change, no new element kind.

---

# Track A — the extraction fix

### A0 — scope: region-based engines only

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

### A1 — one algorithm: `ingest/table_grid.py`

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

### A2 — skew: refuse, don't solve (round-1 cut)

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

`ImageExtractor` doesn't preprocess (:518-520), so the image path has no
mismatch and no refusal condition.

### A3 — wire the OCR-PDF path

`_layout_blocks_and_tables` gains `regions` + `pix_dims` (already held at
orchestrator `:163`, per A0). Populate the `tables` list declared at :421
and returned empty at :467/:477.

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

### A4 — route the image path

`ImageExtractor.extract` (:500) never calls `_layout_blocks_and_tables` —
one page-wide paragraph per page (:533). It already holds
`page_result.regions` and `pix`. Wrinkles: it builds `Element`s directly
rather than via `_accum_to_elements` (so `_table_to_element` needs a shared
home), tables must interleave by y rather than append, the LLM branch
(:521-524) passes through untouched per A0, and the A3 narrative-subtraction
rule applies identically — the page-wide paragraph currently contains the
table text.

### A5 — post-processing conventions and lineage: preserved by construction

Because Track A produces a `TableData` and reuses `_table_to_element`, every
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

### B0 — fix the table metric before measuring against it

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

### B1 — decompose the measurement into two stages

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
- **`sparse_text_344`: declare non-GT and remove its CSV/meta.** It landed
  off-spec (no `_table` suffix; a different meta schema than Appendix A) and
  its 8-word, 4-row single-column block exercises almost nothing. One GT
  convention, one loader — carrying a second format for a marginal fixture
  is exactly the duplication this round avoids.
- Later: more Table-labelled pages from the full DocLayNet clone (per
  THIRD_PARTY_DATA.md) when the scan round arrives.

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

`B0 → B3 rendered-GT harness + B1.2 → A1 → A3 → A4 → B2 breadth → B4/B5`.
B0 first because it is a live doc/metric defect. The rendered-GT harness and
B1.2 land before A3 so the reconstructor is tuned against clean grids
without the detector in the loop.

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

1. Precision-gate thresholds — set from the rendered-clean fixtures + the
   false-table set (not from `dense_text_548`, which no longer gates).
2. B0 remedy: fix `_aggregate_doclaynet_blocks` for tables, or report table
   detection at region granularity with a min-span filter?

Resolved this revision: shared `table_grid.py` (A1 — anti-duplication);
`_table_aware_text` end-state (A1); skew handling (A2 — refuse);
gate-vs-warn (B5 — gate clean, track dense); GT sign-off (landed in
`7f756cc`, conventions as recommended); `sparse_text_344` (B3 — non-GT).

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
