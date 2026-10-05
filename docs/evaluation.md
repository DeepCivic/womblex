# Evaluation Metrics

The metrics each process is evaluated by, how each is applied, and its dependency status. Evaluations with no numbered section below:

- **Chunking** (`CHUNKING.md`) — no chunking ground truth exists, so it gates the reference-free invariants (budget, offsets, contiguity) rather than scoring accuracy.
- **Money** (`MONEY.md`) — `find_money` against tag-labelled transcripts: narrative recall, precision and scale accuracy; table figures reported but not gated.
- **Ground truth** (`GROUND_TRUTH.md`) — micro-averaged CER of corrected ground-truth units against their pinned baselines.
- **Spreadsheets** (`SPREADSHEETS.md`) — spreadsheet sources scored cell-wise against themselves with the section 2 metrics.
- **Stage sequence** (`STAGE_SEQUENCE.md`) — each text-layer overlay sequence scored against the same corrected ground-truth units over identical segment boundaries.

## 1. Text Extraction Accuracy

**Scope:** PDF, DOCX, TXT → narrative text / Parquet

**Status:** Implemented in `utils/metrics.py`. Benchmarked in `docs/accuracy/EXTRACTION.md`. End-to-end scoring against a per-page GT packet runs via `womblex score --labels <dir> --shards <dir> [--group-by FIELD]` (CLI) or `womblex.score.score_labels()` (Python); the labels packet convention is `<stem>.gt.md` + `<stem>.meta.json` with `source_file` and `page` keys.

| Metric | Implementation | Location | Ground Truth Source |
|--------|---------------|----------|---------------------|
| **CER** | Normalised by default (`normalise=True`: NFC, lowercase, whitespace collapse — pass `normalise=False` for a raw comparison). Numpy-accelerated Levenshtein (char-level edit distance / reference length). Short strings (≤500 chars) use pure-Python DP; longer strings use numpy vectorised row operations. | `utils/metrics.py → cer()` | `_transcript.txt` sidecars (FUNSD, DocLayNet, Womblex Collection) |
| **CER-s** | Spatially sorts both GT and OCR words by bounding-box centroid (top-to-bottom, left-to-right within line tolerance), then computes CER on the sorted text. Isolates recognition errors from reading-order errors. | `utils/metrics.py → cer_spatial()`, `spatial_sort_text()` | FUNSD JSON bounding boxes + transcripts |
| **WER** | Whitespace-tokenised Levenshtein (word-level edit distance / reference word count), with the same default normalisation as CER. | `utils/metrics.py → wer()` | Same as CER |
| **Reading Order Accuracy** | Matches GT and extracted words by bounding-box IoU, then measures what fraction of GT word pairs preserve their relative order in the extraction output (concordant pairs / total pairs). | `utils/metrics.py → reading_order_accuracy()` | FUNSD annotations, DocLayNet word sequences |

## 2. Tabular Extraction Accuracy

**Scope:** CSV, XLSX, PSV → `.parquet`

**Status:** Implemented in `utils/tabular_metrics.py`. Integration-tested against `Approved-providers-au-export_20260204.csv` (10,859 rows × 10 columns).

| Metric | Implementation | Location | Ground Truth Source |
|--------|---------------|----------|---------------------|
| **Structural Fidelity** | Validates row count, column count, and column names match between source and extraction output. Reports missing/extra columns. | `utils/tabular_metrics.py → structural_fidelity()` | Source CSV/XLSX files |
| **Data Integrity Score** | Cell-level exact string match. NaN/None normalised to `""`. Returns score (0.0–1.0) and first N mismatches with row/column/value detail. | `utils/tabular_metrics.py → data_integrity()` | Source spreadsheets |
| **Key Column Preservation** | Verifies unique IDs are 100% preserved without duplication. Reports missing keys and duplicate keys. | `utils/tabular_metrics.py → key_column_preservation()` | Source key columns (e.g. `Provider Approval Number`) |
| **Schema Conformance** | `pyarrow.Schema.equals` — each shard parquet matches its canonical schema (`ELEMENT_SCHEMA`, `TABLE_CELLS_SCHEMA`, `FORM_FIELDS_SCHEMA`, `MANIFEST_SCHEMA`). | `utils/tabular_metrics.py → schema_conformance()` | `store/output.py` schema defs |

## 2b. Document-Table Reconstruction Accuracy

**Scope:** OCR'd PDF page / image → `kind="table"` element with cells

**Status:** Implemented in `ingest/ocr_tables.py` (`reconstruct_table`) over the shared `ingest/table_grid.py` binning; benchmarked in the *Table Reconstruction* section of `docs/accuracy/EXTRACTION.md`. Distinct from section 2 — Tabular Extraction Accuracy measures a *spreadsheet file* → parquet (the source is already a grid); this section measures a *table detected on a page image* being reconstructed from OCR quads, where the grid itself is inferred and can be wrong.

Measurement follows a two-stage decomposition: **detection** is the per-class `table` layout F1 (the DocLayNet layout-detection harness, reported in `docs/accuracy/EXTRACTION.md`); **reconstruction** is scored *conditioned on a correct table rect* (the GT rect is fed straight to `reconstruct_table`, no detector in the loop), so the reconstruction number tracks the grid builder alone. A blended end-to-end (detection × reconstruction) stage is deferred to a scanned-document round, once real-scan GT exists to blend against.

The `test_table_benchmark.py` named below lives in the **womblex-benchmark**
repository, under `accuracy/`, with the ground truth it scores against; it
publishes its rows into `docs/accuracy/EXTRACTION.md` here.

| Metric | Implementation | Location | Ground Truth Source |
|--------|---------------|----------|---------------------|
| **Structural Fidelity** | Same `structural_fidelity` as section 2, over an alignment projection (`cells → DataFrame`, header row → uniquified column names). Rows/cols/column-name agreement. | `utils/tabular_metrics.py → structural_fidelity()`; projection in `test_table_benchmark.py → _table_to_frame()` | Rendered-clean GT (source spreadsheet drawn to a page) + `dense_text_548_table.csv` |
| **Cell Match** | Positional `(row, col, text)` agreement after alignment — catches a column-shift a cell count misses. Scorer normalises NFKC + dash-fold + whitespace-collapse; GT stays verbatim. | `test_table_benchmark.py → _score()` | Rendered-clean GT strings |
| **Data Integrity** | Same exact-match `data_integrity` as section 2, over the alignment projection. | `utils/tabular_metrics.py → data_integrity()` | Rendered-clean GT |
| **False-Table Rate** | Reconstructor run over pages with **no** GT table (non-table DocLayNet pages + FUNSD forms); any emitted table is a false positive. Makes "precision over coverage" falsifiable; calibrated `MIN_ROW_FILL_RATIO`. | `test_table_benchmark.py → TestFalseTableCohort` | Non-table fixtures (no GT needed) |
| **GT Acceptance** | The [GT authoring spec](#table-ground-truth-authoring-spec) conformance checks on any `*_table.csv` GT (rectangular, unique headers, no BOM/trailing whitespace, `n_header_rows` consistent, plausible cell count). A GT that fails is a bug in the GT. | `test_table_benchmark.py → TestGroundTruthAcceptance` | `<fixture>_table.csv` + `.meta.json` |

Table-cell money recall is not scored: a bare cell under a money header is resolved from its column, which a flattened transcript does not carry. TEDS is the noted upgrade path if the alignment metric proves too coarse.

### Table ground-truth authoring spec

The normative spec for authoring a `*_table.csv` ground-truth file for a
reconstruction fixture (the anchor fixture is DocLayNet `dense_text_548`, a
*Grants of Plan-Based Awards* proxy table: 11 columns, a 7-line stacked header,
hierarchical rows — the hardest realistic shape in the corpus, which is why it
*tracks* rather than *gates*). The acceptance checks below are enforced by
`test_table_benchmark.py → TestGroundTruthAcceptance`, parametrised over every
`*_table.csv` beside a fixture.

**Files to produce.** Beside the existing `_transcript.txt` sidecar, following
that convention (UTF-8, `\n` line endings, RFC4180 quoting; `_table1.csv` /
`_table2.csv` if a page has two tables):

| File | Purpose |
|---|---|
| `fixtures/fixtures/doclaynet/<fixture>_table.csv` | the GT grid |
| `fixtures/fixtures/doclaynet/<fixture>_table.meta.json` | `{"n_header_rows": N, "key_column": null, "notes": "..."}` |

No bounding box is needed — the GT table rect derives automatically from the
`Table`-labelled bboxes in the fixture json. Do not hand-measure it.

**The governing rule: transcribe the page, not the intended table.** GT measures
extraction fidelity, so it records *what is printed*, verbatim:

- Keep thousands separators exactly as printed — `32,031`, not `32031`.
- Keep the em-dash null marker `—` (U+2014) as its own cell value; empty and
  `—` are different observations.
- Keep unit annotations where printed — `($)`, `(#)`, `($/share)`.
- Do not fix the source document's own oddities (repeating column letters,
  interleaved footnote markers). Transcribe; do not renumber.
- Do not correct spelling, spacing or alignment to what the table “should” be.

**Header handling.** Emit **one flattened header row** as CSV row 1 with
**unique** column names (pandas silently mangles duplicates and
`structural_fidelity` compares column *sets*). Qualify duplicated names by their
spanning group (`Non-Equity Threshold`, `Equity Threshold`). Record the printed
stacked rows (units row, column-letter row) as ordinary data rows underneath,
and set `n_header_rows` in the meta json. The reconstructor is not expected to
flatten a spanning header; the scorer uses `n_header_rows` to compare like with
like.

**Row handling.** Participant names occupy their own rows with the data columns
empty; sub-rows follow as their own rows. Do **not** forward-fill the name —
that describes a normalised relational view a geometric reconstructor cannot
produce. Names split across printed lines join with a single space. Every CSV
row has the full column count; empty cells are empty strings.

**Scoring implication.** `data_integrity` is exact string match, so the
comparison side declares a normalisation — NFKC plus dash folding at minimum.
**GT stays verbatim; the scorer declares its normalisation.**

**Acceptance checks before a GT is used.** The checker asserts: rectangular;
column names unique and non-empty; UTF-8 with no BOM; no trailing whitespace in
cells; `n_header_rows` consistent with the meta; and the GT cell count within a
sane band (0.3×–3×) of the fixture json's `Table`-labelled word count. A GT that
fails these is a bug in the GT, not in extraction.

## 3. Geospatial Extraction Accuracy

**Scope:** SHP → GeoParquet

**Status:** Implemented in `ingest/geospatial.py`. Integration-tested against `NTD_Register_Nat.shp` (20 features, EPSG:7844, Polygon).

| Metric | Implementation | Location | Ground Truth Source |
|--------|---------------|----------|---------------------|
| **Geometry Validity Ratio** | `geopandas.GeoSeries.is_valid` — % of features passing topological validity. | `test_geospatial.py → test_geometry_validity` | Source SHP (`ntd_register_nat_shp`) |
| **CRS Correctness** | `pyogrio.read_info()` CRS string match between source and output. | `ingest/geospatial.py` (stored in provenance metadata) | Source `.prj` file |
| **Attribute Preservation** | Column set equality between source GeoDataFrame and output GeoParquet. | `test_geospatial.py → test_attributes_preserved` | Source SHP `.dbf` table |
| **Feature Count Match** | `len(source) == len(output)`. | `test_geospatial.py → test_row_count_matches` | Source SHP |

## 4. PII Cleaning Effectiveness

**Scope:** Any input → `<ENTITY_TYPE>` tagged output

| Metric | Candidate Technology | Implementation Note | Ground Truth Source | Dependency Status |
|--------|---------------------|---------------------|---------------------|-------------------|
| **Recall** | Custom Span Overlap | `TP / (TP + FN)`. Counts GT entities correctly replaced. | Throsby fixture (12 GT entities) | ✅ Pure Python |
| **Precision** | Custom Span Overlap | `TP / (TP + FP)`. Measures over-redaction risk. | Same as above | ✅ Pure Python |
| **Context Preservation CER** | Custom (Masked) | CER calculated **only** on non-PII text spans. Ensures cleaning didn't corrupt narrative. | Throsby (PII spans masked) | ✅ Pure Python (existing `utils/metrics.py`) |

## 5. Redaction Detection Accuracy

**Scope:** PDF → Redaction-annotated output

| Metric | Candidate Technology | Implementation Note | Ground Truth Source | Dependency Status |
|--------|---------------------|---------------------|---------------------|-------------------|
| **Region Recall (IoU)** | Custom IoU Logic | % of GT `<REDACTED>` regions detected with IoU > 0.5. | Throsby fixture (7 GT redactions) | ✅ Pure Python/NumPy/CV2 |
| **False Positive Rate** | Contour Analysis | % of detected regions that do not overlap GT redactions. | Same as above | ✅ Uses `opencv-python-headless` |
| **Mode Compliance** | String/Flag Inspection | Verifies `flag`/`blackout`/`delete` modes applied correctly. | Config + output | ✅ Pure Python |

## 6. Enrichment & Graph Quality

**Scope:** Chunks → Entity Mentions / Knowledge Graph

**Status:** Not yet implemented. No `graph.jsonl` or `fixtures/isaacus/enrichment/` ground truth exists, and there is no enrichment or graph accuracy report (`tests/test_graph.py` / `tests/test_enrich.py` are unit tests, not accuracy benchmarks). The table below is the proposed metric set, not a measured baseline.

| Metric | Candidate Technology | Implementation Note | Ground Truth Source | Dependency Status |
|--------|---------------------|---------------------|---------------------|-------------------|
| **Entity Resolution Accuracy** | Custom ID Matching | % of extracted entities correctly linked to canonical forms. | Isaacus `enrichment/` fixtures | ✅ Pure Python |
| **Graph Topology Validity** | Manual / `networkx` (optional) | Checks for orphan nodes, broken edges, schema compliance. | `graph.jsonl` fixtures | ⚠️ Optional (`networkx` BSD-3) |
| **Query Fidelity** | Custom Assertion | Graph queries return expected results against Parquet stores. | Manual test cases | ✅ Pure Python |

---

## Dependency Notes

| Library | License | Status | Use | Notes |
|---------|---------|--------|-----|-------|
| **`pyogrio`** | MIT | Optional — not in `pyproject.toml`; install separately | SHP read engine behind `geopandas`; CRS read via `read_info()`. | Ships pre-compiled GDAL wheels. Listed only in the mypy ignore list. |
| **`geopandas`** | BSD-3 | Optional — not in `pyproject.toml`; install separately | GeoDataFrame for SHP → GeoParquet (`to_parquet()`). | `ingest/geospatial.py` imports it lazily and raises an install hint when absent; `test_geospatial.py` skips via `importorskip`. |
| **`shapely`** | BSD-3 | Optional — pulled in by `geopandas` | Geometry validity predicates (`is_valid`). | Bundles GEOS. |
| **`pandas` / `pyarrow`** | BSD-3 / Apache-2.0 | Core | Plain DataFrame comparison and `pyarrow.Schema.equals` in `utils/tabular_metrics.py`. | No new dependencies. |
