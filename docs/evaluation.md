# Evaluation Metrics

Evaluation metric tables mapping each process to its candidate technology and dependency status. All recommendations align with the normalisation-with-accuracy mandate and the Womblex architecture.

See `docs/accuracy/` for measured baselines per stage.

## 1. Text Extraction Accuracy

**Scope:** PDF, DOCX, TXT → narrative text / Parquet

**Status:** Implemented in `utils/metrics.py`. Benchmarked in `docs/accuracy/EXTRACTION.md`. End-to-end scoring against a per-page GT packet runs via `womblex score --labels <dir> --shards <dir> [--group-by FIELD]` (CLI) or `womblex.score.score_labels()` (Python); the labels packet convention is `<stem>.gt.md` + `<stem>.meta.json` with `source_file` and `page` keys.

| Metric | Implementation | Location | Ground Truth Source |
|--------|---------------|----------|---------------------|
| **CER** | Numpy-accelerated Levenshtein (char-level edit distance / reference length). Short strings (≤500 chars) use pure-Python DP; longer strings use numpy vectorised row operations. | `utils/metrics.py → cer()` | `_transcript.txt` sidecars (FUNSD, DocLayNet, Womblex Collection) |
| **CER-s** | Spatially sorts both GT and OCR words by bounding-box centroid (top-to-bottom, left-to-right within line tolerance), then computes CER on the sorted text. Isolates recognition errors from reading-order errors. | `utils/metrics.py → cer_spatial()`, `spatial_sort_text()` | FUNSD JSON bounding boxes + transcripts |
| **WER** | Whitespace-tokenised Levenshtein (word-level edit distance / reference word count). | `utils/metrics.py → wer()` | Same as CER |
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

**Status:** Implemented in `ingest/ocr_tables.py` (`reconstruct_table`) over the shared `ingest/table_grid.py` binning; benchmarked in `docs/accuracy/EXTRACTION.md` (§ *Table Reconstruction*). Distinct from §2 — §2 measures a *spreadsheet file* → parquet (the source is already a grid); §2b measures a *table detected on a page image* being reconstructed from OCR quads, where the grid itself is inferred and can be wrong. Round-1 scope is flat, contemporary tables (see `docs/table-cell-reconstruction-plan.md`); hard shapes (skew, stacked spanning headers, hierarchical rows) are refused cleanly, not solved.

Measurement follows the plan's two-stage decomposition: **detection** is the per-class `table` layout F1 (§1's DocLayNet harness); **reconstruction** is scored *conditioned on a correct table rect* (B1.2 — the GT rect is fed straight to `reconstruct_table`, no detector in the loop), so the reconstruction number tracks the grid builder alone.

| Metric | Implementation | Location | Ground Truth Source |
|--------|---------------|----------|---------------------|
| **Structural Fidelity** | Same `structural_fidelity` as §2, over an alignment projection (`cells → DataFrame`, header row → uniquified column names). Rows/cols/column-name agreement. | `utils/tabular_metrics.py → structural_fidelity()`; projection in `test_table_benchmark.py → _table_to_frame()` | Rendered-clean GT (source spreadsheet drawn to a page) + `dense_text_548_table.csv` |
| **Cell Match** | Positional `(row, col, text)` agreement after alignment — catches a column-shift a cell count misses. Scorer normalises NFKC + dash-fold + whitespace-collapse; GT stays verbatim. | `test_table_benchmark.py → _score()` | Rendered-clean GT strings |
| **Data Integrity** | Same exact-match `data_integrity` as §2, over the alignment projection. | `utils/tabular_metrics.py → data_integrity()` | Rendered-clean GT |
| **False-Table Rate** | Reconstructor run over pages with **no** GT table (non-table DocLayNet pages + FUNSD forms); any emitted table is a false positive. Makes "precision over coverage" falsifiable; calibrated `MIN_ROW_FILL_RATIO`. | `test_table_benchmark.py → TestFalseTableCohort` | Non-table fixtures (no GT needed) |
| **GT Acceptance** | Appendix-A.6 conformance checks on any `*_table.csv` GT (rectangular, unique headers, no BOM/trailing whitespace, `n_header_rows` consistent, plausible cell count). A GT that fails is a bug in the GT. | `test_table_benchmark.py → TestGroundTruthAcceptance` | `<fixture>_table.csv` + `.meta.json` |

Not measured: **money recall** (the downstream payoff) — the benchmark has no labelled money ground truth (see `docs/money.md`), so no honest recall can be quoted. TEDS is the noted upgrade path if the alignment metric proves too coarse.

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

| Metric | Candidate Technology | Implementation Note | Ground Truth Source | Dependency Status |
|--------|---------------------|---------------------|---------------------|-------------------|
| **Entity Resolution Accuracy** | Custom ID Matching | % of extracted entities correctly linked to canonical forms. | Isaacus `enrichment/` fixtures | ✅ Pure Python |
| **Graph Topology Validity** | Manual / `networkx` (optional) | Checks for orphan nodes, broken edges, schema compliance. | `graph.jsonl` fixtures | ⚠️ Optional (`networkx` BSD-3) |
| **Query Fidelity** | Custom Assertion | Graph queries return expected results against Parquet stores. | Manual test cases | ✅ Pure Python |

---

## Dependency Landing Notes (`pyproject.toml`)

| Library | License | Reason for Addition | Replaces / Avoids | CI / Build Notes |
|---------|---------|---------------------|-------------------|------------------|
| **`pyogrio`** | MIT | Fast SHP I/O engine. Used by `geopandas` as the read backend. | `fiona`, direct `GDAL` Python bindings | Ships pre-compiled GDAL wheels. |
| **`geopandas`** | BSD-3 | GeoDataFrame for SHP → GeoParquet conversion. Writes GeoParquet via `to_parquet()`. | Manual geometry handling | Uses `pyogrio` engine + `pyarrow` for I/O. |
| **`shapely`** | BSD-3 | Geometry validity predicates (`is_valid`). Dependency of `geopandas`. | Heavy GIS stacks | Bundles GEOS C-lib. Stable across Linux/macOS/Windows wheels. |
| **`pandas` / `pyarrow`** | BSD-3 / Apache-2.0 | Already in core deps. Used via `.testing` and `.Schema` for tabular validation. | N/A | No new dependencies added. |

## Alignment with Architecture & Mandate

- **Normalisation Focus:** Metrics strictly validate structural preservation and textual fidelity. No complex spatial analytics, no graph traversal math, no embedding fusion.
- **True to Source:** Existing `utils/metrics.py` Levenshtein + `pandas.testing` + `shapely` guarantee exact structural/geometry validation.
- **Zero Friction:** All recommended libs are permissively licensed (MIT/BSD/Apache-2.0), ship pre-compiled wheels, and avoid C-extension install issues.
- **Maintenance:** Validation logic lives in `utils/metrics.py` and `utils/tabular_metrics.py` (keeping under the 750-line cap via modular helpers). The system remains detection-first, config-driven, and dependency-light.
