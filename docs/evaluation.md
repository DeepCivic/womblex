# Evaluation Metrics

Evaluation metric tables mapping each process to its candidate technology and dependency status. All recommendations align with the normalisation-with-accuracy mandate and the Womblex architecture.

See `docs/accuracy/` for measured baselines per stage.

## 1. Text Extraction Accuracy

**Scope:** PDF, DOCX, TXT → narrative text / Parquet

**Status:** Implemented in `utils/metrics.py`. Benchmarked in `docs/accuracy/EXTRACTION.md`.

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
| **Schema Conformance** | `pyarrow.Schema.equals` — Parquet file schema matches `EXTRACTION_SCHEMA` exactly. | `utils/tabular_metrics.py → schema_conformance()` | `store/output.py` schema def |

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
