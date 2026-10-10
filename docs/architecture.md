# Architecture

Womblex extracts and normalises Australian government data for analysis. 
Input files are routed by format to the appropriate ingest path. 
Operations are independent functions with clear preconditions — 
callers compose them directly.

```
Input File
│
├─ Narrative (PDF/DOCX/TXT/MD) ──► Extract Text ──► [.txt or .parquet]
├─ Tabular (CSV/XLSX) ────────► Transform Rows ──► [.parquet]
├─ Tabular (PSV/G-NAF) ──────► Standalone Ingest ──► [.parquet]
├─ Register (XML/ABN) ───────► Standalone Ingest ──► [.parquet]
└─ Geospatial (SHP) ─────────► Transform Geometry ──► [GeoParquet]
        │
        ▼  (Downstream stages over the shard dir — `run-stage` / `enqueue-stages`, in PIPELINE_ORDER)
        ├─ enrich        — Isaacus enrichment (needs elements + manifest; chunks optional)
        ├─ chunk         — split text into token-bounded chunks
        ├─ graph-refresh — offline mention→chunk edge rebuild
        └─ pii           — mask graph PII spans as numbered <PERSON_n> tags in *.clean_text.parquet (terminal, after enrich + embed)
```

`womblex run` is extraction-only (`batch.process_batch`); downstream stages run per shard directory via `womblex <stage> --shards`, `run-stage` or `enqueue-stages`, ordered by `PIPELINE_ORDER` in `pipeline_order.py` (enrich runs before chunk so AI chunking can reuse the enrichment). Each stage's inputs are declared by its `StageContract` in `cloud/stage_contracts.py`: chunk needs an extraction; enrich needs only elements + manifest; graph-refresh needs enrichment entities, graph edges and chunks; pii needs chunks (enrichment entities are a config-derived conditional input). Redaction detection is optional inside `process_batch` and otherwise runs via `womblex redact --shards`. The in-memory graph is built by `build_document_graph` in `analyse/graph.py`.

## Stage Detail

### 1. Ingest — Detection

`detect.py` profiles each document before any text extraction occurs. Detection is signal-based: it examines the text layer, embedded images, table structures, and image morphology to assign a `DocumentType`.

**Detection signals, in priority order:**

| Signal | Method | Drives |
|--------|--------|--------|
| Text layer coverage | `page.plain_text()` length per page | Native vs scanned split |
| Table coverage | Regex on text + `page.find_tables()`, per-page count | STRUCTURED (≥80%) or structured content flag |
| Image presence | `page.images()` | Scanned/hybrid flag |
| Ruled lines | Morphological horizontal line detection | Handwriting signal |
| Glyph regularity | Connected-component height variance | Typed vs handwritten |
| Stroke width variance | Skeleton distance-transform CV | Typed vs handwritten |
| OCR confidence | Per-region confidence scores (0–1) | Typed vs handwritten fallback |

PaddleOCR is only invoked as a fallback when morphological signals (glyph regularity + stroke width) are both inconclusive. Confidence scores per text region are stored in `DocumentProfile.ocr_region_confidences`.

**Classification logic:**

```
if file is .docx → DOCX
if file is .md/.markdown → MARKDOWN
if file is .csv/.xlsx → SPREADSHEET
if text_coverage >= 30%:
    if table_ratio >= 80% → STRUCTURED
    elif has_tables or has_images → NATIVE_WITH_STRUCTURED
    else → NATIVE_NARRATIVE
elif 10% < text_coverage < 30% and has_text and has_images:
    → HYBRID (mixed native + scanned pages)
elif has_images:
    if handwriting_signals >= 80% → SCANNED_HANDWRITTEN
    elif has_handwriting → SCANNED_MIXED
    elif morphology_score >= 0.6 → SCANNED_MACHINEWRITTEN
    elif morphology_score < 0.35 → SCANNED_HANDWRITTEN
    elif ocr_confidence >= 70% → SCANNED_MACHINEWRITTEN (fallback)
    elif ocr_confidence < 70% → UNKNOWN
    else → SCANNED_MACHINEWRITTEN (default when no morphology/OCR signals)
else:
    → UNKNOWN
```

Defensive classification: uncertain documents route to `UNKNOWN` rather than a wrong bucket. High `UNKNOWN` count signals detection gaps to address.

**Document types:**

| Type | Meaning |
|------|---------|
| `NATIVE_NARRATIVE` | PDF with selectable text layer, no structure |
| `NATIVE_WITH_STRUCTURED` | PDF with text layer plus tables or images |
| `SCANNED_MACHINEWRITTEN` | Image-only, typed/printed content |
| `SCANNED_HANDWRITTEN` | Image-only, handwritten content |
| `SCANNED_MIXED` | Image-only, mixed typed and handwritten |
| `HYBRID` | Some pages native, some scanned |
| `STRUCTURED` | Pure tabular content |
| `DOCX` | Word document |
| `SPREADSHEET` | CSV or Excel |
| `TEXT` | Plain text file (passthrough) |
| `MARKDOWN` | Markdown file (headings/lists/tables) |
| `IMAGE` | Photo / diagram — flagged for review |
| `UNKNOWN` | Detection failed |

### 2. Ingest — Extraction

`ingest/pdf/` is the PDF seam womblex's extractors read PDFs and images through. `ingest/pdf/types.py` holds the vocabulary — `Rect`, `Word`, `Span`/`Line`/`Block`, `FoundTable`, `Drawing`, `Widget`, `PageImage`, and the `Page` and `Document` protocols — with no third-party imports, so a backend is only loaded when a document is opened. The types are deliberately not tied to one library's shapes: the adapter converts to top-left coordinates, returns dataclasses rather than dicts and tuples, and rasterises to an array rather than a pixmap. `Word` is the one exception, kept as an 8-field NamedTuple because `grid_projection` slices it positionally. `open_document(path)` is the single entry point and imports the backend at call time. `_pdfium_doc.py` is the adapter: pypdfium2's bottom-left user space is flipped against the crop box into the same top-left space, objects inside form XObjects are composed through their forms' matrices, drawing rects come from path points so stroke width is left out as MuPDF leaves it, and `render` sizes its bitmap by `types.render_box`, MuPDF's round-out rule, so a render's shape does not depend on the backend. A file without a PDF header opens through `_image.py` instead: an image via Pillow (MuPDF's formats plus WebP and AVIF), one image-only page per frame, sized by MuPDF's page-rect rule, so images stay on the orchestrator's per-page OCR path. Text is `_text.py`: pdfium reports characters with boxes and fonts, so it rebuilds lines (stream-consecutive characters on one row, with fragments that arrive out of order rejoined to the line they abut unless a column gutter lies between), blocks (vertical adjacency by pitch, with MuPDF's measured rules for lines that start right of the last, run up the page or sit off-axis) and spans (font, size, bold), each line following its characters' writing direction so rotated text reads as a line. As MuPDF does, it reports the effective font size (pdfium's own is the `Tf` operand, 1 wherever a producer scales through the text matrix), drops characters lying off the page, and joins no line-end hyphen: the locked MuPDF keeps every one under `TEXT_DEHYPHENATE`. Bold is the font weight, the ForceBold flag or `bold` in the name. Reading order is content order, with no column reordering; pdfium sorts vertical text by position, so those characters are put back in content order from the page's text objects. Tables are `_tables.py`, an adapter onto pdfplumber's `TableFinder` (lines and text strategies, pdfplumber's defaults): pdfium supplies the ruled edges (the straight runs of each path, so a grid drawn as one path keeps its inner rules) and the character boxes, so pdfminer's layout pass never runs; edges and character dicts are built only when the strategy reads them. On a rotated page both are carried through the rotation matrix first, so tables are found in the displayed frame, as MuPDF finds them. pdfium's page objects leave annotations out, so `drawings` reads a page that has any from a flattened scratch copy of the document (NoView annotations removed first), never flattening the main one.

`extract.py` defines the `ExtractionStrategy` and `PathExtractionStrategy` protocols, shared helpers, and the `extract_text()` dispatcher. PDFs route via the per-page orchestrator (`ingest/orchestrator.py` + `ingest/page_profile.py`); the per-doc `Native*` / `Scanned*` / `Hybrid` / `Structured` strategy classes have been removed and their bodies inlined into the orchestrator's per-page operations (`_apply_native_page`, `_apply_ocr_page`). Standalone images take the orchestrator path too — the PDF seam opens one as a single-page document, so it reaches the same `_apply_ocr_page` dispatch a scanned PDF page does; `get_extractor()` is reached only by the path-based formats. OCR primitives live in `strategies_scanned.py` and the file-format extractors in `strategies_file.py` (DOCX, plain text) and `markdown.py` (Markdown); `strategies.py` re-exports for back-compat. `spreadsheet.py` handles CSV and Excel files.

`extract_text()` logs the strategy selection (`doc, type, confidence, strategy`) at INFO level, then always returns `list[ExtractionResult]`. PDF, DOCX, and spreadsheet paths each return a single-element list (one result per source file). The list shape is retained for call-site symmetry. Spreadsheet cells live as `kind='sheet_cell'` elements on the single result; `_classify_sheet` survives as a detection-time metadata helper but no longer routes extraction.

**Spreadsheet sheet classification** (`_classify_sheet` in `spreadsheet.py`) — retained as a detection-time metadata helper, but no longer routes extraction. Every workbook now emits a single `ExtractionResult` whose element stream begins with one `kind='sheet_meta'` element per sheet (dimensions, classification) followed by one `kind='sheet_cell'` element per non-empty cell.

**`SpreadsheetExtractor`** in `spreadsheet.py` was separated from `strategies.py` to keep both files under the 750-line cap. Callers import `SpreadsheetExtractor` directly from `ingest.spreadsheet`.

**Layout step** — after page profiling and before any page is extracted, `ingest/layout_step.py` runs the `layout:` model once per selected page (`layout.page_scope`) and carries the normalised regions on `ExtractionResult.layout`; `write_results` writes them to `*.layout_regions.parquet` and the layout fingerprint into that file's and the elements' footers. OCR pages read their own regions from it; redaction reads them for its exclusion zones. See [layout.md](layout.md).

**Layout backend** — scanned extractors use `PPDocLayoutAnalyzer` (`ingest/layout_onnx.py`) for layout region detection: PP-DocLayout-M (Apache-2.0) exported to ONNX and run on `onnxruntime`, its 23 labels mapped to block types via `LABEL_MAP` (e.g. `image` → `figure`, `paragraph_title` → `heading`, `table_title` → `caption`). The class list and preprocessing come from the model's own `inference.yml`. `_layout_blocks_and_tables()` in `strategies_scanned.py` consumes the layout step's regions for an OCR page, mapping the normalised boxes onto the OCR render. When a page's layout regions carry no segmented text the fallback collapses the whole page's OCR onto the dominant region's kind; if that kind is non-text (`figure`) but the OCR is substantial (≥5 words) it is promoted to `paragraph` (`_ocr_region_block_type`) so full-page scans are not dropped from chunking. Layout blocks carry no text and OCR text is not yet assigned to layout regions, so what reaches output is the table regions (the only place `reconstruct_table` runs), the dominant region's kind, and the redaction exclusion regions in `redact/stage.py`; other detected classes (heading, list item, caption, footer, footnote) do not reach the element stream. Backend contracts are formalised as `@runtime_checkable` protocols in `interfaces/protocols.py` (`OCRReader`, `LayoutAnalyzer`, `Preprocessor`).

The orchestrator's OCR per-page path (`_apply_ocr_page`) drives `_ocr_page()` which:

1. Renders the page to a numpy array at the configured DPI
2. Deskews via Hough-line skew detection
3. Binarises — skipped for clean digital renders (histogram analysis detects low noise + narrow dynamic range); OTSU if bimodal histogram, adaptive Gaussian otherwise (handles binding shadows and scanner gradients)
4. Runs OCR and returns `(text, avg_confidence, preprocessing_steps)`; warns if avg confidence < 40%

**Text policy at the extraction boundary is verbatim.** `_normalise_text` no longer runs in the extraction hot path. Whatever the producing extractor (native text layer, paddle OCR, docx, xlsx, spreadsheet_print, figure_image) emits is what lands on the element's `text` field. Downstream stages (PII, redaction, chunking) may rewrite `pages[i].text` in place, but the parquet writer reads `elements`, so on-disk content remains extraction-time verbatim.

**Output schema** is a single `elements: list[Element]` stream on `ExtractionResult`, persisted to four sibling parquet files per batch (`*.elements.parquet`, `*.table_cells.parquet`, `*.form_fields.parquet`, `*._manifest.parquet`). Element kinds: `paragraph`, `heading`, `list_item`, `caption`, `header`, `footer`, `footnote`, `signature`, `figure`, `image`, `table`, `form`, `page_break`, `sheet_meta`, `sheet_cell`. Tables nest cells on `Element.cells` in memory and flatten to `table_cells.parquet` on disk; forms flatten the same way to `form_fields.parquet`. Legacy view properties (`result.text_blocks` / `.tables` / `.forms` / `.images`) remain on `ExtractionResult` as read-only derivations for downstream stages that have not migrated.

### 3. Ingest — G-NAF (Standalone)

`ingest/gnaf.py` provides a standalone ingest path for the [G-NAF](https://data.gov.au/data/dataset/geocoded-national-address-file-g-naf) national address dataset. G-NAF is pure structured relational data distributed as headerless pipe-delimited (`.psv`) files — NLP operations (redaction, chunking, PII, enrichment) are irrelevant and bypassed entirely.

`ingest/gnaf_schema.py` provides static, versioned column definitions for all 35 G-NAF table types (16 Authority Code lookup tables + 19 Standard tables), derived from the official `GNAF_TableCreation_Scripts` SQL.

The ingest reads each PSV via `pyarrow.csv` (streamed, constant memory), applies the schema's column names, and writes one Parquet file per input PSV. Design principles:

- **Zero semantic mutation:** All columns stored as strings. No type coercion, no null inference. Empty strings remain `""`.
- **Provenance metadata:** Each Parquet file carries `gnaf.schema_version`, `gnaf.table_name`, `gnaf.state`, `gnaf.source_file`, `gnaf.row_count`, and `gnaf.source_md5` as key-value metadata.
- **Fail-fast on schema mismatch:** Column count is validated against the static schema. Unrecognised filenames or unknown table names are skipped with a warning.

CLI: `womblex ingest-gnaf <input_dir> -o <output_dir> [--no-md5]`

### 4. Ingest — Geospatial (Standalone)

`ingest/geospatial.py` provides a standalone ingest path for ESRI Shapefiles. Like G-NAF, this bypasses the NLP operations — geospatial data is structured geometry, not narrative text.

The ingest reads SHP files via `pyogrio`, validates geometry with `shapely`, and writes GeoParquet via `geopandas`. Design principles:

- **Zero semantic mutation:** All attributes preserved as-is. Geometry and CRS carried through exactly.
- **Geometry validation:** Invalid geometries are counted and logged as warnings, not silently dropped.
- **Provenance metadata:** Each GeoParquet file carries `geospatial.source_file`, `geospatial.feature_count`, `geospatial.crs`, `geospatial.geometry_type`, `geospatial.invalid_geometries`, and `geospatial.source_md5`.

CLI: `womblex ingest-geo <input_dir> -o <output_dir> [--no-md5]`

### 5. Ingest — ABN Bulk Extract (Standalone)

`ingest/abn_bulk.py` provides a standalone ingest path for the [ABN Lookup bulk extract](https://data.gov.au/data/dataset/abn-bulk-extract) — a weekly snapshot of the Australian Business Register, distributed as 20 XML files (~6 GB uncompressed, ~11M ABNs). Like G-NAF, this is a reference register: NLP operations are irrelevant and bypassed entirely.

Each file is stream-parsed (`ET.iterparse`, constant memory) and projected into two Parquet siblings:

- `<stem>.parquet` — one row per ABR record: ABN/status/dates, entity type, the main entity name or legal-entity name parts (given names kept as separate `given_name_1` / `given_name_2` columns, since a single given name may itself contain a space), state/postcode, ACN, GST.
- `<stem>_names.parquet` — one row per registered name (main/legal, business, trading, DGR fund), keyed by ABN, shaped for `link/` register consumption.

Design principles:

- **Zero semantic mutation:** All columns stored as strings. Absent optional fields become `""`, never null.
- **Provenance metadata:** Each Parquet file carries `abn.schema_version`, `abn.source_file`, `abn.source_md5`, and `abn.row_count` as key-value metadata.
- **Per-file failure isolation:** Any failure (malformed XML, read/write error) logs with the source name, removes partial output, and lets the directory ingest continue. Files whose root element is not `Transfer` are skipped with a warning.

CLI: `womblex ingest-abn <file-or-dir> -o <output_dir> [--no-md5]`

### 6. Redact — Post-Extraction Redaction

`redact/stage.py` runs as a separate operation after extraction. Detection is vector-first: filled near-black rectangles from the page's `drawings()`. Only a page with none falls back to rendering it and running the CV2-based `RedactionDetector`, excluding the layout step's figure and table regions (`use_layout_filter`; a page with no usable layout runs unfiltered, warned once per document and recorded for `unfiltered_redaction_pages`). It then applies the configured mode:

- `flag` — sets `has_redaction=True` on affected chunks (no text change)
- `blackout` — prepends `<REDACTED>` to affected page text
- `delete` — clears affected page text entirely

The `RedactionReport` is stored on `ExtractionResult.redaction_report` for downstream stages. Non-PDF documents (spreadsheets, DOCX) are skipped — redaction detection requires a rasterisable page source.

`redact/utils.py` provides a `pre_ocr_mask()` helper for tooling that needs to mask redactions before OCR. This is not called by extraction strategies (redaction inside `_ocr_page()` caused false positives on form fields and diagram fills).

### 7. Process — Chunking

`chunker.py` wraps [semchunk](https://github.com/isaacus-dev/semchunk) v4 with full parameter exposure. Chunk size defaults to 480 tokens — sized to fit Isaacus classifier and extractor context windows (512 tokens) with 32-token headroom. Uses semchunk's native offset tracking for reliable `(start_char, end_char)` provenance.

**AI chunking + single-enrichment reuse.** Setting `chunking.chunking_model` switches the narrative path to semchunk 4's AI chunking — boundaries follow the Kanon-2 enricher's structure spans instead of the token/recursive split (opt-in, off by default). To avoid enriching the same text twice when the `enrich` stage also runs, the enrich stage persists the raw ILGS Document to `*.enrichment_doc.parquet` and `chunk_batch` reuses it per `source_hash` via `narrative_overrides`, gated by a byte-identity check (`Document.text == reassembled narrative`); on mismatch or absence the doc self-enriches. Run `enrich` before `chunk`.

The `chunk_batch()` entry point (one semchunk call across a whole batch's narratives, another across its tables):
1. Chunks narrative text with native offset tracking (no `text.find()` heuristics)
2. Converts `TableData` objects to markdown tables and chunks separately (no overlap on tables)
3. Tags each chunk with a `content_type` (`"narrative"` or `"table"`) and `has_redaction` flag
4. Repairs `<REDACTED>` markers that were split across chunk boundaries (safe with overlap)

**Adapter boundary.** semchunk owns all chunking; Womblex handles only what
semchunk can't — parquet I/O, element-stream → `ChunkInput` projection,
source-hash plumbing, and `<REDACTED>` cross-boundary repair. Every
`ChunkingConfig` field either maps directly to a semchunk parameter
(`tokenizer`, `chunk_size`, `chunking_model`, `tokenizer_kwargs`,
`memoize`, `cache_maxsize`, `max_token_chars` → `chunkerify`; `overlap`,
`processes`, `progress` → `Chunker.__call__`) or is a Womblex-only concern
(`enabled` stage gate, `chunk_tables` projection, `tokenizer_options` —
passed to the registered tokeniser's factory, not to semchunk). There is **no** Womblex toggle that
re-exposes a semchunk feature under a different name — semchunk's
parameters *are* the feature surface. Three defaults diverge from
upstream, each for a measured corpus reason:
`tokenizer="isaacus/kanon-2-tokenizer"` (matches the analysis side),
`chunk_size=480` (Kanon-2 window — upstream defaults to `None`, which
auto-derives the size from the tokeniser's `model_max_length`; that
path still passes through if `chunk_size` is set to `null`),
`processes=1` (Chromebook portability). The Kanon-2 tokeniser is free on
Hugging Face (vendored under `_models/kanon-2-tokenizer`, resolved
locally), so chunk-size counting is exact and offline. **Plain token
chunking therefore needs no API key** and runs in an air-gapped deployment
— the chunk stage gates only on the tokeniser resolving locally
(`womblex.utils.availability.tokenizer_available`). **AI chunking**
(`chunking_model`) is the one path that calls the Isaacus API per document;
it alone gates on a configured deployment — `ISAACUS_API_KEY` or
`ISAACUS_SAGEMAKER_ENDPOINTS`
(`womblex.utils.availability.isaacus_available`). `offsets=True` is pinned
in the adapter because Womblex always needs char offsets for page mapping.

When redaction mode is `flag`, the chunking stage calls `annotate_chunks()` to propagate `has_redaction=True` from the `RedactionReport` to affected chunks.

Chunking is gated by `config.chunking.enabled` and table handling by `config.chunking.chunk_tables`.

### 8. PII — Personal Information Cleaning

PII is **graph-driven**. `pii/cleaner.py` takes its primary candidates from the Kanon-2 enrichment graph — PII-typed entities (`natural`→PERSON, `address`→ADDRESS) mapped onto chunks via mention offsets — so the stage runs *after* enrichment. There is no separate primary detector; recall is flexed by enrichment granularity, not by a second pass. The per-stage entry point is `pii/pii_stage.py` (`pii_shards()` over a shard dir, drives `womblex pii --shards`); `pii/stage.py` holds the in-memory helpers for the E2E `run` path at configurable points (`post_extraction`, `post_chunk`, `post_enrichment`).

A local regex + cosine-context detector remains as an **opt-in backstop** (`pii.use_regex_backstop`, default **off**): title-case and honorific regex for PERSON validated against reference contexts via cosine similarity with `all-MiniLM-L6-v2` (threshold 0.35, calibrated on Australian government docs; the regex uses `[^\S\n]+` as the word boundary to prevent multi-line capture), plus a street-type anchor regex for ADDRESS. It is ~15% precision on this corpus (orgs/headings get tagged PERSON), so it is reserved for recall experiments.

Masking is **terminal** — it never rewrites the raw chunks that feed Isaacus. The stage writes two siblings: `*.pii_spans.parquet` (one row per span, audit/reversible, carrying the graph `entity_id`) and `*.clean_text.parquet` (the masked publishable layer, `<PERSON_1>` / `<ADDRESS_1>` typed and numbered off the graph entity, written by default). Each `clean_text` row carries `mask_status` (`masked`, `no_entity`, `not_masked`), so a chunk no candidate source covered is not labelled masked. Current coverage: PERSON and ADDRESS.

### 9. Analyse — Enrichment

Wrappers in `analyse/` call the Isaacus SDK:

- `enrich.py` — calls `kanon-2-enricher` to produce structured ILGS Documents containing segments, entities, and relationships. Handles 429 rate-limit errors with exponential backoff.
- `graph.py` — builds a `DocumentGraph` from enrichment results, mapping entities (persons, locations, terms, external documents) to graph nodes and relationships (cross-references, contact info, dates) to edges. Chunk-level mention links are computed from span offsets.
- `enrich_stage.py` — per shard directory, sends the narrative and each table's markdown (`enrichment.include_tables`) as separate request texts in the same token-budgeted requests; a table mention's offsets index that table's markdown (`text_layer = table_markdown`).
- `models.py` — ILGS data models: `Span`, `Segment`, `Person`, `Location`, `Term`, `ExternalDocument`, `Quote`, `DateInfo`, `CrossReference`, `EnrichmentResult`, and contact info types.

### 10. Store — Output

`store/output.py` writes four sibling parquet files per batch — `batch-NNNN.elements.parquet`, `batch-NNNN.table_cells.parquet`, `batch-NNNN.form_fields.parquet`, `batch-NNNN._manifest.parquet`. Downstream stages add their own per-batch sidecars over the same shard dir, each via a `womblex <stage> --shards` command and joinable on `source_hash`: `*.chunks.parquet` (chunk, I2), `*.redactions.parquet` (redact, I3), `*.enrichment_entities.parquet` + `*.enrichment_meta.parquet` + `*.graph_edges.parquet`, plus `*.enrichment_doc.parquet` when `persist_document` is set (enrich, I7), `*.entity_links.parquet` (link, I7), `*.embeddings.parquet` (embed, I7). `store/enrichment_output.py` also has a legacy E2E writer that emits three Parquet files from enrichment results:

- `entities.parquet` — entity type, name, mentions, chunk mapping
- `graph_edges.parquet` — source/target node IDs, relation type, metadata
- `enrichment_meta.parquet` — per-document enrichment summary (segment count, entity counts, etc.)

Every Parquet on the pipeline path also carries provenance in its footer key-value metadata, under one `womblex.*` namespace and additive, so a reader that ignores footers is unaffected. `store/source_provenance.py` supplies the scheme-qualified ingest root and the document's path under it — the same pair the manifest carries as `ingest_root` / `source_relpath` — and `store/run_stamp.py` supplies the run id, the Womblex version, the source commit, the digest of the *validated* configuration, the configuration's own name (its `dataset.name`, recorded as the run's *preset* whenever one is declared), the writing stage, the local models the writing process had loaded and what its pre-run model check found. The commit comes from `store/build_info.py`, which asks the work tree first and a build-time stamp second, and reports `unavailable` with a reason where neither can answer rather than defaulting to one — a version alone does not identify a build, since the same version is cut from every commit between two releases. The models are read from `utils/models.py` at footer time rather than held on the stamp, because a stamp is declared before any model loads — so each file names what was loaded when it was written, and the union across a run's files is the run's set. A second, independent key follows the same shape for the swappable slots themselves (`utils/model_registry.py`): which slot built which model, named by distribution and version, recorded only when the slot's factory is actually built rather than merely resolved — so a registration-only check (`check_registered`) writes nothing. This is what makes an API-backed OCR engine (Mistral, Ollama) establishable from the footer even though it has no artefact to digest. Extraction declares the stamp once per run and re-points it at each writer; a downstream stage does not declare its own but inherits it from a stamped sibling of the batch its sidecar sits beside (`stamp_for_sidecar`), which is what makes a local and a distributed run stamp identically. The effect is that a shard copied out of its run directory still names the run and the corpus it came from.

Two more keys, from `store/contract.py`, ride on every pipeline Parquet whether or not its run can be named: `womblex.contract_version` (the on-disk contract, versioned apart from the package) and `womblex.sensitivity` — `raw` for a file carrying unmasked document text, `masked` for `clean_text`, `none` for one carrying no document text. They are written by `_write_rows`, the one writer every sidecar shares, from the file's role; an unclassified role reads as `raw`. The manifest additionally carries a per-document `content_digest` (`store/content_digest.py`): the stable-content handle, since file bytes and `extracted_at_iso` differ between runs while extraction content under one version, config digest and model set does not.

At the two finalisation moments — the end of `womblex run` and `womblex finalize` — `store/run_manifest.py` consolidates those per-file facts into a **run record** in the run manifest's own footer (`womblex.run_record`). Footer rather than columns because the record is run grain and the manifest is document grain. Everything in it is observed rather than declared: the stages come from the run stamps the files carry, so a stage that was configured and never ran is absent while one that ran and produced nothing is present with no rows; the local models are the union of what each file recorded loading, each with a digest that recomputes from the model files alone; the documents, extraction methods and statuses are counted off the consolidated rows. Nothing is read from a configuration file, because `womblex manifest` regenerates the record from the shard directory alone and may be re-run long after the run. `womblex finalize` is the one caller that supplies its own observations rather than having them read from a directory: a distributed run's shards stay in object storage and only the manifests are staged in, so it reads each shard's footer in place (`RemoteStore.read_parquet_footer`, a ranged read rather than a download) and hands them to the record — otherwise every downstream stage would be missing from it, indistinguishable from one that never ran. The record also lists a `files` entry — the SHA-256 of every output file, taken from the local file as the publish step uploads it and returned with its key by the DBOS extraction and stage workflows (`RunBoard.file_checksums` reads them back, taking a failed or unfinished workflow's completed steps), so `womblex finalize --dsn` records them and a whole run is auditable without opening each file; a job that may have published a file without a digest (failed, unfinished, or recorded before checksums) is named in `partial`; without a DSN, or for a local `womblex run`, the record carries none and `partial` says so. What cannot be established is named in a `partial` list rather than omitted — a run finalised before the stamps existed produces a record that says so instead of one quietly claiming the run had no stages.

The record also names **the container image the run executed inside** (`image`: whether it was containerised, the reference, the digest). That one is *supplied*, not observed: a digest is content-addressed after the push, so it cannot be baked into the image it names, and a container cannot read its own labels from inside — compose injects `WOMBLEX_IMAGE_REF` from the same value it pins each service to, and the local override empties it because those services build. So it is operator-honest rather than self-verifying, and `partial` says so even when the digest is present, the same way the declared services already describe the environment the record was written in rather than one observed during the run. A tag is never recorded as a digest: it is a pointer that can be moved, so it identifies no bytes.

The chain is walkable in both directions. `store/source_resolver.py` is the return leg: given a run's manifest and access to the corpus, `SourceResolver` takes any sidecar row's `source_hash` back to the file it describes. It resolves by the root-relative path P1 records rather than by an absolute one — so a corpus moved to a new root re-resolves against it unchanged — and then verifies the bytes, which is what separates "the file is where the manifest says" from "the file is the one that was extracted". Every call returns a `Resolution` carrying a status and a reason: a re-saved corpus reports the hash mismatch *with the path it found*, rather than the less useful "not found"; a document moved within the corpus is found by hash instead. Nothing returns silently empty, so a consumer walking a whole run gets an account of the rows that failed rather than a shorter list. The two hash bases are distinguished rather than conflated — a file ingest hashes source bytes, the pre-extracted records ingest hashes record id plus text — so a records row is declined by name with `*.provenance.parquet` given as the back-link that does answer for it, which is a different outcome from a document that is missing. Both resolution paths go through `select_supported` in `cli/_shared.py`, the same rule every entry point ingests by, so a document the resolver returns is one a run could have processed and a corpus it refuses is one a run refuses too — agreement by construction rather than by two walks that happen to match. The full-corpus index it can build is lazy and consulted only when the path has not answered, so resolving a corpus still sitting where the run found it costs one file read per lookup rather than a scan.

`store/checkpoint.py` provides `CheckpointManager` for resumable batch runs. Checkpoints are JSON files recording processed document IDs and batch metadata. On resume, already-processed documents are skipped.

### 11. Verify — Integrity Checks

Three mechanisms. None of them scores quality — each checks that what was written is intact and attributable:

- **Per-batch integrity** — `verify_shard_persistence()` (`store/output.py`) runs *during* a run, after every batch write from `cli/pipeline.py`, checking row counts and sidecar joinability. It is the only one that runs before a run is finished.
- **Directory-level audit** — `womblex verify-shards` (`cli/verify.py`) uses `store/shard_audit.py` (`audit_shard_directory` / `scan_shard_directory`) over a finished shard directory, optionally diffing across runs and, with `--input-dir`, comparing the manifest against the count of documents a run would ingest from the source directory — counted through `discover_files`, the same rule the run used, so the comparison cannot report drift that is only a disagreement about what counts as a document.
- **Source resolution** — `womblex resolve-source` (`cli/verify.py`) uses `store/source_resolver.py` to take a run's rows back out to the corpus and verify the bytes.

There is no fourth. Chunk-level quality annotation is `process/quality.py` and its stage, which is a different concern from the integrity checks above.

### 12. Additional Per-Stage Sidecars

Later stages follow the same `<stage>_shards()` over a shard dir + `womblex <stage> --shards` pattern as chunk/enrich/embed above, each writing its own sidecar(s) joinable on `source_hash`:

- **layout** (`process/layout_stage.py`) — reruns the layout step against the source documents and replaces the batch's `*.layout_regions.parquet`; skips a batch whose sidecar fingerprint already matches the config. Elements, tables and redaction stay as extracted. See [layout.md](layout.md)
- **normalise** (`process/normalise_stage.py`) — text-cleaning transforms, writes `*.normalised_text.parquet`
- **spellfix** (`process/spellfix_stage.py`) — Hunspell-gated OCR character-confusion repair, writes `*.spellfix_text.parquet` + `*.spellfix_corrections.parquet` (audit)
- **quality** (`process/quality_stage.py`) — chunk-quality annotation heuristics, writes `*.chunk_quality.parquet`
- **money** (`process/money_stage.py`) — money-span recognition across narrative/table/sheet loci, writes `*.money_spans.parquet` + `*.money_columns.parquet`
- **link** (`link/stage.py`) — record-linkage against a reference register, writes `*.entity_links.parquet`
- **pii** (`pii/pii_stage.py`) — graph-driven PII masking, writes `*.pii_spans.parquet` + `*.clean_text.parquet`

Distributed (cloud) runs execute the same stage bodies against a declarative `StageContract` per stage (`cloud/stage_contracts.py`), reading/writing an object store instead of local disk. DBOS runs them: `cloud/workflows.py` makes a batch and each stage unit a recorded step, `cloud/worker.py` joins the queues a process can serve, and `cloud/jobs.py` is how dispatchers enqueue and read progress.
