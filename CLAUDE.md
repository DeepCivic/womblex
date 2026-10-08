# CLAUDE.md
Context for Claude when working on this codebase.
## Project Purpose
Womblex extracts text from Australian government PDF document releases and prepares it for semantic analysis via Isaacus. The project exists because:
1. Government documents are messy (scanned, redacted, mixed formats)
2. Embedding model should have clean, chunked text as input
3. Typically semantic chunking is better for most things
4. Getting text out is hard; analysis after that is easy
The `raw_documents/` folder contains a curated mix of government document types. This serves as the baseline for testing and refining the extraction → Parquet process.

## Key Design Decisions
### Extraction output is an element stream
Extraction produces an ordered ``elements: list[Element]`` stream on
``ExtractionResult``. Each element is one structural atom (paragraph,
heading, table, form, image, page_break, sheet_meta, sheet_cell).
Tables nest cells and forms nest fields **in memory**; the parquet
writer denormalises them into sidecar files.

On-disk, each batch writes four sibling parquets:
``batch-NNNN.elements.parquet`` plus ``.table_cells.parquet`` /
``.form_fields.parquet`` / ``._manifest.parquet``. Sidecars are
joinable via ``(source_hash, parent_elem_order)``. See
[`docs/extraction.md`](docs/extraction.md) for the canonical reference.

The legacy ``.text_blocks`` / ``.tables`` / ``.forms`` / ``.images``
properties on ``ExtractionResult`` remain as read-only derived views
over ``elements`` so PII / redact / chunk stages keep working
unchanged. The chunker still sees one ``TableData`` per real table or
spreadsheet sheet via this view.

Text is verbatim from the producing extractor — extraction applies no
post-processing. If an extractor produces wrong bytes due to its own
bug (broken ToUnicode font maps etc), the fix belongs in the
extractor, not as a normalisation pass at the schema boundary.

### Per-page profile + plan-driven orchestrator
PDFs route via per-page profiling, not document-level strategy. The
detector profiles every page independently (`ingest/page_profile.py`
→ `PageProfile`); the orchestrator (`ingest/orchestrator.py`) dispatches
operations page-by-page based on those profiles.

Government FOI bundles, papers and scientific document bundles can be
heterogeneous within a single file — cover letter + columnar table + 
form + signed declaration. Document-level routing collapses this to 
one strategy and loses the per-region structure. 
Page-level routing matches the data.

The doc-level `DocumentType` is now a *summary attribute* on the
result, not a switch. PDF strategy classes (`Native*`, `Scanned*`,
`Hybrid`, `Structured`) have been deleted; their bodies are inlined
into the orchestrator's per-page operations
(`_apply_native_page`, `_apply_ocr_page`).

`get_extractor()` only handles the path-based formats (DOCX, SPREADSHEET,
TEXT). Everything `fitz` can open — PDFs **and standalone images** —
routes through `extract_text()` → `extract_pdf_with_plan()`; PyMuPDF
opens an image as a one-page document, so it gets the same per-page OCR
dispatch a scanned PDF page does. There is no separate image extractor.

### Redaction is a post-extraction concern
Redaction runs as a separate operation after extraction via `redact/stage.py`. The redaction detector misfires on form fields, chart regions, and diagram fills when called inside `_ocr_page()`, suppressing legitimate text. Do not call `pre_ocr_mask` from within extraction strategies.

### Thin adapters over mature libraries
For mature, widely-used dependencies (`semchunk`, `rapidocr-onnxruntime`, `presidio-anonymizer`, `sentence-transformers`, the Isaacus SDK), Womblex's role is a thin adapter that handles only the integration concerns the library can't know about — parquet I/O, element-stream projection, source-hash plumbing, corpus-specific wrinkles (e.g. `<REDACTED>` cross-boundary repair). The library's full surface is reached via pass-through (its parameters *are* the feature flags); Womblex defaults track upstream defaults except where the corpus has a measured reason to diverge. Anti-patterns: a Womblex toggle for a library feature, a wrapper that re-implements something the library does natively, hardcoded defaults that shadow upstream without justification. When a library absorbs a concern Womblex previously handled, delete the Womblex code rather than carrying a parallel implementation.

### Chunking is generic semchunk integration
`process/chunker.py` exposes `create_chunker` (wraps `semchunk.chunkerify`) and one entry point `chunk_batch(inputs, chunker, *, overlap, processes, progress)`. Every caller flattens all docs' narratives into one semchunk call and all docs' table markdowns into another, so `processes` and the progress bar parallelise across the whole batch. The chunker accepts any HuggingFace tokeniser identifier or a callable token counter — tokeniser and chunk size are dataset-level config choices in `configs/*.yaml`.

`process/chunk_stage.py` provides `chunk_shards(shard_dir, config)` — the per-stage entry point that walks an existing extraction shard directory, reassembles narrative + tables from each `*.elements.parquet` via `build_chunk_input(source_hash, elements)`, calls `chunk_batch`, and writes a `*.chunks.parquet` sibling. The same `build_chunk_input` powers the in-memory `operations.run_chunking` path, so per-stage and E2E modes feed semchunk identical inputs by construction.

### PII is graph-driven and masked after Isaacus
PII detection consumes the Kanon-2 enrichment graph: PII-typed entities (`natural`→PERSON, `address`→ADDRESS) are the candidates, mapped onto chunks via mention offsets. There is no separate detector and no second enrichment pass — the optional regex/cosine backstop (`pii.use_regex_backstop`, default off) is low-precision and opt-in only; recall is flexed by enrichment *granularity/duration*, not by the backstop. Masking is **terminal**: the `pii` stage writes a masked `*.clean_text.parquet` (`<PERSON_n>`, typed + numbered off the graph entity) *after* enrich + embed and never rewrites the raw chunks that feed Isaacus (the enricher strips `<…>` tags as OCR noise). See [docs/decisions.md](docs/decisions.md).

### Decisions, dead-ends & limitations live in docs/decisions.md
The "why" behind the library — design decisions and rejected alternatives, approaches tried and abandoned (don't re-attempt), library-general limitations, and the deferred backlog — is recorded in [docs/decisions.md](docs/decisions.md). Shipped state is in [CHANGELOG.md](CHANGELOG.md). Read those before re-litigating a settled call or retrying a known dead-end.

### Config-driven, not hardcoded
Dataset-specific settings live in YAML configs. The codebase doesn't know about specific datasets — that's all in config files under `configs/`.

### Checkpointing for long jobs
Processing 1500+ documents takes hours. Checkpoint after each batch so failures don't require full restart.

### Corpus relationship to library
A corpus exists to mature Womblex capability, not host custom code. Corpus-side scripts in `stories/<corpus>/` are appropriate for *configuration + invocation + output formatting* of library functions; any iteration / aggregation / orchestration logic belongs in Womblex. `score.py` and `redact/batch.py` are precedents — promoted from corpus-local scripts to first-class library. Substantive work should ship library-first, with the corpus as its test case.

## Module Responsibilities
| Module | Does | Doesn't |
|--------|------|---------|
| `ingest/detect.py` | Doc-level type classification + non-PDF dispatch (DOCX, spreadsheet, text, image) | Per-page routing or final extracted text |
| `ingest/page_profile.py` | Per-page `PageProfile` (text layer, table signal, form signal, blur, image count); cheap qualifier for spreadsheet-print | Run any extraction operation |
| `ingest/orchestrator.py` | Walk per-page profiles, dispatch native or OCR operations, merge results into one `ExtractionResult` | Hold any extractor logic — calls primitives in `extract.py` and `strategies_scanned.py` |
| `ingest/pdf/__init__.py` + `ingest/pdf/_fitz.py` | `open_document(path, backend=…)` — the seam's one entry point, resolving its backend at call time so no import loads a PDF library; `_fitz.py` adapts PyMuPDF onto the protocols | Hold extraction logic (callers do), or let a backend's shapes escape the adapter |
| `ingest/pdf/types.py` | The PDF seam's backend-neutral vocabulary: `Rect`, `Word` (tuple-shaped for `grid_projection`), `Span`/`Line`/`Block`, `FoundTable`, `Drawing`, `Widget`, `PageImage`, the `Page`/`Document` protocols, and `render_box` (MuPDF's render pixel-box rule, shared by every backend so a render's shape is backend-independent). No third-party imports at runtime | Open a document, import a PDF library, or hold adapter logic (a backend's `_*.py` does) |
| `ingest/pdf/_text.py` | The pdfium text engine: `read_chars` (the only pdfium-touching part), then pure segmentation of `Char`s into lines / blocks / spans / words along each character's writing direction; effective font size (the `Tf` size scaled by the character matrix), characters clipped to the page and no hyphen join, as the locked MuPDF; bold from font weight, ForceBold or name | Reorder columns (content order, as MuPDF); find tables; hold backend state (the page caches its characters) |
| `ingest/pdf/_tables.py` | The pdfium table finder: `find_tables(strategy, bbox, chars, polylines)` → `FoundTable`s via pdfplumber's `TableFinder`; `read_polylines` (the only pdfium-touching part) turns path segments into straight runs, `edges_from_polylines` / `char_dicts` build pdfplumber's inputs lazily | Hold the engine (pdfplumber does), open a document, or reorder text |
| `ingest/pdf/_pdfium_doc.py` + `ingest/pdf/_image.py` | The permissive backend (`backend="pdfium"`): pypdfium2 pages converted to top-left space — geometry, `render` (MuPDF's round-out `render_box`), image objects, filled paths with fill colour, widgets, form XObjects composed to page space; a non-PDF file opens through Pillow (MuPDF's image formats plus WebP/AVIF, one page per frame, MuPDF's page-rect rule) | Be the default (F1) |
| `ingest/elements.py` | Canonical `Element`, `Cell`, `FieldEntry`, `BBox`; kind enumeration | Touch extractor logic or parquet I/O |
| `ingest/extract.py` | Page-level primitives (text, blocks, tables, images), `ExtractionResult` with `elements` stream + derived views, `extract_text()` entry point | Document-level routing (orchestrator does that); post-processing of text |
| `ingest/forms.py` | Form-pair extraction: AcroForm widgets, spatial label-value pairs from `page.get_text("dict")`, line-based pairs from OCR'd text | Know about document types |
| `ingest/spreadsheet_print.py` | Multi-page table extraction for spreadsheet-printed PDFs (FOI manifests, schedules, registers); header/metadata-block capture, rotation handling; grid inference via `table_grid` | Run on every doc — gated by qualifier |
| `ingest/table_grid.py` | Shared table-grid geometry (`Span`/`Column`, y-band binning, data-anchored column clustering, row assembly, centroid row clustering) — one algorithm for both feeders; point-space tolerances as dpi-scalable parameters | Know about PDFs, OCR engines, or parquet |
| `ingest/ocr_tables.py` | OCR table feeder: `reconstruct_table(regions, table_rect, dpi, conf)` → `TableData` or `None` below its precision gates; quad→span reduction, `regions_in_rect` | Detect table regions (layout model does); handle LLM-OCR output (no regions) |
| `ingest/strategies_scanned.py` | OCR primitives (`_ocr_page`, `_layout_blocks_and_tables` — consumes the layout step's regions, never calls an analyser — `_ocr_image_regions`) used by the orchestrator | Doc-level extraction strategies (gone for PDFs *and* images — `ImageExtractor` was deleted as unreachable) |
| `ingest/strategies_file.py` | `DocxExtractor`, `TextExtractor`, `NonTextualExtractor` for non-PDF formats | PDF extraction |
| `ingest/morphology.py` | Page-image morphology helpers — handwriting / glyph regularity / stroke-width variance / OCR confidence sampling | Know about document semantics |
| `ingest/records.py` | Pre-extracted text records → element shards that *feed* the NLP pipeline (`source_hash = sha256(id+text)`, one paragraph element per blank-line block, provenance sidecar). Corpus-agnostic via `RecordFieldMapping` | Extract from files; bypass the pipeline (that's the register ingests) |
| `ingest/gnaf.py` | Standalone G-NAF PSV → Parquet ingest (bypasses NLP pipeline) | Run redaction, chunking, PII, or enrichment |
| `ingest/gnaf_schema.py` | Static, versioned column definitions for all G-NAF table types | Parse SQL at runtime |
| `ingest/abn_bulk.py` | Standalone ABN Lookup bulk extract XML → Parquet ingest (streamed, constant memory; records + names sidecar per file; bypasses NLP pipeline) | Run redaction, chunking, PII, or enrichment; match names to documents (that's `link/`) |
| `ingest/geospatial.py` | Standalone SHP → GeoParquet ingest (bypasses NLP pipeline) | Run redaction, chunking, PII, or enrichment |
| `ingest/paddle_ocr.py` | Wrap RapidOCR (OCR engine registry, reader/engine factories); record the wheel-bundled v4 models when the local v5 directory is absent (they never reach the resolver) | Implement extraction strategy logic; layout analysis (that's `ingest/layout_onnx.py`) |
| `ingest/layout_step.py` | The batch's one layout step: `select_pages` (OCR-routed pages, plus pages without vector redactions when the redaction layout filter is on, or all), one analyser call per page, normalised regions as `LayoutOutcome` on `ExtractionResult`; a failed page is an `error` status, not a failed document | Persist anything (`write_results` does); be read by anything but OCR and redaction (they read `ExtractionResult.layout`; shards read the sidecar) |
| `store/layout_output.py` | `*.layout_regions.parquet` schema + IO, `LayoutFingerprint` (footer keys, equality = same model and settings), `layout_fingerprint(config)`. Self-contained, like `store/pii_output.py` | Run an analyser; decide a page's scope |
| `ingest/layout_onnx.py` | `PPDocLayoutAnalyzer` — PP-DocLayout-M layout region detection on `onnxruntime` (registered default of the `layout` slot); label map from the model's own `inference.yml` | Implement OCR, extraction strategy logic, or redaction detection |
| `redact/detector.py` | Detect and mask redacted regions | Know about document semantics |
| `redact/stage.py` | Run redaction at configurable pipeline points (post_chunk, post_enrichment); `detect_redactions` takes the page's persisted layout regions as exclusion zones — a page with none runs unfiltered, warned once per document and listed on `RedactionReport.unfiltered_pages` (and, under `redact --shards`, in the `*.redactions.parquet` footer) | Run a layout analyser; implement detection logic |
| `pii/cleaner.py` | Detect PII candidates: `detect_spans()` merges enrichment-graph spans (high-confidence) with the opt-in regex/cosine-context detector; `_anonymize()` applies `<ENTITY_TYPE>` tags. Graph is the primary source | Call Isaacus directly |
| `pii/stage.py` | In-memory PII helpers for the E2E `run` path (post_extraction, post_chunk, post_enrichment) | Implement detection logic |
| `pii/pii_stage.py` | `pii_shards()` over a shard dir — read `*.chunks.parquet` + `*.enrichment_entities.parquet`, detect PII per chunk (graph spans on narrative chunks; regex backstop opt-in), write `*.pii_spans.parquet` (audit) + masked `*.clean_text.parquet` (`<PERSON_n>`, terminal — after enrich/embed). Per-stage `CheckpointManager` | Detect before Isaacus; rewrite the raw chunks |
| `store/pii_output.py` | `pii_spans` + `clean_text` parquet schemas + IO (self-contained, like `store/enrichment_output.py`) | Implement detection/masking |
| `process/chunker.py` | `chunk_batch` engine + `create_chunker` + element-stream → ChunkInput projection helpers (`reassemble_narrative`, `collect_tables_from_elements`, `build_chunk_input`); `element_spans` + `chunks_in_document_order` — the narrative offset map and the narrative ↔ table interleave it enables (the two projections are disjoint, so document order survives only as the `elem_order` anchor on table chunks); `_repair_redaction_splits` for cross-boundary `<REDACTED>` markers; `narrative_overrides` + byte-identity guard for AI-chunking Document reuse | Call Isaacus; read/write parquet |
| `process/chunk_stage.py` | `chunk_shards()` over a shard dir — read `*.elements.parquet` + `*.table_cells.parquet` + `*._manifest.parquet`, build ChunkInputs per source_hash, call `chunk_batch`, write `*.chunks.parquet`. With `chunking_model`, rehydrates `*.enrichment_doc.parquet` Documents and passes them as `narrative_overrides`. Per-stage `CheckpointManager` integration | Implement chunking primitives (those are in `process/chunker.py`) |
| `process/normalise.py` | Pure text-cleaning transforms (`normalise_text`): inline-whitespace collapse, `3\|P age` footer-glyph despacing, config-driven letterhead/font-map substitutions. The verbatim-policy downstream cleaning op's core | Read/write parquet; touch extraction (cleanup is downstream, never at the extraction boundary) |
| `process/normalise_stage.py` | `normalise_shards()` over a shard dir — read `*.elements.parquet`, normalise TEXT_KINDS element text, write `*.normalised_text.parquet` (per-element cleaned text layer; passthrough where unchanged). Per-stage `CheckpointManager` | Implement transforms (those are in `process/normalise.py`); re-join redaction-induced paragraph breaks (deferred — cross-element) |
| `store/normalise_output.py` | `*.normalised_text.parquet` schema + IO (self-contained, like `store/pii_output.py`). Cleaning layer — distinct from PII's masking `*.clean_text.parquet` | Implement transforms |
| `process/spellfix.py` | Dictionary-gated OCR character-confusion repair (`repair_text`): out-of-dict trigger + single-edit + unambiguity gates over the bundled en_AU Hunspell dict (`spylls`). Tier A digit→letter homoglyphs (default); Tier B general edit-1 (opt-in). Distinct from the rejected substitution-table dead-end | Read/write parquet; correct valid-word misreads (engine-level concern) |
| `process/spellfix_stage.py` | `spellfix_shards()` over a shard dir — read `*.elements.parquet` (chaining off the normalise overlay when present), repair TEXT_KINDS text, write `*.spellfix_text.parquet` (element-text overlay) + `*.spellfix_corrections.parquet` (audit). Raw elements untouched. Per-stage `CheckpointManager` | Implement the corrector (that's `process/spellfix.py`) |
| `store/spellfix_output.py` | `*.spellfix_text.parquet` (element overlay, same shape as `*.normalised_text.parquet`) + `*.spellfix_corrections.parquet` schemas + IO (self-contained) | Implement repair/detection |
| `process/money.py` | Self-evidencing money recognition (`find_money`): the `docs/money-extraction.md` pattern set (symbol/ISO/word prefixes+suffixes, magnitude, ranges, qualifiers, gated accounting negatives, worded amounts, restatement collapse, opt-in implicit context) over a text string, exact `Decimal` values. Offsets index the string handed in | Read/write parquet; rewrite text; classify columns |
| `process/money_numbers.py` | The layer beneath the patterns: number reading (`parse_number`, `apply_scale`), currency resolution (`resolve_symbol`/`resolve_iso`) and Australian false-positive blocking (`blocked_spans`, `ambiguous_number_spans`, `IntervalIndex`). Re-exported from `money.py`, so import sites are unchanged | Know about `MoneySpan` or patterns |
| `process/money_words.py` | Worded amounts (`find_worded_amounts`, `parse_number_words`): spelled-out numbers incl. `hundred`, scale words and fraction forms → exact `Decimal`; a currency word is required, so a worded number alone is never money | Resolve currency (that's `money.py`); read/write parquet |
| `process/money_vocab.py` | Money vocabulary tables (currency tiers + full ISO 4217, symbols incl. `$US`/`$A` order and `¢`, currency words, scale table, number formats incl. space-grouped thousands, false-positive regexes, header money/veto terms, null markers). Data only | Execute over documents |
| `process/money_columns.py` | Column-evidenced money: `classify_column` (number format → header vocabulary + numeric fraction → never numeric-alone; whole-word vetoes; null markers excluded from the fraction; header currency + scale) and per-cell parsing (brackets = accounting negative inside a money column) | Know about elements or loci (the stage does) |
| `process/layout_stage.py` | `layout_shards()` over a shard dir — per batch, skip when the sidecar's fingerprint equals the config's, else get each PDF/image source (through `SourceResolver` locally, or the runner's `source_fetcher` on a store run), rerun the layout step and replace `*.layout_regions.parquet` (all-or-nothing per batch; a failed document leaves the batch as it was). Elements, tables and redaction untouched. `store/layout_output.layout_fingerprint_statuses` reports `match` / `mismatch` / `unknown` against the elements footer | Re-extract; call OCR; apply a new model to tables or redaction |
| `process/money_stage.py` | `money_shards()` over a shard dir — read `*.elements.parquet` + `*.table_cells.parquet`, annotate three loci (narrative offsets in the `text_source` layer, `table_cell`, `sheet_cell`), write `*.money_spans.parquet` + `*.money_columns.parquet`. Per-stage `CheckpointManager`, per-document isolation | Open source files (input is the extraction parquet); implement the detectors |
| `store/money_output.py` | `*.money_spans.parquet` (one row per amount; `locus` discriminates the anchor group; `value` is `decimal128(38,4)`, exact not float) + `*.money_columns.parquet` (column-verdict audit) schemas + IO (self-contained) | Implement detection or classification |
| `process/segmenter.py` | `segment_elements()` — cut an ordered element stream into contiguous, gapless `Segment`s under a `SegmentationConfig` token budget and page ceiling, via `utils/token_packer.pack_by_tokens`; a single over-budget element is emitted solo and flagged `oversize`. `element_text()` is the per-element projection the budget is measured over (narrative verbatim, table markdown, `label: value` form lines). The ground-truth unit's boundaries | Read/write parquet; derive boundaries from chunks (that would make ground truth a function of the chunker); render a segment for review |
| `process/renderer.py` | `render_elements()` — project a segment's element run into reviewer-facing markdown, and `rendered_order()` — the `order` sequence as it appears in the output, the handle a reading-order check compares against the source. The closed structural set: TEXT_KINDS render verbatim, `table` elements as GFM markdown (reusing `chunker.table_to_markdown`) and `form` elements as one `label: value` line per field, interleaved in element order with one blank line between elements, reusing `chunker.reassemble_narrative` so a reviewed unit and a chunked one share one coordinate space. Images, page breaks and sheet cells are outside the set and contribute nothing | Read/write parquet; reorder, deduplicate or tidy; invent content beyond the closed structural set; compare against ground truth (that is the benchmark's suite) |
| `process/ground_truth_roundtrip.py` | The reverse of the renderer: `split_rendered()` pairs a corrected `.gt.md` block-for-block with the contributing elements on `NARRATIVE_JOIN`, `apply_corrections()` returns element copies carrying the reviewer's corrected text / table cells / form fields — ground truth keyed to element identity, re-renderable via the unchanged `render_elements` in any order or table format. A changed block count is a `RoundTripError`, not a silent re-attribution | Store the corrected form on disk (the benchmark does); re-segment; own the forward projection (that's `renderer`) |
| `process/ground_truth.py` | `build_ground_truth()` over an extraction shard dir — per source_hash: `segment_elements` → `render_elements` per segment → write the `.gt.md` baseline and its `.meta.json` sidecar. Fills identity from the run manifest, derivation from the extraction shard's run-stamp footer (run_id / config_digest / version / preset) plus config (text_source) and the renderer (renderer_version / baseline_digest), review as unreviewed. Per-document isolation | Own the segmentation / rendering / schema (those are `segmenter` / `renderer` / `store/ground_truth_output`); classify a unit's census status or score it (the benchmark's suite) |
| `process/text_overlay.py` | Shared element-text overlay resolver: `load_overlay(base, text_source)` / `apply_overlay(source_hash, elements, overrides)` for the normalise + spellfix layers, selected by one `processing.text_source`; applied before reassembly at both the chunk and enrich sites so they share one coordinate space | Implement the cleaning transforms; pick the text_source (callers/config do) |
| `analyse/*.py` | Wrap Isaacus API calls; `query.py` loads enrichment graph from Parquet for PII masking | Handle PDFs directly |
| `analyse/enrich_stage.py` | `enrich_shards()` over a shard dir — reassemble narrative via `reassemble_narrative`, call Kanon-2 in **token-budgeted batches** (`min(max_texts_per_request, token_budget)` via `utils/token_packer`; over-ceiling docs split + offset-merged), write `*.enrichment_entities.parquet` + `*.enrichment_meta.parquet` + `*.graph_edges.parquet` siblings. With `persist_document`, also writes the raw ILGS Document to `*.enrichment_doc.parquet` for AI-chunking reuse (not for split docs). Per-stage `CheckpointManager`, per-request failure isolation | Implement matching/linking; know about registers |
| `analyse/enrich_merge.py` | Stitch per-segment `EnrichmentResult`s of a split long document into one whose spans index the full text — shift every span by the segment's `start_char`, namespace every entity/segment id by segment index | Call the API; decide when to split (that's `enrich_stage`) |
| `analyse/graph_refresh.py` | `refresh_graph_edges()` over a shard dir — offline, API-free rebuild of mention→chunk edges from the entity + chunk sidecars (both carry char offsets) after AI chunking; populates `enrichment_entities.chunk_index`, rewrites `mentioned_in` edges (keeps hierarchy/citation edges). Idempotent. Per-stage `CheckpointManager` | Relink segments/cross-references (needs the full result); call the API |
| `pipeline_order.py` | `PIPELINE_ORDER` — the one declaration of what runs when; `sort_by_pipeline` / `in_pipeline_order` / `stage_rank` for the consumers that hold their own stage mappings; `DOWNSTREAM_STAGES` + `enabled_downstream_stages(config)` — what a dispatcher may enqueue, config-gated (never `extract` / `pii` / `quality`) | Know how to *run* a stage (contracts and the runner do); gate `run-stage`, which stays free to run any stage in any order |
| `store/output.py` | Extraction output only: the four shard schemas (`ELEMENT`/`TABLE_CELLS`/`FORM_FIELDS`/`MANIFEST`), `write_results` + the role readers, the `*.chunks.parquet` sidecar, and shard integrity verification | Hold a downstream stage's sidecar — those get their own `store/<stage>_output.py` |
| `store/run_manifest.py` | `write_run_manifest(shard_dir)` — consolidate per-batch `*._manifest.parquet` into `<run_root>/manifest.parquet` (the published source_hash → doc_id/filename documents table); written at end of `womblex run`, regenerable via `womblex manifest --shards`. Also `build_run_record` — the run-grain account of what produced the run (`womblex.run_record` footer key): stages observed from the files' own stamps, local models from their digests, the container image from `build_info.image_info()`, services from the declared deployment (an ARN reduced to its endpoint name and region, so no AWS account id reaches the record), and a `partial` list naming what it could not establish — including a directory holding only staged manifests. A caller whose shards are elsewhere passes `footers=` rather than staging the run in | Add columns beyond `MANIFEST_SCHEMA`; declare stages from config (they are observed); know about corpus metadata |
| `store/register_manifest.py` | `write_register_manifest(output_dir)` — index a standalone-register-ingest output dir (`ingest-abn`/`ingest-gnaf`/`ingest-geo`) into a glob-free `manifest.parquet` (one row per output file: source_file, output_file, role, row_count, …), read from each Parquet's footer kv-metadata. Generic `REGISTER_MANIFEST_SCHEMA`, distinct from the NLP `MANIFEST_SCHEMA` | Know about extraction; consolidate per-batch NLP sidecars (that's `run_manifest.py`) |
| `store/enrichment_doc.py` | `*.enrichment_doc.parquet` schema + IO (`source_hash`, `text_source`, `document_json`) — self-contained, SDK-free. Holds the raw ILGS Document the chunk stage reuses for semchunk-4 AI chunking | Import the isaacus SDK; decide reuse (the chunk stage's byte-identity guard does) |
| `store/build_info.py` | Which build is running: `BuildInfo` (package version + source commit) and `resolve_commit()` — the work tree first, a build-time `_build_stamp.py` second, `unavailable:<reason>` where neither answers. Also `ImageInfo` / `image_info()` — which container image, read from the runtime's marker file plus the injected `WOMBLEX_IMAGE_REF`, since a digest cannot be baked into the image it names. Cached per process | Decide the version (that is `__init__`); write parquet or know about runs; discover the digest itself (it is supplied) |
| `store/run_stamp.py` | Which run produced a file: `RunStamp` (run id, version, commit, config digest, stage, preset) and its `womblex.*` footer keys, plus the `preset` (config `dataset.name`) and loaded-model keys — each written only when it has a value, the models read from `utils/models.py` at footer time (a stamp is declared before any model loads); `config_digest` over the *validated* config; `stamp_for_sidecar` / `stamp_from_footers` — the run a downstream sidecar inherits from the extraction shard beside it, since a stage is handed a shard directory and not a run | Decide the run id (the run entry points do); stamp anything that is not a run (`womblex extract <file>`, the register and records ingests) |
| `store/content_digest.py` | `content_digest(elements)` — SHA-256 over a document's ordered elements (kind, order, text, table cells, form fields, spreadsheet cells, alt text, meta); the `content_digest` manifest column, the determinism handle: stable given the same source, version, config and models, whatever the file bytes | Read/write parquet; cover timestamps or other run-varying values |
| `store/contract.py` | Which contract a file conforms to and how sensitive it is: `CONTRACT_VERSION` and the `womblex.contract_version` / `womblex.sensitivity` footer keys (`raw` / `masked` / `none`, by file role; an unknown role is `raw`), attached by `_write_rows` to every pipeline Parquet whether or not its run is stamped | Decide the run (that is `run_stamp.py`); version the standalone register ingests (they carry their own `schema_version`) |
| `store/source_provenance.py` | Where a source document came from: scheme-qualify an ingest root (`file://` / `s3://`), resolve a document's path under it, and build the namespaced `womblex.*` Parquet footer keys. `IngestProvenance` carries root + collection (+ explicit local-path → store-key mapping for the distributed path) | Write parquet; decide the root (config / the enqueue invocation does) |
| `store/source_resolver.py` | The way back: `SourceResolver` resolves a sidecar row's `source_hash` to its source file — by the recorded relpath first, then by content hash over a lazy corpus index — and always returns a `Resolution` (resolved / hash_mismatch / not_found / unsupported_basis) with its hash basis and a reason. Indexes via `select_supported`, so it agrees with what a run ingests by construction | Invent a hash basis (a records-ingest row is declined naming `*.provenance.parquet`); return silently empty; know about the standalone register ingests |
| `store/egress_output.py` | `source_index.parquet` schema (`source_hash`, `doc_id`, `filename`, `ext`, `raw_key`, `status`) + IO for the egress bundle — one row per document, not deduplicated by hash; `status` reuses `SourceResolver`'s vocabulary; `raw_key` is bundle-root-relative (`raw_key_for` → `sources/<source_hash><ext>`), null wherever no file was produced; `ext` keeps its leading dot, as in `MANIFEST_SCHEMA`. Footer carries the exported run's stamp (stage `egress`, passed in explicitly). Self-contained, like `store/pii_output.py` | Resolve sources, copy raw files, or write `egress_manifest.json` (the bundle builder does) |
| `store/egress.py` | `build_bundle(run_root, store, run_id=…)` — the bundle builder: mirror a finished local run's `documents/` shards + `manifest.parquet` under `<dest>/<run_id>/corpus/`, resolve + deduplicate-copy raw sources via `SourceResolver` to `sources/<hash><ext>`, write `source_index.parquet` (stage `egress`) and `egress_manifest.json`. `include_sources=False` skips resolution entirely for a corpus-only export. Local run in, any `RemoteStore` destination out — one code path for a local directory or an object store | Read from an object-store-ingested run (no local corpus to resolve sources against — finalise and stage it locally first); know about `womblex egress` CLI parsing (that's `cli/cloud.py`) |
| `store/provenance_output.py` | `*.provenance.parquet` sidecar (dynamic string schema: `source_hash` + `doc_id` + corpus-declared provenance columns) for pre-extracted corpora, + `write_corpus_manifest` consolidating them into a run-root `manifest.parquet`. Self-contained | Extend `MANIFEST_SCHEMA`; know corpus specifics (the mapping declares them) |
| `link/normalise.py` | Minimal name/address normalisation for matching (casefold, punctuation, street-abbrev, drop state/PO-box tokens) | Address *validation* (G-NAF is a separate, corpus-dependent concern) |
| `link/reference.py` | Bundle-aware reference-register consumption → normalised `ReferenceTable` via corpus-declared column-roles (CSV implemented; multi-file/geospatial seam reserved) | Know about specific registers (corpus declares roles) |
| `link/matcher.py` | Generic record-linkage: `resolve(candidates, reference, …)` → links via alias / address-exact / token-set name-fuzzy (stdlib `difflib`, no rapidfuzz). No rules DSL | Load data or read/write parquet |
| `link/stage.py` | `link_shards()` over a shard dir — read `*.enrichment_entities.parquet`, select candidate kinds, match to register, write `*.entity_links.parquet` (mention grain; doc grain is a derived read view). Per-stage `CheckpointManager` | Implement match primitives (those are in `link/matcher.py`) |
| `store/entity_links_output.py` | `*.entity_links.parquet` schema + IO (mention grain; `read_entity_links(grain='doc')` derives the per-doc view on read, never persisted). Self-contained, like `store/pii_output.py` | Implement matching (that's `link/matcher.py`) |
| `store/embed_output.py` | `*.embeddings.parquet` schema + IO (one row per embedded chunk; `dim` stored explicitly for readers). Self-contained | Call the API (that's `analyse/embed.py`) |
| `analyse/embed.py` | Thin wrapper over Isaacus `embeddings.create` (kanon-2-embedder); batches to the 128-text limit, 429 retry, preserves order. Task-aware (`retrieval/document` vs `retrieval/query`) | Read/write parquet; pick granularity |
| `analyse/embed_stage.py` | `embed_shards()` over a shard dir — embed `*.chunks.parquet` texts, write `*.embeddings.parquet` siblings (vector per chunk). Per-stage `CheckpointManager`, batch-level failure isolation | Implement the API wrapper (that's `analyse/embed.py`) |
| `utils/model_check.py` | Pre-run model check (`check_models`): `off`/`load`/`smoke` per slot, scoped to `extract` or the stage that uses the model; failures collected as `SlotCheck`s (with the resolved `model_variant`), never raised; `check_registered` for a host that cannot load models; the last result is what `store/run_stamp.py` writes to the `womblex.model_check` footer; each Isaacus model an enabled stage calls, and any model exposing `ping()` (the Bedrock / Ollama OCR readers), gets one minimal request at `load` and `smoke` alike. Its whole pass runs under `model_registry.suppress_use_recording` + `models.suppress_recording`, so a model built only to verify it loads never lands in `womblex.slot_models` / `womblex.models` | Decide what a failure does (`womblex run`, the worker and stage preflight do); build the Isaacus client (that's `utils/isaacus_client.py`) |
| `utils/model_registry.py` | Name → model factory per slot (`register`, `resolve`, `known_names`); built-ins register under their existing names and aliases, installed packages through the `womblex.models.<slot>` entry-point group; factory `womblex_traits` (e.g. `markdown`). Unknown name lists the known names; import paths refused. `suppress_use_recording` + `recording_suppressed` let the pre-run model check build a model without `record_use` counting it as used — a caching wrapper (`get_layout_analyzer`, `load_dictionary`) checks `recording_suppressed()` itself to bypass its cache too, or the run's own later build would hit that cache and never call `record_use` at all | Load models; resolve model files (that's `utils/models.py`) |
| `utils/models.py` | Resolve local model paths before falling back to downloads; record what a run resolved (`loaded_models()`, byte digests recomputable from the model files alone) for the footer stamp — a probe opts out with `record=False`, and `record_loaded_path` covers an artefact a library loads from its own wheel. `suppress_recording` is the same opt-out as a context manager, for a caller (the pre-run model check) that cannot thread `record=False` through every factory it calls | Load models (callers do that) |
| `utils/isaacus_client.py` | Build the Isaacus SDK client for whichever deployment the env declares — hosted API (`ISAACUS_API_KEY`) or private SageMaker (`ISAACUS_SAGEMAKER_ENDPOINTS`, parsed to per-model endpoints); `unserved_models` pre-checks a stage's model against the deployed endpoints; `make_ai_chunking_client` for semchunk | Call the API; know about stages (callers pass the model ids) |
| `utils/log_format.py` | `JsonFormatter` (stdlib JSON-lines) and `log_context(run_id, job_id, stage, source_hash)` — the context a worker or local batch attaches to every record it emits; selected by `--log-format json` / `WOMBLEX_LOG_FORMAT` in `cli/_shared.setup_logging` | Add a logging dependency; alter the text format `capture_batch_log` writes for the console |
| `utils/token_packer.py` | `TokenCounter` (cached, offline kanon-2 tokenizer wrapper), `pack_by_tokens` (group items to `min(max_items, token_budget)`, over-budget items solo), `split_on_boundaries` (split an over-ceiling doc on blank lines into offset-tagged segments). Rate limits bind on tokens/request | Call the API; decide the budget (config/caller does) |
| `utils/metrics.py` | CER, WER, CER-s (spatial sort), Levenshtein distance | Know about document types or pipeline stages |
| `utils/tabular_metrics.py` | Structural fidelity, data integrity, key column preservation, schema conformance for tabular extraction | Know about specific datasets or file formats |
| `operations/` | Independent operations (extract, redact, chunk, PII, enrich) — one module each under the package, re-exported from `operations/__init__` so `from womblex.operations import run_*` is unchanged | Orchestrate or sequence operations |
| `batch.py` | `process_batch()` — the single shared per-batch pipeline body (extract → optional redaction *detection* → write one `batch-NNNN.*.parquet` shard). Extraction only: chunk/PII/enrich/embed/money are downstream `run-stage` contracts, never run in-batch. Sequences operations so `cmd_run` (local) and the cloud worker (distributed) execute byte-identically | Run chunking/PII/enrichment/any downstream stage; checkpoint or track cumulative size — those are caller/stage concerns |
| `store/remote.py` | `RemoteStore` — fsspec stage-in/stage-out object-storage adapter (S3/MinIO/GCS/local) for distributed runs; `is_remote_uri`, `storage_options_from_env`, `find` (object keys only, recursive), `read_parquet_footer` (footer + row count in place, for the run record over shards that are never staged in), `write_stream` (an API upload, copied straight to the store). Confines all remote-storage knowledge to one place so `Path`-based stages stay untouched | Thread a filesystem abstraction through the stages; know about the queue |
| `api/auth.py` | Service-token auth for `womblex serve`: `load_registry` / `parse_registry` over the `WOMBLEX_API_CLIENTS` YAML (`client_id`, `token_sha256`, `scopes`), `ClientRegistry.authenticate` (SHA-256, `hmac.compare_digest`, no early exit) → `Caller` (`has`, `owner` — `None` for admin), `generate_token`, lazy-FastAPI `caller_dependency` (401) / `require_scope` (403); `registry=None` is the insecure-dev caller. `fastapi` is imported lazily so the registry and `api-token` work without the `ui` extra | Serve routes; refuse an empty registry at startup (the `serve` command does); store tokens (only hashes live in the registry) |
| `api/app.py` | `create_api_app()` — the `/v1` service API bound to one store + queue (+ ingest): health/ready (open), uploads (multipart, size-capped, supported types only, to `<client_id>/<upload_id>/` of the ingest location), run submit (server-minted run id, non-admin `input_prefix` confined to `<client_id>/`, config gated before any row), list/detail/manifest/files (keys + footer contract keys)/metrics. Runs are queue-known and owner-scoped: another client's run is 404, admin sees all | Authenticate itself (`api/auth.py`); run extraction or stages (it writes queue rows via `cloud/dispatch.py`); refuse an empty registry (`womblex serve` does) |
| `api/readers.py` | `LAYERS` (masked `clean_text` / raw `chunks` / raw `elements`, each with its contract sensitivity) and `document_rows()` — one document's rows in document order, read in place with the `source_hash` predicate pushed into Parquet | Authenticate or gate (the route does, from the layer's sensitivity); resolve runs; mask text |
| `api/models.py` | Pydantic request/response models for the OpenAPI surface | Hold route logic |
| `cloud/queue.py` | `JobQueue` — Postgres `FOR UPDATE SKIP LOCKED` job queue (one `womblex_jobs` table, two row kinds); `enqueue` (extraction batches, idempotent on `(run_id, batch_num)`), `enqueue_stages` (downstream stages, one row per `(run_id, stage)`, positioned past every batch so extraction drains first and stages self-sequence), `claim`, `complete`, `fail` (with retry), `requeue_stale`, `stats`, `runs` (per-run status rollup). A nullable `owner` column scopes a run to the service caller that submitted it: `enqueue*` refuse a run owned by another owner (`RunOwnedError`) and the read views take an `owner=` filter; null means CLI/console, visible unscoped only. The row `status` is the distributed checkpoint | Run extraction or stages; touch object storage; decide which stages a config wants (`pipeline_order` does) |
| `cloud/dispatch.py` | The queue write actions shared by the console and the service API: `enqueue_extraction` (list supported documents under an ingest prefix, batch, write idempotent queue rows) and `enqueue_downstream_stages` (the stages `enabled_downstream_stages(config)` names), both taking `owner=`; `downstream_stages` — the shared config → stage-list gate, callable before any row is written, the `ExecutionDisabled` guard (store + ingest + DSN, in that order) and `execution_status`; the queue connect is bounded by `QUEUE_CONNECT_TIMEOUT`. `ui/execute.py` re-exports it | Run extraction or stages; read the queue; hold console-only reads (`ingest_preflight` stays in `ui/execute.py`) |
| `cloud/worker.py` | `run_worker()` — claim a job and run it: a batch stages its inputs from `RemoteStore`, runs `process_batch` and publishes shards; a stage job hands the run's shard prefix to `stage_runner.run_stage_remote`. Marks the row either way; per-job failure isolation, idle/once/stale-recovery modes | Implement the queue, the pipeline body or the stage runner (`cloud/queue.py` / `batch.py` / `cloud/stage_runner.py`); sequence stages (the claim gate does) |
| `cloud/stage_contracts.py` | Declarative `StageContract` per downstream stage — required inputs, **config-derived** conditional inputs (`chunking_model` → `.enrichment_doc`; `text_source` → the overlay sidecar) and **config-derived** outputs (`write_clean_text`, `persist_document`), `StageScope` (per-batch / whole-run), `MutationMode` (sidecar / in-place), Isaacus need, source-document need (`layout`), checkpoint dirname, preflight. Data only | Touch the store, run anything, or list `manifest` (that's `finalize`) |
| `cloud/stage_runner.py` | Execute a contract against an object store: discover bases from **extraction-role siblings only**, stage one unit in, call the unchanged `*_shards()`, publish **all declared outputs or none**. Skip-by-published-output (sidecar producers only), per-base failure isolation, opt-in checkpoint-dir staging. For a `needs_sources` contract (`layout`) it builds a per-document fetcher over the ingest store (key = the manifest's `source_relpath`, bytes checked against `source_hash`). Idempotent — re-run as batches land | Change any `*_shards()` signature; thread a filesystem abstraction through the stages |
| `cli/cloud.py` | `enqueue` / `enqueue-stages` / `worker` / `jobs` / `finalize` / `run-stage` / `egress` CLI — distributed counterpart to `womblex run` over a shared `--store` object-store URI + Postgres `--dsn`. `finalize` consolidates a distributed run's shard manifests into `<run>/manifest.parquet` in the store (the explicit end-step `cmd_run` does locally); `run-stage` generalises that shape to the downstream per-batch stages (`--store`/`--run-id`/`--output-prefix`, mutually exclusive with `--shards`); `enqueue-stages` queues that same sequence for the workers instead, ordered by `pipeline_order` and config-gated (never `pii` / `quality`); `egress <run> --to <dest>` is the CLI half of `store/egress.build_bundle` — a *local* run root in, any `RemoteStore` destination out (see `docs/egress.md`) | Implement queue/worker/storage/contract logic (those are `cloud/` + `store/remote.py`); run a stage in-process (`enqueue-stages` writes rows only); implement bundle-building logic (that's `store/egress.py`) |

## Coding Conventions
### Style
- Python 3.11+
- Type hints everywhere
- Dataclasses for structured data
- Pydantic for config/validation
- Australian spelling in comments and docs
- **750 line hard cap per file** — validate after every file save with `wc -l`; split if exceeded
- **500 line hard cap per merge** — see [Change size](#change-size)

### Change size
A merge lands at most **500 changed lines** (added + removed) against the
branch it merges into. Check before opening or updating a PR:
```bash
git diff --stat $(git merge-base HEAD origin/main)..HEAD
```
Generated and vendored files don't count toward the cap — `uv.lock`,
`docs/accuracy/*.md` (written by the test suite), and anything under
`fixtures/`. Everything else does, including tests and docs.

Over the cap means the change is doing more than one thing. Split it into
sequential merges that each stand alone: land the mechanical part
(renames, moves, signature threading) first, then the behaviour change on
top. Don't split a change so the halves leave the tree broken or the
suite red — each merge must pass tests on its own. If a single coherent
change genuinely can't fit (a schema migration touching every stage, a
library-wide rename), say so in the PR body with the reason and get human
approval rather than quietly exceeding it.

### Error Handling
- Individual document failures shouldn't stop the batch
- Log errors with document ID for debugging
- Store error status in output for review

### Dependencies
- PyMuPDF (`fitz`) for PDF handling
- rapidocr-onnxruntime for OCR (runs PaddleOCR ONNX det/rec/cls models, no PaddlePaddle framework: bundled v5 under `_models/paddleocr-v5` when present, else the wheel's v4)
- boto3 (core dependency) for the `mistral-ocr` engine — Mistral Pixtral Large via AWS Bedrock (Converse API) — and for the `isaacus-sagemaker` HTTP client's SigV4 signing. Imported lazily at its use sites (`ingest/llm_ocr.py:_ensure_client` → `boto3.client("bedrock-runtime")`; `utils/isaacus_client.py` for the SageMaker session); nothing on the *default* extraction path touches it, which is why the whole suite runs without AWS credentials and only the VLM benchmark / live SageMaker paths skip
- **Isaacus SDK (`isaacus`, `isaacus-sagemaker`) and boto3 are core, not extras.** Every real deployment uses enrichment/embeddings and (often) hosted OCR or SageMaker; they are tiny next to the vision/ML stack, and gating them behind extras only produced misconfiguration (a missing SDK surfaced as "no API key"). They stay dormant until configured — no key / no `ISAACUS_SAGEMAKER_ENDPOINTS` / no `mistral-ocr` engine means nothing calls out. Remaining extras are deployment-shaped: `[local]` (empty — the base install), `[cloud]` (empty — object-storage staging via fsspec + s3fs and the psycopg3 job queue are now core deps, so `s3://` and `--dsn` work on any install; the extra is retained only as a self-documenting marker and so existing `pip install womblex[cloud]` invocations keep resolving), `[ui]` (fastapi + uvicorn), `[dev]`
- onnxruntime for PP-DocLayout-M layout analysis (`ingest/layout_onnx.py`), as well as the rapidocr-onnxruntime OCR models above
- opencv-python-headless for image processing (binarisation, deskew)
- semchunk for chunking
- isaacus for analysis
- presidio-anonymizer for PII replacement
- sentence-transformers for PII context validation
- spylls (pure-Python Hunspell) for the `spellfix` OCR-repair op; bundled en_AU dict under `_models/en_AU`
- No heavyweight ML frameworks in core (models bundled in rapidocr-onnxruntime wheel, loaded lazily)
- Local models in `_models/` (bundled) and `models/` are resolved automatically by `utils/models.py` — no network access required at runtime. Every model is outlined in `docs/models.md`

## Common Pitfalls
### PyMuPDF import
```python
import fitz  # Not `import pymupdf`
```
### semchunk tokeniser loading
Pass a HuggingFace identifier or a callable to `create_chunker`:
```python
chunker = create_chunker("some-org/some-tokenizer", chunk_size=512)
# or with a callable for tests:
chunker = create_chunker(lambda text: len(text.split()), chunk_size=50)
```
### Isaacus task types matter
Embeddings need different task types for queries vs documents:
```python
# For documents being indexed
client.embeddings.create(..., task="retrieval/document")
# For search queries
client.embeddings.create(..., task="retrieval/query")
```
### Native PDF text extraction needs dehyphenation
Always pass `TEXT_DEHYPHENATE` when extracting from native text layers to avoid split words across line breaks:
```python
text = page.get_text("text", flags=fitz.TEXT_DEHYPHENATE)
```
### Text policy at the extraction boundary is verbatim
`_normalise_text` no longer runs in the extraction hot path. Whatever the producing extractor (native text layer, PaddleOCR, DOCX, spreadsheet-print, …) emits is what lands on the element's `text` field, and the parquet writer serialises `elements` — so on-disk content stays extraction-time verbatim. PII and redaction stages may still rewrite `pages[i].text` in place for their own internal use; the parquet is unaffected.

**Chunking reads `elements`, not `pages[i].text`** (as of I2, 2026-05-27). `chunk_batch` consumes `ChunkInput`s built from the element stream via `reassemble_narrative` + `collect_tables_from_elements`. In-memory `pages[i].text` mutations from PII / redact-blackout no longer flow to chunks under `womblex run`; downstream stages that want post-rewrite text will consume the `*.clean_text.parquet` sidecar (P1, not yet written).

If an extractor is producing wrong bytes due to its own bug (broken ToUnicode font maps producing `$` for `'s`, URL corruption like `http:lL`, spaced-character OCR footers), the fix belongs in the extractor itself, not as a post-extraction normalisation pass. Systematic post-extraction cleanup belongs to a downstream cleaning stage that rewrites element text. See `docs/extraction.md`.

### Local model resolution
`utils/models.py` is the single source of truth for finding pre-downloaded models. Always use `resolve_local_model_path(name)` rather than constructing paths manually:
```python
from womblex.utils.models import resolve_local_model_path
model_path = resolve_local_model_path("all-MiniLM-L6-v2")
# Returns Path if found locally, falls back to the string "all-MiniLM-L6-v2"
```
Override with `WOMBLEX_MODELS_DIR` if the `models/` directory is not a sibling of `src/`.

### PII regex must not cross newlines
`_TITLE_CASE_RE` and `_HONORIFIC_RE` in `pii/cleaner.py` use `[^\S\n]+` (non-newline whitespace) as the word-boundary separator. Never change this to `\s+` — that allows the regex to match multi-line spans (e.g. "Janine Fairburn \nAssistant Director") which dilutes cosine similarity scores and causes false negatives.

### PII context similarity threshold
Default 0.35, calibrated on Australian government regulatory documents where vocabulary is uniformly formal. Raising the threshold above 0.5 causes false negatives; the typical cosine score for a real PERSON span in this corpus is 0.35–0.45.

### Accuracy docs are generated by the benchmark's suites, not by tests here
Every file under `docs/accuracy/` is written by a suite in the
**womblex-benchmark** repository (`accuracy/`) and published here through
`accuracy/_paths.py`. Do not edit them by hand, and do not add a suite here that
writes one — run the benchmark's suites to regenerate. Both directions are
checked: `tests/test_repo_boundaries.py` here fails a library test that writes
one of these reports, and `accuracy/test_report_producers.py` there fails a
report in this directory that names no producing suite.

Most of them are scores against womblex-collection ground truth, which is why
the suites live beside it. `CHUNKING.md` is the exception and says so in its own
opening lines: no chunking ground truth exists, so its suite records observed
behaviour and gates the invariants that hold without a reference. A report is
still generated either way — that is what the directory means.

### Large PDFs can exhaust memory
Process page-by-page, don't load entire document into memory:
```python
for page in doc:
    # Process page
    # Don't accumulate large objects
```
## Testing Approach
### Unit tests
- Detection logic with synthetic scans and programmatic PDFs
- Extraction strategies exercised via the synthetic fixture set
- Chunker output validated with the known text of the synthetic scans
### Integration tests
- Full pipeline on small document set
- Isaacus calls (mocked for CI, real for local validation)
### Test fixtures
Every file a test reads comes from the synthetic set in `fixtures/synthetic/`: invented Australian-animal documents in the shapes real government releases take (a budget DOCX, redacted and scanned PDFs, a printed-spreadsheet manifest, register CSV and statistics XLSX, scanned forms, lines and pages each with a `.gt.txt`). `fixtures/synthetic/generate.py` writes them deterministically and they are committed, so source hashes and the `test_default_digest` pins are stable; tests reach them through `tests/_synthetic.py`. A shape the set lacks is added to the generator, never as a real document. See [fixtures/synthetic/README.md](fixtures/synthetic/README.md).

No third-party data is in this repository. Real documents and the public datasets live in womblex-benchmark; the one module here that scores against them (`test_bench_ocr_accuracy.py`, marked `benchmark`) is maintainer-only and reads a local copy at `fixtures/fixtures/` (git-ignored). See [THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md).
### Running tests
`pytest` lives in the `[dev]` extra, not the base deps, so install it first, then
run via `uv run` (not bare `pytest`) to keep the project venv active:
```bash
# One-time. `--extra ui` is not optional for a full run: without fastapi,
# `test_demo_corpus.py` and the UI tests fail *collection* rather than skipping,
# which aborts the whole run. `--extra api` brings python-multipart, without
# which every `test_api_app.py` test errors building the upload route.
uv sync --extra dev --extra ui --extra api

# What CI runs: everything except the maintainer-only `benchmark` module.
uv run python -m pytest tests/ -v -m "not benchmark"

# Fast subset — also skip the OCR (`slow`) tests:
uv run python -m pytest tests/ -v -m "not slow and not benchmark"
```

CI gates on three steps in this order, and **the type check is one of them** — a
green suite is not a green build:
```bash
uv run ruff check src/ tests/
uv run mypy src/          # src/ only; tests are not type-checked
uv run python -m pytest tests/ -v -m "not benchmark"   # with a Postgres service for the queue tests
```

Accuracy benchmarks are **not** in this repository. They live in
womblex-benchmark and regenerate `docs/accuracy/*.md` here:

```bash
# from the womblex-benchmark checkout, with this project's venv on PATH:
uv run python -m pytest accuracy/ -v
```

### Expected conditional skips
The skip count depends on credentials and services, never on fixtures — none
are on broken code. On a bare checkout (no Isaacus key, no Postgres DSN)
`pytest tests/ -m "not benchmark"` reports ~32 skips; CI, which runs a Postgres
service, ~12.
Run with `-rs` to see live reasons. The recurring ones:

- **20 — job queue** (`test_cloud.py`): need a real Postgres
  (`WOMBLEX_DB_DSN` / `DATABASE_URL`); CI provides one.
- **geospatial** (`test_geospatial.py`): needs `geopandas` / `pyogrio`, which
  are not declared in `pyproject.toml` (no extra installs them); install them
  separately to run these tests.
- **isaacus SDK** (`test_enrich.py` / `test_graph.py` / `test_query.py` /
  `test_enrichment_output.py`): module-level `importorskip("isaacus")` — the SDK
  is a core dependency now, so these run (mocked clients); the guard only skips
  on a broken install where `isaacus` cannot be imported.
- **isaacus API key** (`test_embed_stage.py` / `test_enrich_stage.py` /
  `test_pipeline.py::TestAiChunkingLive`): the `isaacus_client` fixture skips
  without `ISAACUS_API_KEY` (these make real Kanon-2 calls — embed, enrich, and
  AI chunking — when a key is present).
- **rapidocr** OCR paths (`test_fixtures.py`): `importorskip("rapidocr_onnxruntime")`.
- **benchmark** (`test_bench_ocr_accuracy.py`, maintainer-only, deselected by
  `-m "not benchmark"`): skips without the benchmark fixtures at
  `fixtures/fixtures/`; its `mistral-ocr` cases also need AWS Bedrock
  credentials with Pixtral Large access.

Skips don't fail the build.

## Analysing Accuracy and Pipeline Performance
When reviewing accuracy results or recommending improvements, use a systematic component-level analysis rather than jumping to isolated fixes. Walk through each layer of the pipeline and ask:

1. **Classification** — Is the document type correctly identified? Are downstream stages tailored to it? Are any `DocumentType` values unreachable?
2. **Preprocessing** — Is preprocessing applied conditionally or universally? Does it help or harm based on document source (scanned vs digital)?
3. **Layout detection** — What is the per-class detection F1? Are failures from low recall, over-segmentation, or misalignment?
4. **Reading order** — Are metrics evaluating OCR fairly (spatial sorting for CER)? Is reconstructed reading order useful or misleading for downstream tasks?
5. **End-to-end task success** — Does lower CER actually improve field extraction, classification, or search downstream?
6. **Component interdependence** — Would improving one stage (e.g. classification) eliminate problems in another (e.g. preprocessing)?
7. **Per-fixture failure mode** — For each test fixture, trace where the pipeline first went off-track: classification, preprocessing, layout, or OCR?
8. **Edge cases** — Are we investing in capabilities (e.g. handwriting) that the OCR engine architecturally can't support?
9. **Adaptive design** — Can early signals enable conditional branches? Is there telemetry to trace which path each document takes?
10. **Metric integrity** — Are CER/WER reported with clear context? Are improvements real, or do they mask regressions?

Recommendations should be ordered by impact-to-effort ratio. See `docs/accuracy/` for current benchmark numbers and `docs/steering.md` for the priority list.

## When Modifying
### Adding a new PDF shape (sub-mode of an existing DocumentType)
For new shapes that fit within the existing native/OCR dispatch:
1. Add detection signals to `PageProfile` if needed (and computation in
   `profile_pages()`)
2. Add a cheap qualifier in `page_profile.py` (re-uses existing fields
   wherever possible — see `qualify_for_spreadsheet_print` for the
   pattern)
3. Add the extractor primitive as a self-contained module
   (`ingest/<shape>.py`) returning the appropriate output type
4. Wire into `orchestrator.extract_with_plan()` behind the qualifier so
   it only runs on candidates
5. Add config under `extraction.native` (or `extraction.ocr`) with a
   nested `BaseModel` in `config/__init__.py`; thread through `operations.py`
   into `extract_text()` → `extract_pdf_with_plan()`

### Adding a new non-PDF document type
Only for formats `fitz` cannot open — anything it can (images included)
belongs on the orchestrator path, not here.
1. Add enum value to `DocumentType`
2. Add detection logic to `detect.py`
3. Create extractor class in `strategies_file.py` (or a new file-based
   module)
4. Register in `get_extractor()` in `extract.py`, add the type to
   `extract_text()`'s path-based guard, and add to the `strategies.py`
   re-export shim
### Adding a new Isaacus capability
1. Add wrapper in `analyse/`
2. Add config section
3. Wire into pipeline
4. The Isaacus SDK does the heavy lifting — keep wrappers thin
### Adding a new dataset
1. Create new config in `configs/`
2. Add manifest parser if index format differs
3. Adjust hypotheses for classification if needed
4. No code changes required if document types are already supported
## Files to Understand First
1. `configs/example.yaml` — see what's configurable
2. `ingest/detect.py` — document type detection logic
3. `operations/` — independent operations (one module each), how they compose
4. `process/chunker.py` — semchunk integration (`chunk_batch` engine, element-stream → ChunkInput helpers)
5. `process/chunk_stage.py` — per-stage `chunk_shards()` over a shard directory; consumed by `womblex chunk --shards`
## Don't
- Add dataset-specific logic to core modules
- Assume documents have text layers
- Skip redaction handling for "clean-looking" documents
- Add heavy ML dependencies (keep extraction lightweight)
- Modify `pyproject.toml` dependencies without human approval
- Create oversized files — stay under 750 lines unless justified
- Land oversized merges — stay under 500 changed lines; split into sequential merges instead
- Add TODOs or FIXMEs — fix issues immediately or document in issues
- Cite line references (`file.py:123`, `:55-79`) in reviews, plans, commits, or docs without approval — line numbers drift and go stale the moment code moves. Refer to code by file name only. Ask first if a line reference is genuinely needed.
- Use the section-sign character "§" anywhere in this repository — never write it, and delete any occurrence you discover in prose (comments, docstrings, docs, tests). Refer to sections by name or with the word "section" instead. The sole exception is vendored model data where "§" is genuine content (e.g. `src/womblex/_models/kanon-2-tokenizer/{vocab.json,merges.txt}`) — leave those untouched.
- Over-engineer — no premature abstractions or "strategy patterns"
- Reject unusual data — warn about it, but continue processing
- Create unnecessary files — edit existing files when possible
- Add excessive docstrings — docstrings are concise, practical and only where needed
- Add quality scoring — we don't understand the data well enough yet
- Add a test here that scores library output against womblex-collection ground truth, or that writes a `docs/accuracy/` report. That suite belongs in womblex-benchmark, beside the ground truth it scores against; this repository receives the report. Reading the synthetic fixtures as ordinary test input is fine — publishing a report from here is not
## Do
- Read files before modifying — use Read tool to understand existing code
- Follow existing patterns — check similar files (e.g., other scrapers) before implementing
- Run tests after changes — verify nothing is broken
- Keep code simple — direct implementations over complex abstractions
- Add docstrings — but keep them concise (explain what/why, not how)
- Check files before commit — ensure each commit is aligned to docs/steering and one-off use files don't get merged
- Update the technical docs in the same PR as the code — a change that adds, moves or retires a module updates `docs/architecture.md` and `docs/project-structure.md` beside it, and a change that satisfies a written requirement retires that requirement where it is recorded. Doc edits count against the 500-line merge cap like any other line, so keep them proportionate; where the requirement lives in womblex-benchmark, raise the two merges as a pair
- Add type hints to all functions
- Handle individual document failures gracefully
- Keep extraction strategies isolated and swappable
- Log document IDs with all errors
- Write checkpoint after each batch
- Manage dependencies via `pyproject.toml` + `uv lock`; no separate requirements files
- `uv.lock` carries **boto3 / s3transfer** as core dependencies (`boto3>=1.34`) — the AWS Bedrock client behind the `mistral-ocr` OCR engine and the SigV4 signing behind the `isaacus-sagemaker` SageMaker client. Installed by a default `uv sync`. (They were previously behind the `[bedrock]` extra; promoted to core so the SDK is present wherever a key/endpoint is configured.)
- If `uv sync` rewrites `uv.lock`, that is a real dependency change, not noise — keep it out of unrelated commits (`git checkout -- uv.lock`) and land it as its own dependency-scoped change with human approval. `uv lock --check` tells you whether the lock and `pyproject.toml` agree
- Never bypass the commit hook with `--no-verify`. If a scan fires, fix it or record why it is safe at the site — `# nosemgrep: <rule-id> -- <reason>` or `# pragma: allowlist secret`. Never widen an exclusion or drop a rule to get a green run. The local rulesets are in `.semgrep/rules/` and each documents its known false-positive classes in its header
- Verify mechanism claims against code or measurement before writing docs — `grep`/`Read` the file or run a probe, attach the evidence. Inferred descriptions without grounding tend to be wrong and need correcting later.
