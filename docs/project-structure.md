# Project Structure

A file-level map of the Womblex source tree. For the *why* behind the layout
(per-page orchestrator, element-stream schema, per-stage shard commands), see
[`architecture.md`](architecture.md), [`dataflow.md`](dataflow.md), and the
module-responsibility table in [`../CLAUDE.md`](../CLAUDE.md).

```
womblex/
├── configs/           # Dataset-specific configurations
├── docs/              # Architecture docs, ADRs, accuracy reports, the deployment-image audit, the consumer contract (contract.md), the service API (service-api.md), the egress bundle (egress.md)
├── fixtures/          # Test fixtures (separate repo, see ../THIRD_PARTY_DATA.md)
├── models/            # Locally pre-downloaded models, resolved by utils/models.py (see models.md)
├── sql/womblex_jobs.sql  # Postgres schema for the cloud job queue
├── ui/                # Console SPA frontend (SvelteKit), built into the console image by Dockerfile.ui
├── src/womblex/
│   ├── cli/                # CLI subpackage — per-topic modules
│   │   ├── __init__.py     # main() + ALL_COMMANDS aggregation + dispatch
│   │   ├── __main__.py     # `python -m womblex.cli` → main()
│   │   ├── _shared.py      # Command NamedTuple, setup_logging, select_supported / discover_files, normalise_prefix, check_stage_models
│   │   ├── pipeline.py     # run, extract, chunk, manifest subcommands
│   │   ├── cloud.py        # enqueue / enqueue-stages / worker / jobs / finalize / run-stage / egress subcommands
│   │   ├── serve.py        # serve subcommand (the `/v1` service API)
│   │   ├── api_token.py    # api-token subcommand
│   │   ├── ui.py           # ui subcommand (admin and debugging console)
│   │   ├── redact.py       # redact, annotate-redactions, validate-redactions subcommands
│   │   ├── link.py         # enrich, link, graph-refresh subcommands
│   │   ├── embed.py, normalise.py, spellfix.py, quality.py, money.py, pii.py  # one subcommand each
│   │   ├── ingest.py       # ingest-gnaf, ingest-geo, ingest-abn subcommands
│   │   ├── score.py        # score subcommand
│   │   ├── ground_truth.py # ground-truth subcommand (segment + render a shard dir into baselines + sidecars)
│   │   ├── profile.py      # profile subcommand
│   │   ├── demo.py         # seed-demo subcommand (publish the vendored demo run)
│   │   └── verify.py       # verify-shards, resolve-source subcommands
│   ├── config/             # Pydantic config models (__init__: YAML loader + WomblexConfig; process.py: process-stage models)
│   ├── pipeline_order.py   # PIPELINE_ORDER — what runs when; enabled_downstream_stages(config)
│   ├── batch.py            # process_batch() — shared per-batch pipeline body (extract → optional redaction detection; extraction only)
│   ├── operations/         # Independent operations, one module each: extract, redact, chunk, pii, enrich — callers compose directly
│   │   ├── models.py       # DocumentResult / BatchResult dataclasses + PreconditionError
│   │   └── persist.py      # write_batch_parquet / write_batch_enrichment
│   ├── score.py            # womblex score subcommand — labels-vs-parquet CER scoring
│   ├── profile/            # womblex profile subcommand — column schema inference
│   ├── ingest/
│   │   ├── pdf/                 # The PDF seam — backend-neutral types the extractors are written against
│   │   │   ├── __init__.py      # open_document(path, backend=…) — the one entry point; imports its backend at call time
│   │   │   ├── types.py         # Rect / Word / Span / Line / Block / FoundTable / Drawing / Widget / PageImage + the Page and Document protocols
│   │   │   ├── _fitz.py         # The PyMuPDF backend — the only module in src/ that imports fitz
│   │   │   └── _pdfium_doc.py   # The pypdfium2 backend: document, page geometry, render, images, drawings, widgets (text P6, tables P7)
│   │   ├── detect.py            # Doc-level type classification (non-PDF dispatch + summary type for PDFs)
│   │   ├── page_profile.py      # Per-page PageProfile + cheap qualifiers (e.g. spreadsheet-print)
│   │   ├── orchestrator.py      # Plan-driven PDF extractor — walks per-page profiles, dispatches operations
│   │   ├── elements.py          # Element model + kinds + Cell / FieldEntry / BBox (canonical)
│   │   ├── views.py             # ExtractionResult + legacy view types (TableData / FormField / TextBlock / ImageData) as read-only projections over elements
│   │   ├── extract.py           # extract_text() entry point + page-level primitives (re-exports views)
│   │   ├── forms.py             # Form-pair extraction (AcroForm + spatial + line-based for OCR)
│   │   ├── spreadsheet_print.py # Multi-page table extractor for spreadsheet-printed PDFs
│   │   ├── table_grid.py        # Shared table-grid geometry (spans/columns, y-band binning, row assembly)
│   │   ├── ocr_tables.py        # OCR table feeder: reconstruct_table() over OCR regions, behind precision gates
│   │   ├── records.py           # Pre-extracted text records → element shards that feed the NLP pipeline
│   │   ├── morphology.py        # Page-image morphology helpers (handwriting / glyph regularity), over `Page.render`
│   │   ├── grid_projection.py   # Column-aware text reconstruction (block-aware paragraph emission)
│   │   ├── strategies.py        # Re-export shim — path-based (non-fitz) extractors
│   │   ├── strategies_scanned.py # OCR primitives (_ocr_page, _layout_blocks_and_tables)
│   │   ├── strategies_file.py   # Non-PDF extractors (DOCX, plain text, non-textual)
│   │   ├── markdown.py          # Markdown extractor — headings/lists/tables via markdown-it-py
│   │   ├── interfaces/
│   │   │   └── protocols.py     # Backend protocols (OCRReader, LayoutAnalyzer, Preprocessor)
│   │   ├── paddle_ocr.py        # PaddleOCR wrapper via rapidocr-onnxruntime (det/rec/cls)
│   │   ├── layout_onnx.py       # PPDocLayoutAnalyzer — PP-DocLayout-M layout detection (onnxruntime; registered default `pp-doclayout-m`)
│   │   ├── llm_ocr.py           # LLM/VLM OCR backends: Mistral Pixtral Large via AWS Bedrock, + local Ollama
│   │   ├── spreadsheet.py       # CSV/Excel extraction — one ExtractionResult per workbook with cells as elements
│   │   ├── gnaf.py              # G-NAF PSV → Parquet ingest (standalone)
│   │   ├── gnaf_schema.py       # G-NAF table schemas — static column definitions
│   │   ├── abn_bulk.py          # ABN bulk extract XML → Parquet ingest (standalone, streamed)
│   │   ├── geospatial.py        # SHP → GeoParquet ingest (standalone)
│   │   ├── redaction.py         # Backwards-compatible re-export of redact.detector
│   │   ├── heuristics_cv2.py    # Image heuristics: skew, blur, table grids, contour analysis (OpenCV)
│   │   └── heuristics_numpy.py  # Signal analysis: Otsu threshold bimodality (NumPy)
│   ├── redact/
│   │   ├── detector.py      # CV2 RedactionDetector — raster + vector-drawing redacted region detection and masking
│   │   ├── stage.py         # Post-extraction redaction stage (vector-first, raster fallback): detect_redactions, apply_text_redaction, annotate_*
│   │   ├── batch.py         # Batch redaction: annotate_redactions_for_shards, validate_redactions_against_labels
│   │   └── utils.py         # Low-level pre-OCR masking helper (not used by extractors)
│   ├── pii/
│   │   ├── cleaner.py       # PII detection — PERSON + ADDRESS from graph spans (primary); regex/cosine backstop opt-in (pii.use_regex_backstop, default off) + masking
│   │   ├── pii_stage.py     # pii_shards() over a shard dir — drives `womblex pii --shards`; writes pii_spans + clean_text
│   │   └── stage.py         # In-memory PII helpers for the E2E `run` path
│   ├── process/
│   │   ├── chunker.py       # semchunk integration — chunk_batch engine + element-stream → ChunkInput helpers; repairs split <REDACTED> markers
│   │   ├── chunk_stage.py   # chunk_shards() over a shard dir — drives `womblex chunk --shards`
│   │   ├── normalise.py     # Pure text-cleaning transforms (normalise_text)
│   │   ├── normalise_stage.py # normalise_shards() — drives `womblex normalise --shards`; writes *.normalised_text.parquet
│   │   ├── spellfix.py      # OCR character-confusion repair transforms
│   │   ├── spellfix_stage.py # spellfix_shards() — drives `womblex spellfix --shards`; writes *.spellfix_text.parquet + corrections
│   │   ├── quality.py       # Chunk-quality annotation heuristics
│   │   ├── quality_stage.py # quality_shards() — drives `womblex quality --shards`; writes *.chunk_quality.parquet
│   │   ├── segmenter.py     # segment_elements() — contiguous element ranges under a token budget + page ceiling
│   │   ├── renderer.py      # render_elements() / rendered_order() — segment → reviewer markdown (narrative + GFM tables + label:value forms, interleaved; reuses reassemble_narrative + table_to_markdown); RENDERER_VERSION + baseline_digest
│   │   ├── ground_truth.py  # build_ground_truth() over a shard dir — segment → render → *.gt.md baseline + *.meta.json sidecar (identity from manifest, derivation from footer stamp + config, review unreviewed)
│   │   ├── ground_truth_roundtrip.py  # split_rendered() / apply_corrections() — reverse of render_elements: corrected .gt.md → element-keyed corrections (narrative text, table cells, form fields), block-for-block on NARRATIVE_JOIN
│   │   ├── money.py         # Self-evidencing money recognition (find_money) — patterns, FP blocking, exact Decimals
│   │   ├── money_numbers.py # Number reading, currency symbol/ISO resolution, Australian false-positive blocking
│   │   ├── money_words.py   # Worded amounts (find_worded_amounts, parse_number_words)
│   │   ├── money_vocab.py   # Currency tiers / ISO 4217 / scale / false-positive / header vocabulary tables (data only)
│   │   ├── money_columns.py # Column-evidenced money — classify_column + per-cell parsing
│   │   ├── money_stage.py   # money_shards() — drives `womblex money --shards`; writes *.money_spans.parquet + *.money_columns.parquet
│   │   └── text_overlay.py  # Shared overlay read/merge helper for the offline text layers
│   ├── link/
│   │   ├── matcher.py       # Generic record-linkage: alias / address-exact / token-set name-fuzzy (stdlib difflib)
│   │   ├── reference.py     # Reference-register → normalised ReferenceTable via corpus-declared column roles
│   │   ├── normalise.py     # Minimal name/address normalisation for matching
│   │   └── stage.py         # link_shards() over a shard dir — drives `womblex link --shards`; writes *.entity_links.parquet
│   ├── analyse/
│   │   ├── enrich.py        # Isaacus enrichment wrappers (kanon-2-enricher)
│   │   ├── enrich_stage.py  # enrich_shards() — drives `womblex enrich --shards`; writes *.enrichment_entities.parquet
│   │   ├── enrich_merge.py  # Stitch per-segment results of a split long document back into one
│   │   ├── graph_refresh.py # refresh_graph_edges() — offline mention→chunk edge rebuild after AI chunking
│   │   ├── embed.py         # Thin wrapper over Isaacus embeddings.create (kanon-2-embedder)
│   │   ├── embed_stage.py   # embed_shards() — drives `womblex embed --shards`; writes *.embeddings.parquet
│   │   ├── graph.py         # build_document_graph() — entity graph construction from enrichment results
│   │   ├── models.py        # ILGS data models (Span, Segment, Person, Location, Term, etc.)
│   │   └── query.py         # Load enrichment graph from Parquet for PII masking and internal use
│   ├── store/
│   │   ├── output.py        # Parquet output: elements + table_cells + form_fields + manifest + chunks sidecars + integrity checks
│   │   ├── shard_audit.py   # Directory-level shard integrity + chunks-side audit + reconcile-with-checkpoint
│   │   ├── enrichment_output.py  # Enrichment-specific output
│   │   ├── enrichment_doc.py     # *.enrichment_doc.parquet — raw ILGS Document, for AI-chunking reuse
│   │   ├── pii_output.py    # pii_spans + clean_text parquet schemas + IO
│   │   ├── normalise_output.py   # *.normalised_text.parquet schema + IO
│   │   ├── spellfix_output.py    # *.spellfix_text.parquet + *.spellfix_corrections.parquet schemas + IO
│   │   ├── quality_output.py     # *.chunk_quality.parquet schema + IO
│   │   ├── embed_output.py  # *.embeddings.parquet schema + IO
│   │   ├── entity_links_output.py # *.entity_links.parquet schema + IO (mention grain; doc grain derived on read)
│   │   ├── money_output.py  # *.money_spans.parquet (decimal128 values) + *.money_columns.parquet schemas + IO
│   │   ├── provenance_output.py  # *.provenance.parquet sidecar + manifest for pre-extracted-record corpora
│   │   ├── ground_truth_output.py  # Ground-truth *.meta.json sidecar: schema, validation, unit-id, IO (JSON, not parquet)
│   │   ├── source_provenance.py  # Ingest root + source relpath, and their womblex.* Parquet footer keys
│   │   ├── source_resolver.py    # SourceResolver — resolve a published row's source_hash back to its source file
│   │   ├── egress_output.py      # source_index.parquet schema + IO for the egress bundle (self-contained)
│   │   ├── egress.py             # build_bundle() — export one finished local run's corpus + resolved raw sources to any RemoteStore destination (contract: docs/egress.md; CLI verb `womblex egress`)
│   │   ├── build_info.py    # BuildInfo — package version + source commit, or unavailable with a reason
│   │   ├── run_stamp.py     # RunStamp — run id / version / commit / config digest / stage / preset (config name) / loaded models / slot models (distribution + version), as womblex.* footer keys
│   │   ├── content_digest.py # content_digest(elements): the manifest's determinism handle (kind / order / text / cells / fields / sheet cells / meta)
│   │   ├── contract.py      # womblex.contract_version + womblex.sensitivity (raw / masked / none) footer keys, on every pipeline Parquet
│   │   ├── run_manifest.py  # Consolidate per-batch manifests into a run-root manifest.parquet + the run record in its footer
│   │   ├── register_manifest.py  # Manifest for standalone register ingests (G-NAF/ABN/geospatial)
│   │   ├── demo_corpus.py   # Publish the vendored demo run into a console's run store (`womblex seed-demo`)
│   │   ├── remote.py        # fsspec stage-in/stage-out object-storage adapter for distributed runs
│   │   ├── retention.py     # run_id-based retention policy + describe_run() (doc count, stages, timestamps) + is_safe_run_id (run-root join containment)
│   │   └── checkpoint.py    # Per-stage CheckpointManager — JSON checkpoints for resumable batch runs
│   ├── cloud/                  # Distributed run support — `womblex enqueue` / `enqueue-stages` / `worker` / `jobs` / `finalize` / `run-stage` / `egress`
│   │   ├── queue.py            # JobQueue — Postgres FOR UPDATE SKIP LOCKED batch queue; run `owner` scoping
│   │   ├── dispatch.py         # enqueue_extraction / enqueue_downstream_stages (owner=), downstream_stages gate + guard — shared by the console and the service API
│   │   ├── worker.py           # run_worker() — claim/stage/process/publish loop
│   │   ├── stage_contracts.py  # Declarative StageContract per downstream stage (inputs/outputs/scope)
│   │   └── stage_runner.py     # Execute a contract against an object store
│   ├── api/                    # Service API (`womblex serve`)
│   │   ├── app.py              # create_api_app() — `/v1` health/ready, run submit, list/detail, manifest, files, document text, metrics; owner-scoped
│   │   ├── readers.py          # Document text rows by layer (masked / chunks / elements), read in place with a source_hash filter
│   │   ├── auth.py             # Service-token auth: client registry (WOMBLEX_API_CLIENTS), Caller + scopes, FastAPI dependencies
│   │   └── models.py           # Pydantic response models — the OpenAPI surface
│   ├── ui/                     # Admin and debugging console (`womblex ui`) — FastAPI over pipeline artefacts; reads runs, never writes to one; no auth, loopback by default. Integrations use api/
│   │   ├── app.py              # create_app() — binds one run source for the app's lifetime
│   │   ├── deps.py             # UISettings — local output_root vs store-backed (+ optional queue/presets dirs), resolved from args/env
│   │   ├── readers.py          # Thin pyarrow readers + preset writers, local and store-backed, over the same store/ modules
│   │   ├── dashboard.py        # Queue + per-stage checkpoint views for GET /api/dashboard
│   │   ├── composer.py         # Stage-graph, config JSON Schema, validate + YAML render for the Pipeline Composer
│   │   ├── presets.py          # Named pipeline presets — built-in (DEFAULT-Isaacus) + operator-saved (format: filename/bytes/parse)
│   │   ├── execute.py          # Execution capability + ingest preflight + enqueue-into-queue (the one writable-to-a-run surface)
│   │   ├── resources.py        # Store / queue / Isaacus connection cards + live test actions
│   │   └── routes/
│   │       ├── runs.py         # /api/runs — manifest, stage-presence, audit, chunk detail
│   │       ├── dashboard.py    # GET /api/dashboard — queue state + per-stage progress
│   │       ├── composer.py     # /api/composer — graph, schema, validate, yaml, GET/POST/DELETE presets (presets/ sibling)
│   │       ├── resources.py    # /api/resources — connection cards + test/store, test/queue
│   │       └── execute.py      # /api/execute — status + prefix-scoped ingest preflight + enqueue an extraction run into the queue
│   └── utils/
│       ├── metrics.py       # CER, WER, CER-s accuracy metrics (numpy-accelerated Levenshtein + spatial sort)
│       ├── tabular_metrics.py # Tabular extraction accuracy (structural fidelity, data integrity)
│       ├── model_check.py   # Pre-run model check: off/load/smoke per slot and scope, one minimal request per Isaacus model / service-backed model, remembered for the footer and run record
│       ├── model_registry.py # Named model registry per slot (OCR, layout, PII context, tokeniser, spellfix dictionary): built-in names + `womblex.models.<slot>` entry-point plugins; records which entry each slot actually built, with its distribution and version
│       ├── models.py        # Local model path resolution (WOMBLEX_MODELS_DIR, _models/, models/ + HF snapshot layout) + load record with byte digests
│       ├── checksum.py      # Shared streamed MD5 helper for the standalone register ingests
│       ├── isaacus_client.py # Build the Isaacus SDK client (hosted API or private SageMaker)
│       ├── run_log.py       # capture_batch_log() — tee a batch's womblex log lines to a file beside its shards
│       ├── money_metrics.py # Money recall / precision / scale accuracy against typed transcript tags
│       ├── log_format.py    # JsonFormatter + log_context for --log-format json (run_id/job_id/stage/source_hash)
│       ├── token_packer.py  # TokenCounter, pack_by_tokens, split_on_boundaries for token-budgeted API batching
│       └── availability.py  # isaacus_available() gates API stages (enrich/embed, AI chunking); tokenizer_available() gates offline token chunking on the vendored tokeniser
├── tests/
├── Dockerfile         # Pipeline/worker image (CLI, worker, per-stage commands)
├── Dockerfile.ui      # Console image (adds a Node stage for the SPA)
├── deploy/images.env  # The published image digests a release pinned (written by publish-images.yml)
├── docker-compose.yml # Local + cloud stack; per-service image decisions in docs/deployment-images.md
└── docker-compose.local.override.yml  # Pins the bundled-local stack's connection surface
```
