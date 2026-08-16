# Project Structure

A file-level map of the Womblex source tree. For the *why* behind the layout
(per-page orchestrator, element-stream schema, per-stage shard commands), see
[`architecture.md`](architecture.md), [`dataflow.md`](dataflow.md), and the
module-responsibility table in [`../CLAUDE.md`](../CLAUDE.md).

```
womblex/
├── configs/           # Dataset-specific configurations
├── docs/              # Architecture docs, ADRs, accuracy reports
├── fixtures/          # Test fixtures (separate repo, see ../THIRD_PARTY_DATA.md)
├── src/womblex/
│   ├── cli/                # CLI subpackage — per-topic modules: pipeline, cloud, redact, link, embed,
│   │                       #   normalise, spellfix, quality, money, pii, ingest, score, profile, verify, ui
│   ├── config.py           # Pydantic config models
│   ├── batch.py            # process_batch() — shared per-batch pipeline body (extract → redact/chunk/pii)
│   ├── operations/         # Independent operations, one module each: extract, redact, chunk, pii, enrich
│   │   ├── models.py       # DocumentResult / BatchResult dataclasses + PreconditionError
│   │   └── persist.py      # write_batch_parquet / write_batch_enrichment
│   ├── score.py            # womblex score subcommand — labels-vs-parquet CER scoring
│   ├── profile/            # womblex profile subcommand — column schema inference
│   ├── ingest/
│   │   ├── detect.py            # Doc-level type classification (non-PDF dispatch + summary type for PDFs)
│   │   ├── page_profile.py      # Per-page PageProfile + cheap qualifiers (e.g. spreadsheet-print)
│   │   ├── orchestrator.py      # Plan-driven PDF extractor — walks per-page profiles, dispatches operations
│   │   ├── elements.py          # Element model + kinds + Cell / FieldEntry / BBox (canonical)
│   │   ├── views.py             # ExtractionResult + legacy view types (TableData / FormField / TextBlock / ImageData) as read-only projections over elements
│   │   ├── extract.py           # extract_text() entry point + page-level primitives (re-exports views)
│   │   ├── forms.py             # Form-pair extraction (AcroForm + spatial + line-based for OCR)
│   │   ├── spreadsheet_print.py # Multi-page table extractor for spreadsheet-printed PDFs
│   │   ├── morphology.py        # Page-image morphology helpers (handwriting / glyph regularity)
│   │   ├── grid_projection.py   # Column-aware text reconstruction (block-aware paragraph emission)
│   │   ├── strategies.py        # Re-export shim — path-based (non-fitz) extractors
│   │   ├── strategies_scanned.py # OCR primitives (_ocr_page, _layout_blocks_and_tables)
│   │   ├── strategies_file.py   # Non-PDF extractors (DOCX, plain text, non-textual)
│   │   ├── interfaces/
│   │   │   └── protocols.py     # Backend protocols (OCRReader, LayoutAnalyzer, Preprocessor)
│   │   ├── paddle_ocr.py        # PaddleOCR wrapper via rapidocr-onnxruntime + YOLOLayoutAnalyzer
│   │   ├── llm_ocr.py           # LLM/VLM OCR backends: Mistral Pixtral Large via AWS Bedrock, + local Ollama
│   │   ├── spreadsheet.py       # CSV/Excel extraction — one ExtractionResult per workbook with cells as elements
│   │   ├── gnaf.py              # G-NAF PSV → Parquet ingest (standalone)
│   │   ├── gnaf_schema.py       # G-NAF table schemas — static column definitions
│   │   ├── abn_bulk.py          # ABN bulk extract XML → Parquet ingest (standalone, streamed)
│   │   ├── geospatial.py        # SHP → GeoParquet ingest (standalone)
│   │   ├── redaction.py         # Backwards-compatible re-export of redact.detector
│   │   ├── heuristics_cv2.py    # OpenCV-based detection heuristics
│   │   └── heuristics_numpy.py  # NumPy-based detection heuristics
│   ├── redact/
│   │   ├── detector.py      # CV2 raster + vector-drawing redacted region detection
│   │   ├── stage.py         # Post-extraction redaction stage (vector-first, raster fallback)
│   │   ├── batch.py         # Batch redaction: annotate_redactions_for_shards, validate_redactions_against_labels
│   │   └── utils.py         # Masking utilities
│   ├── pii/
│   │   ├── cleaner.py       # PII detection (graph spans primary; regex/cosine backstop opt-in) + masking
│   │   ├── pii_stage.py     # pii_shards() over a shard dir — drives `womblex pii --shards`; writes pii_spans + clean_text
│   │   └── stage.py         # In-memory PII helpers for the E2E `run` path
│   ├── process/
│   │   ├── chunker.py       # semchunk integration — chunk_batch engine + element-stream → ChunkInput helpers
│   │   ├── chunk_stage.py   # chunk_shards() over a shard dir — drives `womblex chunk --shards`
│   │   ├── normalise.py     # Pure text-cleaning transforms (normalise_text)
│   │   ├── normalise_stage.py # normalise_shards() — drives `womblex normalise --shards`; writes *.normalised_text.parquet
│   │   ├── spellfix.py      # OCR character-confusion repair transforms
│   │   ├── spellfix_stage.py # spellfix_shards() — drives `womblex spellfix --shards`; writes *.spellfix_text.parquet + corrections
│   │   ├── quality.py       # Chunk-quality annotation heuristics
│   │   ├── quality_stage.py # quality_shards() — drives `womblex quality --shards`; writes *.chunk_quality.parquet
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
│   │   ├── enrich.py        # Isaacus enrichment wrappers
│   │   ├── enrich_stage.py  # enrich_shards() — drives `womblex enrich --shards`; writes *.enrichment_entities.parquet
│   │   ├── enrich_merge.py  # Stitch per-segment results of a split long document back into one
│   │   ├── graph_refresh.py # refresh_graph_edges() — offline mention→chunk edge rebuild after AI chunking
│   │   ├── embed.py         # Thin wrapper over Isaacus embeddings.create (kanon-2-embedder)
│   │   ├── embed_stage.py   # embed_shards() — drives `womblex embed --shards`; writes *.embeddings.parquet
│   │   ├── graph.py         # Entity graph construction
│   │   ├── models.py        # Enrichment data models
│   │   └── query.py         # Load enrichment graph from Parquet for PII masking
│   ├── store/
│   │   ├── output.py        # Parquet output: elements + table_cells + form_fields + manifest + chunks sidecars + integrity checks
│   │   ├── shard_audit.py   # Directory-level shard integrity + chunks-side audit + reconcile-with-checkpoint
│   │   ├── enrichment_output.py  # Enrichment-specific output
│   │   ├── enrichment_doc.py     # *.enrichment_doc.parquet — raw ILGS Document, for AI-chunking reuse
│   │   ├── pii_output.py    # pii_spans + clean_text parquet schemas + IO
│   │   ├── normalise_output.py   # *.normalised_text.parquet schema + IO
│   │   ├── spellfix_output.py    # *.spellfix_text.parquet + *.spellfix_corrections.parquet schemas + IO
│   │   ├── quality_output.py     # *.chunk_quality.parquet schema + IO
│   │   ├── money_output.py  # *.money_spans.parquet (decimal128 values) + *.money_columns.parquet schemas + IO
│   │   ├── provenance_output.py  # *.provenance.parquet sidecar + manifest for pre-extracted-record corpora
│   │   ├── feedback_output.py    # One-file-per-report console feedback records (JSON, not parquet)
│   │   ├── run_manifest.py  # Consolidate per-batch manifests into a run-root manifest.parquet
│   │   ├── register_manifest.py  # Manifest for standalone register ingests (G-NAF/ABN/geospatial)
│   │   ├── remote.py        # fsspec stage-in/stage-out object-storage adapter for distributed runs
│   │   ├── retention.py     # run_id-based retention policy + describe_run() (doc count, stages, timestamps)
│   │   └── checkpoint.py    # Per-stage CheckpointManager
│   ├── cloud/                  # Distributed run support — `womblex-cloud` counterpart to local `womblex run`
│   │   ├── queue.py            # JobQueue — Postgres FOR UPDATE SKIP LOCKED batch queue
│   │   ├── worker.py           # run_worker() — claim/stage/process/publish loop
│   │   ├── stage_contracts.py  # Declarative StageContract per downstream stage (inputs/outputs/scope)
│   │   └── stage_runner.py     # Execute a contract against an object store
│   ├── ui/                     # Console sidecar (`womblex ui`) — FastAPI over pipeline artefacts; reads runs, never writes to one
│   │   ├── app.py              # create_app() — binds one run source for the app's lifetime
│   │   ├── deps.py             # UISettings — local output_root vs store-backed, resolved from args/env
│   │   ├── readers.py          # Thin pyarrow readers, local and store-backed, over the same store/ modules
│   │   └── routes/
│   │       ├── runs.py         # /api/runs — manifest, stage-presence, audit, chunk detail
│   │       └── feedback.py     # POST /api/runs/{run_id}/feedback — the report action (writes a feedback/ sibling)
│   ├── utils/
│   │   ├── metrics.py       # WER/CER accuracy metrics
│   │   ├── tabular_metrics.py # Tabular extraction accuracy (structural fidelity, data integrity)
│   │   ├── models.py        # Local model path resolution (models/ dir, HF snapshot layout)
│   │   ├── checksum.py      # Shared streamed MD5 helper for the standalone register ingests
│   │   ├── isaacus_client.py # Build the Isaacus SDK client (hosted API or private SageMaker)
│   │   ├── token_packer.py  # TokenCounter, pack_by_tokens, split_on_boundaries for token-budgeted API batching
│   │   └── availability.py  # isaacus_available() — gates stages that need the API-only Kanon-2 tokeniser
│   └── verify/
│       └── engine.py        # Two-pass extraction quality verification
└── tests/
```
