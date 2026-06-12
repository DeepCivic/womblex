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
│   ├── cli/                # CLI subpackage — per-topic modules (pipeline, redact, ingest, score, profile)
│   ├── config.py           # Pydantic config models
│   ├── operations/         # Independent operations (extract/redact/chunk/pii/enrich), one module each
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
│   │   ├── strategies.py        # Re-export shim — non-PDF extractors + legacy ImageExtractor
│   │   ├── strategies_scanned.py # OCR primitives (_ocr_page, _layout_blocks_and_tables) + ImageExtractor
│   │   ├── strategies_file.py   # Non-PDF extractors (DOCX, plain text, non-textual)
│   │   ├── interfaces/
│   │   │   └── protocols.py     # Backend protocols (OCRReader, LayoutAnalyzer, Preprocessor)
│   │   ├── paddle_ocr.py        # PaddleOCR wrapper via rapidocr-onnxruntime + YOLOLayoutAnalyzer
│   │   ├── llm_ocr.py           # Optional LLM-based OCR backend (vision models via OpenAI-compatible API)
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
│   │   └── text_overlay.py  # Shared overlay read/merge helper for the offline text layers
│   ├── link/
│   │   ├── matcher.py       # Generic record-linkage: alias / address-exact / token-set name-fuzzy (stdlib difflib)
│   │   ├── reference.py     # Reference-register → normalised ReferenceTable via corpus-declared column roles
│   │   ├── normalise.py     # Minimal name/address normalisation for matching
│   │   └── stage.py         # link_shards() over a shard dir — drives `womblex link --shards`; writes *.entity_links.parquet
│   ├── analyse/
│   │   ├── enrich.py        # Isaacus enrichment wrappers
│   │   ├── enrich_stage.py  # enrich_shards() — drives `womblex enrich --shards`; writes *.enrichment_entities.parquet
│   │   ├── embed.py         # Thin wrapper over Isaacus embeddings.create (kanon-2-embedder)
│   │   ├── embed_stage.py   # embed_shards() — drives `womblex embed --shards`; writes *.embeddings.parquet
│   │   ├── graph.py         # Entity graph construction
│   │   ├── models.py        # Enrichment data models
│   │   └── query.py         # Load enrichment graph from Parquet for PII masking
│   ├── store/
│   │   ├── output.py        # Parquet output: elements + table_cells + form_fields + manifest + chunks sidecars + integrity checks
│   │   ├── shard_audit.py   # Directory-level shard integrity + chunks-side audit + reconcile-with-checkpoint
│   │   ├── enrichment_output.py  # Enrichment-specific output
│   │   ├── pii_output.py    # pii_spans + clean_text parquet schemas + IO
│   │   ├── normalise_output.py   # *.normalised_text.parquet schema + IO
│   │   ├── retention.py     # run_id-based retention policy
│   │   └── checkpoint.py    # Per-stage CheckpointManager
│   ├── utils/
│   │   ├── metrics.py       # WER/CER accuracy metrics
│   │   ├── tabular_metrics.py # Tabular extraction accuracy (structural fidelity, data integrity)
│   │   ├── models.py        # Local model path resolution (models/ dir, HF snapshot layout)
│   │   └── checksum.py      # Shared streamed MD5 helper for the standalone register ingests
│   └── verify/
│       └── engine.py        # Two-pass extraction quality verification
└── tests/
```
