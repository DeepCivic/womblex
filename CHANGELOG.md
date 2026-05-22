# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed — BREAKING
- **Extraction output reshaped to element-stream + typed sidecars.**
  `ExtractionResult` now carries `elements: list[Element]` as the
  canonical structural stream. Per-batch parquet output is split across
  four sibling files: `batch-NNNN.elements.parquet` plus
  `.table_cells.parquet`, `.form_fields.parquet`, `._manifest.parquet`.
  Sidecars are joinable via `(source_hash, parent_elem_order)`. The
  previous one-parquet-per-batch shape (`EXTRACTION_SCHEMA` with nested
  struct lists for tables / forms / images / text_blocks) is removed.
  See `docs/extraction.md` for the canonical reference.
- **Narrative ↔ table interleaving preserved.** DOCX bodies are walked
  in OOXML order so paragraphs and tables emit in their true position;
  PDF per-page outputs are sorted by `(y, x)` so within-page reading
  order survives.
- **Spreadsheets: one ExtractionResult per workbook.** Cells are the
  element grain (`kind='sheet_cell'`); sheets emit a leading
  `sheet_meta` element. The previous one-result-per-row shape is
  removed.
- **Verbatim text at the schema boundary.** `_normalise_text` no longer
  runs in the extraction hot path. Page text and element text are
  emitted verbatim from the producing extractor. Downstream stages may
  still mutate `pages[i].text`; the on-disk parquet is unaffected.
- **Legacy view properties remain.** `ExtractionResult.text_blocks` /
  `.tables` / `.forms` / `.images` are now read-only derived views over
  `elements` so downstream PII / redact / chunk stages continue working
  without change.
- **Sidecar integrity check.** `verify_shard_persistence` now confirms
  every `(source_hash, parent_elem_order)` in `table_cells` /
  `form_fields` resolves to an element with the matching kind. Drift
  raises `ShardVerificationError`.

### Added
- **Per-page extraction plan + orchestrator.** PDFs now route via per-page
  profiling (`ingest/page_profile.py`) and a doc-wide orchestrator
  (`ingest/orchestrator.py`) that dispatches operations page-by-page,
  rather than picking one document-level strategy. Correctly handles
  heterogeneous FOI bundles (e.g. one image-only page in an otherwise-
  native doc) that the previous sample-based detector missed.
- **Spreadsheet-print extractor** (`ingest/spreadsheet_print.py`) for
  native PDFs printed from a spreadsheet (CSV/Excel) — common shape for
  government FOI manifests, schedules, and registers. Extracts row-by-row
  data plus a metadata block (the label-value fields above the first
  data row). Handles 90°-rotated landscape pages, multi-line headers,
  centred header text within wider cells, and frozen headers across
  pages. Triggered by a cheap qualifier (filename hints + table signal)
  before structural vetting. Config-driven via
  `extraction.native.spreadsheet_print` (`metadata_location`,
  `filename_hints`).
- **Form-pair text extractor** (`ingest/forms.py`) — picks up "Field:
  value" form pairs in native PDFs (`_extract_form_pairs_from_text`)
  and OCR'd text (`_extract_form_pairs_from_lines`). Forms-column
  populated for ~94 % of docs in the ACT FOI corpus (was 0 %).
- **`block_type` classifier rewrite.** Native blocks now typed by
  position + typography (footer, header, signature, heading, paragraph)
  instead of length-only fallback. `caption` reserved for image-adjacent
  blocks downstream.
- **Per-image OCR on native pages** with embedded image regions (e.g.
  letterhead text). Generalised from hybrid-only via the existing
  `_ocr_image_regions` helper; recovers text-in-images that previously
  fell through extraction.
- **Multi-strategy table detection** — PyMuPDF `find_tables(strategy=
  "lines")` falls back to `strategy="text"` when the document uses
  whitespace-aligned columnar layout instead of ruled cells.
- **`TableData.context` and `ExtractionResult.document_metadata`** —
  per-table and document-level metadata blocks for spreadsheet-print
  layouts.
- `womblex profile` CLI command for per-column schema inference on
  CSV / XLSX / XLS / Parquet / NDJSON files.

### Changed
- **PDF dispatch flips from doc-level to page-level.** PDF strategy
  classes (`Native*Extractor`, `Scanned*Extractor`, `HybridExtractor`,
  `StructuredExtractor`) are deleted; `get_extractor()` now only
  dispatches non-PDF types (DOCX, spreadsheet, text, image). The
  orchestrator is the canonical PDF path.
- **Source redaction and PII tokens unified on angle brackets.**
  `redact/stage.py` blackout mode emits `<REDACTED>` (was `[REDACTED]`);
  `pii/cleaner.py` emits `<ENTITY_TYPE>` (e.g. `<PERSON>`, `<ADDRESS>`,
  `<EMAIL>`). Aligns with the labelling-packet ground-truth fixtures
  (`_transcript-with-redacted-tags*.txt`). The BPE/SentencePiece
  tokenisation argument that previously motivated square brackets didn't
  hold up — neither bracket style is single-piece in standard pretrained
  tokenisers without explicit special-token registration. Source
  redactions remain page-prefix (not inline per span) — inline-per-span
  requires bbox-to-text mapping not currently routed through to the
  redact stage; tracked as a follow-up.

### Internal
- Moved page-image morphology helpers from `detect.py` into
  `ingest/morphology.py` to keep `detect.py` under the 750-line cap.
- `_normalise_text` regex extended to catch fullwidth Unicode `Page`
  variants in OCR'd footers (`5lｐａge`, `3lＰａｇｅ`).

## [0.1.0] - 2026-04-28

### Added
- Initial public release.
- PDF extraction with document-type routing (native, scanned, hybrid, forms, tables).
- DOCX and spreadsheet (CSV/XLSX/XLS) ingestion.
- Standalone G-NAF and geospatial (SHP) ingest paths.
- Post-extraction redaction stage with configurable insertion points.
- Optional PII cleaning (PERSON detection via regex + cosine-similarity validation).
- semchunk-based chunking with table-aware splitting.
- Optional Isaacus enrichment via the `[isaacus]` extra.
- Local model resolution via `WOMBLEX_MODELS_DIR` for offline / edge deployment.

[Unreleased]: https://github.com/DeepCivic/womblex/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/DeepCivic/womblex/releases/tag/v0.1.0
