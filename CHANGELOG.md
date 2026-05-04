# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `womblex profile` CLI command for per-column schema inference on
  CSV / XLSX / XLS / Parquet / NDJSON files. Reports inferred type,
  nullability, uniqueness, and value ranges. New `womblex.profile` module
  exposes `profile_file()` and `profile_dataframe()` for programmatic use.

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
