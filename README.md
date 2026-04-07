# Womblex

Document extraction pipeline for converting Australian government documents into ML-friendly corpus or collections. Extracts text from PDFs and Word documents (native, scanned, forms, hybrid). Spreadsheets are ingested and produce one result per logical row, ready for per-record semantic analysis.

## Design disclosure
This project is designed for everyone. All design decisions favour air-gapped edge deployment, running on limited resources. This means Womblex doesn't include many of the more robust 'all in one' OCR models.

Mature OCR models are used to compete with Womblex for evaluations and guide development.

## Add-ons/integrations
Optionally outputs are prepared for semantic analysis via [Isaacus](https://isaacus.com/).

## The Problem

Government document releases arrive as a mix of file formats:
- **PDFs** — native (selectable text), scanned (narrative, forms, tables), hybrid, or redacted
- **Word documents** (`.docx`) — paragraphs and embedded tables
- **Spreadsheets** (`.csv`, `.xlsx`, `.xls`) — row-level data, glossaries, key-value lookups, and narrative sheets

One-size-fits-all OCR fails because each format and sub-type needs a different extraction strategy. Womblex detects the document type first, then routes to the right extractor.

## Installation

```bash
pip install womblex
```

With Isaacus enrichment:

```bash
pip install womblex[isaacus]
```

For development:

```bash
git clone --recurse-submodules https://github.com/Team-DeepCivic/Womblex.git
cd Womblex
pip install -e ".[dev]"
```

If you already cloned without `--recurse-submodules`, fetch the test fixtures with:

```bash
git submodule update --init
```

### System Dependencies

No system-level dependencies beyond Python. All extraction backends are pure Python packages:
- **PyMuPDF** (`fitz`) — native PDF text and structure
- **PaddleOCR** (`rapidocr-onnxruntime`) — scanned-page OCR with layout analysis (no Tesseract or PaddlePaddle required)
- **python-docx** — Word document extraction
- **pandas** + **openpyxl** — spreadsheet ingestion (CSV/Excel)

Once you have extraction working, semantic analysis via Isaacus (embeddings, classification, extractive QA) is straightforward.

### Isaacus API Key (optional)

Required only for the enrichment stage (`pip install womblex[isaacus]`). Text extraction works without it.

```bash
cp .env.example .env
# Edit .env and add your key from https://isaacus.com/
```

Or export directly:

```bash
export ISAACUS_API_KEY="your-key-here"
```

## Quick Start

```bash
# Process a document set using a config
womblex run --config configs/example.yaml

# Resume from checkpoint after interruption
womblex run --config configs/example.yaml --resume

# Process individual files (PDF, DOCX, CSV, Excel)
womblex extract document.pdf -o output/
womblex extract report.docx -o output/
womblex extract dataset.xlsx -o output/
```

## How It Works

### 1. Document Type Detection

Before extraction, each file is profiled to determine the appropriate strategy:

**PDFs** — routed by content analysis (text layer coverage, morphological features, table patterns):

| Document Type | Detection Signal | Extraction Strategy |
|---------------|------------------|---------------------|
| Native Prose | Text layer > 100 chars/page | PyMuPDF direct |
| Native + Tables | Text layer + table patterns | PyMuPDF + structure |
| Scanned (machine) | No text layer, regular glyphs | PaddleOCR + YOLO layout |
| Scanned (handwritten) | No text layer, irregular strokes | PaddleOCR (CRNN+Attention) |
| Scanned (mixed) | No text layer, mixed regions | PaddleOCR + contour split |
| Structured | Grid/form layout | PaddleOCR + heuristic tables |
| Hybrid | Partial text layer | Text + PaddleOCR for gaps |
| Image | Photos, diagrams | PaddleOCR, flagged for review |

**Other formats** — routed by file extension:

| Format | Extensions | Extraction Strategy |
|--------|-----------|---------------------|
| Word | `.docx` | python-docx (paragraphs + tables) |
| Spreadsheet | `.csv`, `.xlsx`, `.xls` | pandas per-row or per-sheet |

### 2. Extraction

Each document type routes to an appropriate extractor. `extract_text()` always returns a `list[ExtractionResult]`:

- **PDFs** return a single-element list. PaddleOCR returns per-region confidence scores stored in the document profile. YOLO layout analysis populates `TextBlock.block_type` via COCO class mapping (paragraph, table, figure).
- **DOCX** returns a single-element list with paragraphs and tables.
- **Spreadsheets** return one `ExtractionResult` per logical row (for `data` and `glossary` sheets) or one per sheet (for `narrative` and `key_value` sheets). Sheet type is auto-classified by column count, cell length, and row count.

Each result carries a `document_id` (e.g. `filename:PR-00006191`) used as the primary key downstream.

Post-extraction normalisation runs automatically, fixing known font encoding artefacts (broken apostrophes, corrupted URLs, running OCR footers).

### 3. Redaction

Redaction runs as a post-extraction stage, separate from extraction. This avoids false positives that occur when running redaction detection inside OCR (form fields, chart regions, and diagram fills trigger the detector).

Redacted regions can be replaced with `[REDACTED]` markers (preserving sentence structure) or deleted entirely. The stage is configurable: apply after chunking, after enrichment, or both.

### 4. Chunking

Extracted text is split into semantically meaningful chunks using [semchunk](https://github.com/isaacus-dev/semchunk) with the Kanon tokeniser (default 480 tokens, leaving 32-token headroom for Isaacus 512-token context windows). Tables are converted to markdown and chunked separately, with each chunk tagged as `"narrative"` or `"table"`. `[REDACTED]` markers are preserved across chunk boundaries.

### 5. PII Cleaning

An optional PII cleaning stage strips personal identifiers from chunk text before output or enrichment. Operates on chunks post-chunking as an isolated pipeline stage.

Currently detects: **PERSON** (regex + cosine-similarity context validation via `all-MiniLM-L6-v2`). URL, phone, and email regex support is planned. See `docs/accuracy/PII_CLEANING.md` for measured baseline.

The `all-MiniLM-L6-v2` model is pre-bundled in `models/` and loaded from disk — no network access required at runtime.

### 6. Embeddings and Enrichment

Clean chunks feed into Isaacus models:

- **kanon-2-embedder**: Semantic embeddings for search/retrieval
- **kanon-universal-classifier**: Zero-shot document classification
- **kanon-answer-extractor**: Structured field extraction (dates, names, references)

### Graph construction

Using Isaacus outputs an entity graph can be created for further analysis.


## Configuration

Configs are YAML files defining paths, detection thresholds, and analysis settings:

```yaml
dataset:
  name: my_dataset

paths:
  input_root: ./data/raw/my_dataset
  output_root: ./data/processed/my_dataset
  checkpoint_dir: ./data/checkpoints/my_dataset

detection:
  min_text_coverage: 0.3
  form_signal_threshold: 0.5
  table_signal_threshold: 0.4

extraction:
  ocr:
    engine: paddleocr
    dpi: 200

chunking:
  tokenizer: "isaacus/kanon-2-tokenizer"
  chunk_size: 480
  enabled: true
  chunk_tables: true

processing:
  batch_size: 25
  checkpoint_every: 25
```

See `configs/example.yaml` for a complete example.

## Output

Processing produces Parquet files:

**documents.parquet**
- One row per extraction unit with full text, metadata, tables, forms, and confidence scores

With `womblex[isaacus]` enrichment enabled:

**entities.parquet** — Flat entity mentions for filtering

**graph_edges.parquet** — Relationship edges for graph queries

**enrichment_meta.parquet** — Document-level enrichment metadata

## Project Structure

```
womblex/
├── configs/           # Dataset-specific configurations
├── docs/              # Architecture docs, ADRs, accuracy reports
├── fixtures/          # Git submodule: test fixtures (FUNSD, IAM-line, DocLayNet, womblex-collection)
├── src/womblex/
│   ├── cli.py              # CLI entry point (womblex run / womblex extract)
│   ├── config.py           # Pydantic config models
│   ├── operations.py       # Independent operations (extract, redact, chunk, PII, enrich)
│   ├── ingest/
│   │   ├── detect.py        # Document type detection and profiling
│   │   ├── extract.py       # ExtractionResult schema + strategy dispatch
│   │   ├── strategies.py    # PDF/DOCX extractor implementations
│   │   ├── paddle_ocr.py    # PaddleOCR wrapper via rapidocr-onnxruntime
│   │   ├── spreadsheet.py   # CSV/Excel per-row extraction
│   │   ├── heuristics_cv2.py    # OpenCV-based detection heuristics
│   │   └── heuristics_numpy.py  # NumPy-based detection heuristics
│   ├── redact/
│   │   ├── detector.py      # Redacted region detection
│   │   ├── stage.py         # Post-extraction redaction stage
│   │   └── utils.py         # Masking utilities
│   ├── pii/
│   │   ├── cleaner.py       # PII detection and stripping
│   │   └── stage.py         # PII cleaning pipeline stage
│   ├── process/
│   │   └── chunker.py       # semchunk integration
│   ├── analyse/
│   │   ├── enrich.py        # Isaacus enrichment wrappers
│   │   ├── graph.py         # Entity graph construction
│   │   └── models.py        # Enrichment data models
│   ├── store/
│   │   ├── output.py        # Parquet output writer
│   │   ├── enrichment_output.py  # Enrichment-specific output
│   │   └── checkpoint.py    # Batch checkpoint management
│   ├── utils/
│   │   ├── metrics.py       # WER/CER accuracy metrics
│   │   └── models.py        # Local model path resolution (models/ dir, HF snapshot layout)
│   └── verify/
│       └── engine.py        # Two-pass extraction quality verification
└── tests/
```

## Development

```bash
# Install with dev dependencies
uv sync --all-extras

# Fetch test fixtures (git submodule)
git submodule update --init

# Run unit tests
uv run python -m pytest

# Run OCR and accuracy benchmarks (requires fixture images — takes ~3 min)
uv run python -m pytest tests/test_fixture_accuracy.py tests/test_womblex_collection_accuracy.py -v

# Type checking
uv run mypy src/

# Lint
uv run ruff check src/
```

Accuracy docs (`docs/accuracy/*.md`) are regenerated automatically at the end of each test run — no manual editing needed.

## License

Apache 2.0

## Acknowledgements

- [Isaacus](https://isaacus.com/) for legal AI models
- [semchunk](https://github.com/isaacus-dev/semchunk) for semantic chunking
- [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF handling
- [RapidOCR](https://github.com/RapidAI/RapidOCR) for OCR (bundles PaddleOCR v4 ONNX models, no PaddlePaddle required)
- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8 layout analysis
- [python-docx](https://python-docx.readthedocs.io/) for Word document extraction
- [pandas](https://pandas.pydata.org/) + [openpyxl](https://openpyxl.readthedocs.io/) for spreadsheet ingestion
