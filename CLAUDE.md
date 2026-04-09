# CLAUDE.md
Context for Claude when working on this codebase.
## Project Purpose
Womblex extracts text from Australian government PDF document releases and prepares it for semantic analysis via Isaacus. The project exists because:
1. Government documents are messy (scanned, redacted, mixed formats)
2. Isaacus models need clean, chunked text as input
3. Getting text out is hard; analysis after that is easy
The `raw_documents/` folder contains a curated mix of government document types. This serves as the baseline for testing and refining the extraction → Parquet process.

## Key Design Decisions
### Document type detection drives extraction strategy
The previous Kaggle approach used one-size-fits-all OCR. This failed because:
- Native PDFs don't need OCR
- Forms need layout-aware extraction
- Tables need different handling than prose
- Hybrids need selective OCR
Detection happens first, then routes to appropriate extractor.
### Redaction is a post-extraction concern
Redaction runs as a separate operation after extraction via `redact/stage.py`. The redaction detector misfires on form fields, chart regions, and diagram fills when called inside `_ocr_page()`, suppressing legitimate text. Do not call `pre_ocr_mask` from within extraction strategies.
### Chunking is generic semchunk integration
`process/chunker.py` wraps semchunk with no opinion about tokeniser or chunk size — those are dataset-level config choices in `configs/*.yaml`. The chunker accepts any HuggingFace tokeniser identifier or a callable token counter.
### Config-driven, not hardcoded
Dataset-specific settings live in YAML configs. The codebase doesn't know about specific datasets — that's all in config files under `configs/`.
### Checkpointing for long jobs
Processing 1500+ documents takes hours. Checkpoint after each batch so failures don't require full restart.
## Module Responsibilities
| Module | Does | Doesn't |
|--------|------|---------|
| `ingest/detect.py` | Profile PDFs for extraction routing (samples text/OCR for classification) | Produce final extracted text |
| `ingest/extract.py` | Get text out of PDFs, DOCX, and TXT files | Chunk or analyse |
| `ingest/gnaf.py` | Standalone G-NAF PSV → Parquet ingest (bypasses NLP pipeline) | Run redaction, chunking, PII, or enrichment |
| `ingest/gnaf_schema.py` | Static, versioned column definitions for all G-NAF table types | Parse SQL at runtime |
| `ingest/geospatial.py` | Standalone SHP → GeoParquet ingest (bypasses NLP pipeline) | Run redaction, chunking, PII, or enrichment |
| `ingest/paddle_ocr.py` | Wrap RapidOCR and YOLOv8 layout analysis | Implement extraction strategy logic |
| `redact/detector.py` | Detect and mask redacted regions | Know about document semantics |
| `redact/stage.py` | Run redaction at configurable pipeline points (post_chunk, post_enrichment) | Implement detection logic |
| `pii/cleaner.py` | Detect PERSON candidates via regex; validate with cosine similarity context model; merge enrichment-derived spans | Call Isaacus directly |
| `pii/stage.py` | Run PII cleaning as an isolated pipeline stage (post_extraction, post_chunk, post_enrichment) | Implement detection logic |
| `process/chunker.py` | Split text into chunks | Call Isaacus |
| `analyse/*.py` | Wrap Isaacus API calls; `query.py` loads enrichment graph from Parquet for PII masking | Handle PDFs directly |
| `utils/models.py` | Resolve local model paths before falling back to downloads | Load models (callers do that) |
| `utils/metrics.py` | CER, WER, CER-s (spatial sort), Levenshtein distance | Know about document types or pipeline stages |
| `utils/tabular_metrics.py` | Structural fidelity, data integrity, key column preservation, schema conformance for tabular extraction | Know about specific datasets or file formats |
| `operations.py` | Independent operations (extract, redact, chunk, PII, enrich) | Orchestrate or sequence operations |
## Coding Conventions
### Style
- Python 3.11+
- Type hints everywhere
- Dataclasses for structured data
- Pydantic for config/validation
- Australian spelling in comments and docs
- **750 line hard cap per file** — validate after every file save with `wc -l`; split if exceeded
### Error Handling
- Individual document failures shouldn't stop the batch
- Log errors with document ID for debugging
- Store error status in output for review
### Dependencies
- PyMuPDF (`fitz`) for PDF handling
- rapidocr-onnxruntime for OCR (bundles PaddleOCR v4 ONNX det/rec/cls models, no PaddlePaddle framework)
- ultralytics for YOLOv8 layout analysis (bundled yolov8n.pt in `models/`)
- opencv-python-headless for image processing (binarisation, deskew)
- semchunk for chunking
- isaacus for analysis
- presidio-anonymizer for PII replacement
- sentence-transformers for PII context validation
- No heavyweight ML frameworks in core (models bundled in rapidocr-onnxruntime wheel, loaded lazily)
- Local models in `models/` are resolved automatically by `utils/models.py` — no network access required at runtime
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
### Post-extraction normalisation runs automatically
`extract_text()` applies `_normalise_text()` to every page after extraction. Known artefacts it fixes:
- `'$` → `'s` (broken ToUnicode font map producing `$` after curly apostrophes)
- `http:lL` → `http://` (URL corruption from same font encoding bug)
- Running OCR footers in spaced-character form (e.g. `1 | P a g e`)
If you see new systematic artefacts, add them to `_normalise_text()` in `extract.py`.
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

### Accuracy docs are generated by tests
`docs/accuracy/EXTRACTION.md`, `REDACTION_HANDLING.md`, and `PII_CLEANING.md` are written automatically at the end of `test_fixture_accuracy.py` and `test_womblex_collection_accuracy.py` runs. Do not edit them by hand — run the tests to regenerate.

### Large PDFs can exhaust memory
Process page-by-page, don't load entire document into memory:
```python
for page in doc:
    # Process page
    # Don't accumulate large objects
```
## Testing Approach
### Unit tests
- Detection logic with real benchmark images and programmatic PDFs
- Extraction strategies exercised via real benchmark fixtures
- Chunker output validated with ground-truth text from benchmark annotations
### Integration tests
- Full pipeline on small document set
- Isaacus calls (mocked for CI, real for local validation)
### Test fixtures
All test data comes from real documents in `fixtures/` (FUNSD, IAM-line, DocLayNet, womblex-collection). No synthetic data — see `fixtures/fixtures/README.md` for sample descriptions and ground-truth formats. Real PDF fixtures will be added from the larger document collection as extraction quality improves.

Fixtures live in a separate repository ([womblex-development-fixtures](https://github.com/DeepCivic/womblex-development-fixtures)) and are pulled in as a git submodule at `fixtures/`. After cloning, run:
```bash
git submodule update --init
```
### Running tests
Always use `uv run python -m pytest` (not bare `pytest`) to ensure the venv is active:
```bash
# Ensure fixtures submodule is present
git submodule update --init

# Fast unit tests (slow tests excluded by default via pyproject.toml addopts)
uv run python -m pytest tests/ -v

# Include slow tests (OCR-dependent tests skip cleanly if rapidocr-onnxruntime is not installed)
uv run python -m pytest tests/ -v -m ""

# Full accuracy benchmarks (~3 min, regenerates docs/accuracy/*.md)
uv run python -m pytest tests/test_fixture_accuracy.py tests/test_womblex_collection_accuracy.py -v
```
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
### Adding a new document type
1. Add enum value to `DocumentType`
2. Add detection logic to `detect.py`
3. Create extractor class in the appropriate strategy module (`strategies_native.py`, `strategies_scanned.py`, or `strategies_file.py`)
4. Register in `get_extractor()` in `extract.py` and add to the re-export shim in `strategies.py`
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
3. `operations.py` — independent operations, how they compose
4. `process/chunker.py` — semchunk integration
## Don't
- Add dataset-specific logic to core modules
- Assume documents have text layers
- Skip redaction handling for "clean-looking" documents
- Add heavy ML dependencies (keep extraction lightweight)
- Modify `pyproject.toml` dependencies without human approval
- Create oversized files — stay under 750 lines unless justified
- Add TODOs or FIXMEs — fix issues immediately or document in issues
- Over-engineer — no premature abstractions or "strategy patterns"
- Reject unusual data — warn about it, but continue processing
- Create unnecessary files — edit existing files when possible
- Add excessive docstrings — docstrings are concise, practical and only where needed
- Add quality scoring — we don't understand the data well enough yet
## Do
- Read files before modifying — use Read tool to understand existing code
- Follow existing patterns — check similar files (e.g., other scrapers) before implementing
- Run tests after changes — verify nothing is broken
- Keep code simple — direct implementations over complex abstractions
- Add docstrings — but keep them concise (explain what/why, not how)
- Check files before commit — ensure each commit is aligned to docs/steering and one-off use files don't get merged
- Add type hints to all functions
- Handle individual document failures gracefully
- Keep extraction strategies isolated and swappable
- Log document IDs with all errors
- Write checkpoint after each batch
- Manage dependencies via `pyproject.toml` + `uv lock`; no separate requirements files
