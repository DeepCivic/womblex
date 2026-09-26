# Models

Every model Womblex loads or calls, what it does, and where it comes from.
Local models are resolved through `utils/models.py`; hosted models are called
only when their stage is enabled and a key, endpoint or credential is present.

## Local models

Resolved per artefact from `WOMBLEX_MODELS_DIR`, then `src/womblex/_models/`
(bundled in the wheel), then `models/` (repo-root, editable installs). Every
successful resolution is recorded with a content digest and written into each
Parquet footer, so a run's record names the models it actually loaded.

| Model | Type | Used by | What it does |
|---|---|---|---|
| `paddleocr-v5/` (`ppocrv5-mobile-det`, `-rec`, `-cls` ONNX + dict) | PaddleOCR v5 mobile, ONNX | `ingest/paddle_ocr.py` (`PaddleOCRReader`) | Text detection, recognition and orientation on scanned pages and images. Run by `rapidocr-onnxruntime`; no PaddlePaddle framework. Preferred when all four files are present |
| `rapidocr-bundled-v4` | PaddleOCR v4, ONNX (inside the `rapidocr-onnxruntime` wheel) | `ingest/paddle_ocr.py` | Fallback OCR models when `paddleocr-v5/` is absent or incomplete. Loaded by the library itself; recorded via `record_loaded_path` |
| `yolo11n_doc_layout.pt` | YOLO11 nano, DocLayNet-finetuned, 11 classes | `ingest/paddle_ocr.py` (`YOLOLayoutAnalyzer`), via `ultralytics` | Layout regions on OCR'd pages. Primary layout model; inference at 832 px |
| `yolov8n.pt` | YOLOv8 nano, COCO-pretrained | `ingest/paddle_ocr.py` | Layout fallback only when the DocLayNet checkpoint is missing. No document classes; keeps the code path working, not useful predictions. Inference at 640 px |
| `kanon-2-tokenizer/` | Kanon-2 tokeniser (Hugging Face `isaacus/kanon-2-tokenizer`) | `process/chunker.py`, `utils/token_packer.py` | Local token counting: semchunk chunk sizes, and token-budgeted packing of enrichment requests and ground-truth segments. No API call |
| `all-MiniLM-L6-v2/` | Sentence Transformer embedding model | `pii/cleaner.py` | Cosine-context check on PERSON candidates in the regex backstop. The backstop is off by default (`pii.use_regex_backstop`), so this model loads only when it is enabled |
| `en_AU/` (`index.dic`, `index.aff`) | Hunspell dictionary (read with `spylls`) | `process/spellfix.py` | Dictionary gate for OCR character-confusion repair in the `spellfix` stage |

### What the layout model feeds

On OCR'd pages, `_layout_blocks_and_tables` in `ingest/strategies_scanned.py`
uses layout regions for three things that reach output:

1. **Table regions** — the only place `ocr_tables.reconstruct_table` runs.
2. **Dominant region kind** — the page's OCR text is collapsed onto one block,
   typed by the largest region (promoted to `paragraph` when it carries prose).
3. **Redaction exclusion regions** — `redact/stage.py` drops raster redaction
   hits inside figure / chart / form-background regions when
   `redaction.use_layout_filter` is on.

Other detected classes (heading, list item, caption, footer, footnote) do not
reach the element stream: layout blocks carry no text, and OCR text is not yet
assigned to layout regions. The per-class layout scores in
`docs/accuracy/EXTRACTION.md` measure the detector's raw regions, not the
elements written. Native-text pages never run the layout model.

Both layout call sites catch failure: without `ultralytics` or the weights, OCR
pages fall back to one paragraph block per page and the redaction filter
becomes a no-op.

## Hosted models

| Model | Provider | Used by | What it does | Enabled by |
|---|---|---|---|---|
| `kanon-2-enricher` | Isaacus (hosted API or SageMaker) | `analyse/enrich.py`, `analyse/enrich_stage.py` | Builds the ILGS document graph: segments, entities, relationships. PII detection reads PERSON / ADDRESS candidates from this graph | `enrichment.enabled` + `ISAACUS_API_KEY` or `ISAACUS_SAGEMAKER_ENDPOINTS` |
| `kanon-2-enricher` (as `chunking.chunking_model`) | Isaacus | `process/chunker.py` via semchunk 4 | AI chunking: boundaries follow the enrichment's structure. Reuses a persisted `*.enrichment_doc.parquet` Document when its text matches | `chunking.chunking_model` set |
| `kanon-2-embedder` | Isaacus | `analyse/embed.py`, `analyse/embed_stage.py` | Chunk embeddings (`retrieval/document`; `retrieval/query` for queries) | `embed` stage + Isaacus credentials |
| `mistral.pixtral-large-2502-v1:0` | Mistral Pixtral Large via AWS Bedrock | `ingest/llm_ocr.py` | VLM OCR returning markdown with reading order resolved. No regions, so it skips the layout pass and table reconstruction | `ocr.engine: mistral-ocr` + AWS credentials. Override with `engine_options.model` or `MISTRAL_OCR_MODEL_ID` |
| `llama3.2-vision` (default) | Local Ollama, OpenAI-compatible endpoint | `ingest/llm_ocr.py` (`OllamaOCRReader`) | Same role and shape as the Mistral engine | `ocr.engine: ollama`; endpoint from `OLLAMA_BASE_URL` |

The default OCR engine is `paddleocr`, so a default run calls no hosted OCR.

## Related

- `models/README.md` — artefact sources, sizes and checksums for the repo-root models
- `docs/decisions.md` — why the DocLayNet layout model replaced COCO, and the
  region-based-engines-only scope of table reconstruction
- `utils/models.py` — resolution order and the run record
