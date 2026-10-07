# Models

Every model Womblex loads or calls, what it does, and where it comes from.
Local models are resolved through `utils/models.py`; hosted models are called
only when their stage is enabled and a key, endpoint or credential is present.

## Local models

Resolved per artefact from `WOMBLEX_MODELS_DIR`, then `src/womblex/_models/`,
then `models/` (repo-root, editable installs), then any root an installed
package declares under the `womblex.model_roots` entry-point group — last, so a
plugin cannot shadow a built-in artefact. Of
`_models/`, only `en_AU/` and `kanon-2-tokenizer/` ship in the wheel; the
others are in the repository and reach a wheel install through
`WOMBLEX_MODELS_DIR` or a model-roots package. Every
successful resolution is recorded with a content digest and written into each
Parquet footer, so a run's record names the models it actually loaded.

| Model | Type | Used by | What it does |
|---|---|---|---|
| `paddleocr-v5/` (`ppocrv5-mobile-det`, `-rec`, `-cls` ONNX + dict) | PaddleOCR v5 mobile, ONNX | `ingest/paddle_ocr.py` (`PaddleOCRReader`) | Text detection, recognition and orientation on scanned pages and images. Run by `rapidocr-onnxruntime`; no PaddlePaddle framework. Preferred when all four files are present |
| `rapidocr-bundled-v4` | PaddleOCR v4, ONNX (inside the `rapidocr-onnxruntime` wheel) | `ingest/paddle_ocr.py` | Fallback OCR models when `paddleocr-v5/` is absent or incomplete. Loaded by the library itself; recorded via `record_loaded_path` |
| `pp-doclayout-m/` (`inference.onnx` + `inference.yml`) | PP-DocLayout-M, 23 layout classes, ONNX. Source: `PaddlePaddle/PP-DocLayout-M` (HF, Apache-2.0) at revision `7dbfcce3154a55776dc71ca026a4a2a8388dad8d`, exported with `paddle2onnx` 2.1.0 (opset 14, batch size 1). SHA-256 of `inference.onnx`: `a5a92ea2e5507c2a25eeb21720a1cfa69f722e512564969f690b1e4475c04250`; of `inference.yml`: `76aeb103310f432ea773d4a6d187e15b26d92c051c5e2875598e1d49f725d70d` | `ingest/layout_onnx.py` (`PPDocLayoutAnalyzer`) | Layout regions on OCR'd pages and redaction exclusion zones. The registered layout model (default of the `layout` slot); 640 px input, 0.3 threshold |
| `kanon-2-tokenizer/` | Kanon-2 tokeniser (Hugging Face `isaacus/kanon-2-tokenizer`) | `process/chunker.py`, `utils/token_packer.py` | Local token counting: semchunk chunk sizes, and token-budgeted packing of enrichment requests and ground-truth segments. No API call |
| `all-MiniLM-L6-v2/` | Sentence Transformer embedding model | `pii/cleaner.py` | Cosine-context check on PERSON candidates in the regex backstop. The backstop is off by default (`pii.use_regex_backstop`), so this model loads only when it is enabled. Selected by name through `pii.model` (slot `pii-context`; a plugin supplies an `encode(texts) -> embeddings` object). `context_similarity_threshold` is calibrated to this model: recalibrate it when `pii.model` changes |
| `en_AU/` (`index.dic`, `index.aff`) | Hunspell dictionary (read with `spylls`) | `process/spellfix.py` | Dictionary gate for OCR character-confusion repair in the `spellfix` stage |

### What the layout model feeds

The layout step (`ingest/layout_step.py`) runs the model once per selected page
and persists its regions in `*.layout_regions.parquet` ([layout.md](layout.md)).
Three things read them and reach output:

1. **Table regions** — on OCR'd pages, `_layout_blocks_and_tables` in
   `ingest/strategies_scanned.py` passes them to `ocr_tables.reconstruct_table`,
   the only place it runs.
2. **Dominant region kind** — the page's OCR text is collapsed onto one block,
   typed by the largest region (promoted to `paragraph` when it carries prose).
3. **Redaction exclusion regions** — `redact/stage.py` drops raster redaction
   hits inside `figure` / `table` regions when `redaction.use_layout_filter` is
   on.

Other detected classes (heading, list item, caption, footer, footnote) do not
reach the element stream: layout blocks carry no text, and OCR text is not yet
assigned to layout regions. The per-class layout scores in
`docs/accuracy/EXTRACTION.md` measure the detector's raw regions, not the
elements written. A native-text page is analysed only when redaction's filter
needs it (no vector redaction) or `layout.page_scope` is `all`.

Without `onnxruntime` or `pp-doclayout-m/inference.onnx`, the pre-run model
check stops `womblex run`. With the check off, the step records each selected
page as an `error` row: OCR pages fall back to one paragraph block per page, and
redaction runs those pages unfiltered and records them.

Re-exporting `pp-doclayout-m/`: fetch `PaddlePaddle/PP-DocLayout-M` at the revision above, then run
`paddle2onnx --model_dir . --model_filename inference.json --params_filename inference.pdiparams --save_file inference.onnx --opset_version 14`
in a venv with `paddle2onnx`, `paddlepaddle`, `onnx`, `onnxruntime` and `setuptools` (`paddle2onnx` 2.1.0 needs
`setuptools` and pulls in none of them). The export takes `image` (N,3,640,640) and `scale_factor` (N,2), batch size 1.
A different `paddle2onnx` version may change the ONNX digest.

## Hosted models

| Model | Provider | Used by | What it does | Enabled by |
|---|---|---|---|---|
| `kanon-2-enricher` | Isaacus (hosted API or SageMaker) | `analyse/enrich.py`, `analyse/enrich_stage.py` | Builds the ILGS document graph: segments, entities, relationships. PII detection reads PERSON / ADDRESS candidates from this graph | `enrichment.enabled` + `ISAACUS_API_KEY` or `ISAACUS_SAGEMAKER_ENDPOINTS` |
| `kanon-2-enricher` (as `chunking.chunking_model`) | Isaacus | `process/chunker.py` via semchunk 4 | AI chunking: boundaries follow the enrichment's structure. Reuses a persisted `*.enrichment_doc.parquet` Document when its text matches | `chunking.chunking_model` set |
| `kanon-2-embedder` | Isaacus | `analyse/embed.py`, `analyse/embed_stage.py` | Chunk embeddings (`retrieval/document`; `retrieval/query` for queries) | `embed` stage + Isaacus credentials |
| `mistral.pixtral-large-2502-v1:0` | Mistral Pixtral Large via AWS Bedrock | `ingest/llm_ocr.py` | VLM OCR returning markdown with reading order resolved. No regions, so it skips the layout pass and table reconstruction | `ocr.engine: mistral-ocr` + AWS credentials. Override with `engine_options.model` or `MISTRAL_OCR_MODEL_ID` |
| `llama3.2-vision` (default) | Local Ollama, OpenAI-compatible endpoint | `ingest/llm_ocr.py` (`OllamaOCRReader`) | Same role and shape as the Mistral engine | `ocr.engine: ollama`; endpoint from `OLLAMA_BASE_URL` |

The default OCR engine is `paddleocr`, so a default run calls no hosted OCR.

### Connecting Isaacus

The Isaacus SDK is a core dependency and stays dormant until one of two
deployments is declared (see `.env.example`):

- **Hosted API.** Set `ISAACUS_API_KEY`.
- **Your own AWS account, on SageMaker.** Set `ISAACUS_SAGEMAKER_ENDPOINTS`
  *instead of* the key, after deploying the Marketplace package(s). Every
  stage that calls Kanon-2 (AI chunking, `enrich`, `embed`) then routes
  through the endpoints with no other change. Entries are comma-separated
  `name[@region][=model|model|...]`; an entry with no `=models` part serves
  every model.

  ```bash
  export ISAACUS_SAGEMAKER_ENDPOINTS="kanon-2-universal-001"                                   # one endpoint, all models
  export ISAACUS_SAGEMAKER_ENDPOINTS="embed-001=kanon-2-embedder,enrich-001=kanon-2-enricher"  # per model
  export ISAACUS_SAGEMAKER_REGION="ap-southeast-2"   # optional; else the AWS SDK default
  export ISAACUS_SAGEMAKER_PROFILE="my-aws-profile"  # optional; else the AWS SDK default
  ```

  A stage whose model no endpoint serves fails before its first request,
  naming the model and the models the endpoints do serve. Chunk-size token
  counting stays local, because the Kanon-2 tokeniser is vendored.

**SageMaker credentials and MinIO.** SageMaker calls are SigV4-signed through
boto3's standard credential chain, which on EC2 resolves the instance role.
boto3 checks environment variables before the instance role, so an
`AWS_ACCESS_KEY_ID` set for a MinIO object store (for example `minioadmin`)
also replaces the role for the SageMaker signer, and calls fail with 403.
Setting the real AWS keys instead makes s3fs fail against MinIO. Give the
object store its own credentials on `WOMBLEX_S3_ACCESS_KEY_ID` /
`WOMBLEX_S3_SECRET_ACCESS_KEY` and leave `AWS_ACCESS_KEY_ID` unset. These keys
are read at process start-up by the CLI, workers and console alike, so
rotating one is an environment change and a redeploy.

## Related

- `models/README.md` — artefact sources, sizes and checksums for the repo-root models
