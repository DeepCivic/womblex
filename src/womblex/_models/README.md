# Bundled Models

Models shipped inside the installed package for offline / air-gapped use.
Womblex resolves these automatically via `utils/models.py` — no manual path
configuration required. See [`docs/models.md`](../../../docs/models.md) for
what every model (local and hosted) does in the pipeline.

## Models

### paddleocr-v5/

- **Type:** PaddleOCR v5 mobile detection, recognition and orientation models (ONNX) + character dictionary
- **Files:** `ppocrv5-mobile-det.onnx`, `ppocrv5-mobile-rec.onnx`, `ppocrv5-cls.onnx`, `ppocrv5_dict.txt`
- **Size:** ~21 MB
- **Used by:** `ingest/paddle_ocr.py` (`PaddleOCRReader`) — run by `rapidocr-onnxruntime`, no PaddlePaddle framework. When any file is missing, the reader falls back to the v4 models inside the `rapidocr-onnxruntime` wheel

### pp-doclayout-m/

- **Type:** PP-DocLayout-M layout detector (ONNX) + the model's `inference.yml` (labels, preprocessing)
- **Source:** [PaddlePaddle/PP-DocLayout-M](https://huggingface.co/PaddlePaddle/PP-DocLayout-M) (HF), Apache-2.0; exported with `paddle2onnx` 2.1.0, opset 14. Digests in `docs/models.md`
- **Size:** ~23 MB
- **Used by:** `ingest/layout_onnx.py` (`PPDocLayoutAnalyzer`) — layout backend; also `redact/stage.py` for raster-fallback exclusion regions. Not in package-data; resolved via `WOMBLEX_MODELS_DIR` or an editable install

### kanon-2-tokenizer/

- **Type:** Kanon-2 tokeniser (Hugging Face `isaacus/kanon-2-tokenizer`)
- **Size:** ~5 MB
- **Used by:** `process/chunker.py` (semchunk token counting) and `utils/token_packer.py` (token-budgeted enrichment requests and ground-truth segments). Local only — no API call

### all-MiniLM-L6-v2/

- **Type:** Sentence Transformer (embedding model)
- **Source:** `sentence-transformers/all-MiniLM-L6-v2` (Hugging Face)
- **Size:** ~88 MB
- **Used by:** `pii/cleaner.py` — context-similarity check on PERSON candidates in the opt-in regex backstop (`pii.use_regex_backstop`, default off)
- **Layout:** HuggingFace hub snapshot layout (`refs/main` → `snapshots/<hash>/`)

### en_AU/

- **Type:** en_AU Hunspell dictionary (`index.dic`, `index.aff`), read with `spylls`
- **Source:** SCOWL / wordlist.sourceforge.net — see `en_AU/LICENSE`
- **Size:** ~570 KB
- **Used by:** `process/spellfix.py` — dictionary gate for OCR character-confusion repair

## How path resolution works

`utils/models.py` searches per artefact, in order: `WOMBLEX_MODELS_DIR`, this
bundled `_models/` directory, then `models/` beside `src/` (editable installs).
Every artefact it finds is recorded, with a content digest, in each Parquet
footer the run writes.

```python
from womblex.utils.models import resolve_local_model_path

path = resolve_local_model_path("pp-doclayout-m")
# → Path(".../_models/pp-doclayout-m")
#   or "pp-doclayout-m" if not found
```

All models are loaded lazily — no import cost until the relevant stage runs.
