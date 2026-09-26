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

### yolo11n_doc_layout.pt

- **Type:** YOLO11 nano object-detection weights (DocLayNet-finetuned, 11 classes)
- **Source:** [Armaggheddon/yolo11-document-layout](https://huggingface.co/Armaggheddon/yolo11-document-layout) (HF), MIT license
- **Size:** 5.37 MB
- **Used by:** `ingest/paddle_ocr.py` — primary layout backend via `YOLOLayoutAnalyzer`; also `redact/stage.py` for raster-fallback exclusion regions

### yolov8n.pt

- **Type:** YOLOv8 nano object-detection weights (COCO-pretrained)
- **Source:** Ultralytics
- **Size:** ~6 MB
- **Used by:** `ingest/paddle_ocr.py` — layout fallback only when `yolo11n_doc_layout.pt` is missing. COCO classes carry no document semantics

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

path = resolve_local_model_path("yolo11n_doc_layout.pt")
# → Path(".../_models/yolo11n_doc_layout.pt")
#   or "yolo11n_doc_layout.pt" if not found
```

All models are loaded lazily — no import cost until the relevant stage runs.
