# Models Directory

Pre-downloaded ML models for offline/edge deployment. Womblex resolves these
automatically via `utils/models.py` — no manual path configuration required.

## Models

### all-MiniLM-L6-v2

- **Type:** Sentence Transformer (embedding model)
- **Source:** `sentence-transformers/all-MiniLM-L6-v2` (Hugging Face)
- **Size:** ~91 MB
- **Used by:** `pii/cleaner.py` — context-similarity validation for PERSON candidate spans
- **Layout:** HuggingFace hub snapshot layout (`refs/main` → `snapshots/<hash>/`)

### yolo11n_doc_layout.pt

- **Type:** YOLO11 nano object-detection weights (DocLayNet-finetuned)
- **Source:** [Armaggheddon/yolo11-document-layout](https://huggingface.co/Armaggheddon/yolo11-document-layout) (HF), MIT license
- **Size:** 5.37 MB
- **SHA-256:** `3629fc7abe8cca55ff490e16cccad7a100cbd814881163258815513e0a37881f`
- **Classes:** 11 DocLayNet — `Caption`, `Footnote`, `Formula`, `List-item`, `Page-footer`, `Page-header`, `Picture`, `Section-header`, `Table`, `Text`, `Title`
- **Recommended inference resolution:** 832 default. The model card recommends 1280, but on the ACT FOI corpus 832 matches or beats 1280 on dominant text classes at ~3× the speed, and small-class recall (Caption / Footnote) is poor at any resolution on this corpus. Increase to 1280 for document genres where small classes matter.
- **Used by:** `ingest/paddle_ocr.py` — primary layout backend via `YOLOLayoutAnalyzer`; also consumed by `redact/stage.py` as exclusion regions on raster-fallback redaction detection
- **Layout:** Bare `.pt` file

### yolov8n.pt

- **Type:** YOLOv8 nano object-detection weights (COCO-pretrained)
- **Source:** Ultralytics
- **Size:** ~6 MB
- **Used by:** `ingest/paddle_ocr.py` — fallback layout backend if the DocLayNet checkpoint is unavailable (e.g. partial installs). COCO classes have no document semantics; the fallback exists only to keep the layout path functional, not to produce useful predictions.
- **Layout:** Bare `.pt` file

## How path resolution works

`utils/models.py` walks up from the installed package to find the `models/`
directory (sibling of `src/`). Override with `WOMBLEX_MODELS_DIR` env var if
your layout differs.

```python
from womblex.utils.models import resolve_local_model_path

path = resolve_local_model_path("all-MiniLM-L6-v2")
# → Path(".../models/all-MiniLM-L6-v2/snapshots/<hash>/")
#   or "all-MiniLM-L6-v2" if models/ not found (falls back to HF download)

path = resolve_local_model_path("yolov8n.pt")
# → Path(".../models/yolov8n.pt")
#   or "yolov8n.pt" if not found
```

Both models are loaded lazily — no import cost until the relevant stage runs.
