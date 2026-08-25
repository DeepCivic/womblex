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

### yolov8n.pt

- **Type:** YOLOv8 nano object-detection weights (COCO-pretrained)
- **Source:** Ultralytics
- **Size:** ~6 MB
- **Used by:** `ingest/paddle_ocr.py` — primary layout backend via `YOLOLayoutAnalyzer`
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
