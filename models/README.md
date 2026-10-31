# Models Directory

Pre-downloaded ML models for offline/edge deployment. Womblex resolves these
automatically via `utils/models.py` — no manual path configuration required.
The full set, including models bundled only in `src/womblex/_models/` and the
hosted models, is outlined in [`docs/models.md`](../docs/models.md).

## Models

### all-MiniLM-L6-v2

- **Type:** Sentence Transformer (embedding model)
- **Source:** `sentence-transformers/all-MiniLM-L6-v2` (Hugging Face)
- **Size:** ~91 MB
- **Used by:** `pii/cleaner.py` — context-similarity validation for PERSON candidate spans
- **Layout:** HuggingFace hub snapshot layout (`refs/main` → `snapshots/<hash>/`)

## How path resolution works

`utils/models.py` searches per artefact, in order: `WOMBLEX_MODELS_DIR`, the
bundled `src/womblex/_models/`, then this `models/` directory (sibling of
`src/`).

```python
from womblex.utils.models import resolve_local_model_path

path = resolve_local_model_path("all-MiniLM-L6-v2")
# → Path(".../models/all-MiniLM-L6-v2/snapshots/<hash>/")
#   or "all-MiniLM-L6-v2" if models/ not found (falls back to HF download)
```

All models are loaded lazily — no import cost until the relevant stage runs.
