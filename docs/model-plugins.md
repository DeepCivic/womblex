# Model Plugins

How an installed package supplies a model for a pipeline slot. The registry is
`utils/model_registry.py`; the requirement and design calls are in
[`functional_requirements.md`](functional_requirements.md) (O1) and
[`decisions.md`](decisions.md).

## How a package registers a model

Declare an entry point in the group `womblex.models.<slot>`. The entry-point
name is the name a config uses; its value is the factory.

```toml
[project.entry-points."womblex.models.ocr"]
my-ocr = "my_pkg.ocr:make_reader"
```

A config names registered models only. An unknown name raises and lists the
known names; anything shaped like an import path is refused. A plugin cannot
take a built-in's name or alias. Options in the config (`engine_options`,
`layout_options`, `model_options`, `tokenizer_options`, `dict_options`)
reach the factory as keyword arguments, unchanged.

## Slots

| Slot | Group | Config key | Factory receives | Factory returns |
|---|---|---|---|---|
| OCR engine | `womblex.models.ocr` | `extraction.ocr.engine` | `lang` plus `engine_options` | a reader with `read_page(img) -> OCRPageResult` |
| Layout analyser | `womblex.models.layout` | `extraction.ocr.layout_model`, `redaction.layout_model` | `layout_options` | an object with `analyze(img, conf_threshold) -> list[LayoutRegion]` |
| PII context model | `womblex.models.pii-context` | `pii.model` | `model_options` | an object with `encode(texts) -> embeddings` |
| Chunk tokeniser | `womblex.models.tokenizer` | `chunking.tokenizer` | `tokenizer_options` | a Hugging Face id string, or a `(str) -> int` token counter |
| Spellfix dictionary | `womblex.models.spellfix-dictionary` | `spellfix.dict_name` | `dict_options` | an object with `lookup(word)`, as a Hunspell dictionary has |

Slot names and the interfaces above are defined in the code that resolves
them: `ingest/paddle_ocr.py`, `ingest/interfaces/protocols.py`,
`pii/cleaner.py`, `process/chunker.py`, `process/spellfix.py`.

### OCR output shape

A reader returns an `OCRPageResult`. A region engine fills `regions`
(four-corner bbox, text, confidence 0-1) and goes through the layout pass and
table reconstruction like PaddleOCR. An engine that returns page markdown
fills `markdown` and sets `reading_order_native=True`; declare it on the
factory so the pipeline handles it like the Mistral and Ollama engines:

```python
make_reader.womblex_traits = {"markdown": True}
```

### Layout vocabulary

Each region's `block_type` must be one of `LAYOUT_BLOCK_TYPES` in
`ingest/interfaces/protocols.py`. `check_layout_regions` enforces it; a
non-conforming result is logged and the page falls back to full-page text.
Run it over your model's output in your package's tests. The same model
applies to redaction detection when named in `redaction.layout_model`.

### PII context model

`pii.context_similarity_threshold` is calibrated to the default model.
Recalibrate it whenever `pii.model` changes; scores from another model are not
comparable.

## Supplying model files

A plugin resolves its files by name through
`utils/models.resolve_local_model_path`, the same call bundled models use, so
resolution is offline. To make a package's directory a search root, declare it
under `womblex.model_roots`. The value is a directory path, a callable
returning one, or a callable returning several:

```toml
[project.entry-points."womblex.model_roots"]
my-pkg = "my_pkg:models_dir"
```

Plugin roots are searched after `WOMBLEX_MODELS_DIR`, the bundled `_models/`
and the repo `models/`, so a plugin cannot shadow a built-in artefact. Every
file resolved is recorded in the run's loaded-model record with its digest.

## Minimal example

A region-based OCR engine and its model root:

```python
# my_pkg/ocr.py
from womblex.ingest.interfaces.protocols import OCRPageResult, OCRRegionResult

class Reader:
    def __init__(self, lang: str = "eng", **options): ...
    def read_page(self, img):
        return OCRPageResult(regions=[OCRRegionResult(
            bbox=[[0, 0], [10, 0], [10, 10], [0, 10]], text="hi", confidence=0.9)])

def make_reader(lang: str = "eng", **options) -> Reader:
    return Reader(lang, **options)
```

With the entry point above installed, `extraction.ocr.engine: my-ocr` selects it. No
Parquet schema changes, and with nothing configured `content_digest` is
unchanged (pinned by `tests/test_default_digest.py`).
