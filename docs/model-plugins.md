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

### Pre-run check

Before a run (and at worker start and stage preflight), `utils/model_check.py`
checks every model the config names, at `processing.models_check`: `load`
(the default) builds the model through its factory; `smoke` also runs one
inference on a built-in input. Make a lazily-built model load at check time by
giving it a public `load()` method. A model that lives behind a service
(nothing local to load) should define `ping()`: it is called at `load` and
`smoke` alike and should raise unless the service is reachable, the credentials
are accepted and the model is served, at the smallest cost the service allows.
To report which variant of a model resolved
(as PaddleOCR reports v5 or the wheel's v4), set a `model_variant` string on the
object; it appears in the check result and the run record. A `smoke` check
needs, per slot: OCR, a non-empty reading of a rendered text line; layout,
regions that pass `check_layout_regions`; PII context, `encode` returning
`(1, dim)`; tokeniser, a non-empty chunking of a sentence; dictionary,
`lookup("the")` truthy. A failure names the slot, the model and the reason.

### PII context model

`pii.context_similarity_threshold` is calibrated to the default model.
Recalibrate it whenever `pii.model` changes; scores from another model are not
comparable.

## Provenance

Every pipeline Parquet's footer names, per slot, the model that produced it —
not only its name but the distribution and version that supplied it, so a
reader can tell a built-in from a plugin and pin the plugin's own release.
This is written automatically: `utils/model_registry.py` records a slot the
moment its factory is actually built (not merely resolved — a config-check
that only validates a name writes nothing, and nor does the pre-run model
check building a model only to confirm it loads: `check_models` runs under
`suppress_use_recording`, so that build is a probe, not a use), and
`store/run_stamp.py` reads that record at footer time into the
`womblex.slot_models` key, read back with `read_footer_slot_models`.
`store/run_manifest.py` unions it into the run record's `slot_models`, with
the stages that used each model. A plugin author does nothing to make this
happen; it follows from the factory being called through the registry, which
every slot already is — but a plugin that caches its own built model behind
the registry (as the built-in layout analyser and spellfix dictionary do)
should check `recording_suppressed()` to bypass that cache during the check,
or a real run immediately afterwards would hit the cached instance and never
record it either.

This is distinct from the loaded-model record below, which names model
*artefacts* on disk by digest: an API-backed engine (Mistral via Bedrock,
Ollama) has no artefact to digest but still used a slot, and the slot record
is what names it.

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

A plugin that loads its weights from inside its own package rather than
through a models root (bundled wheel data, for instance) calls
`utils/models.record_loaded_path(name, path)` directly so those bytes still
reach the record — the same path the built-in PaddleOCR reader uses for its
wheel-bundled v4 fallback.

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
