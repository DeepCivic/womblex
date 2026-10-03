# Model plugins

Womblex runs a packaged, CPU-friendly default for every model slot. An installed
package can register another model for a slot; the config then names it, and the
run uses it in place of the default. No Parquet schema changes.

Model selection lives in `utils/model_registry.py`. Design calls and rejected
alternatives are in [decisions.md](decisions.md); the requirement is
[functional_requirements.md](functional_requirements.md) Requirement 24.

## Slots

| Slot | Config | Default | A factory returns |
|---|---|---|---|
| `ocr` | `extraction.ocr.engine`, `engine_options` | `paddleocr` | An `OCRReader` (`read_page(img) -> OCRPageResult`) |
| `layout` | `extraction.ocr.layout_model`, `layout_options`; `redaction.layout_model`, `layout_options` | `pp-doclayout-m` | A `LayoutAnalyzer` (`analyze(img, conf_threshold) -> list[LayoutRegionResult]`) |
| `pii-context` | `pii.model`, `model_options` | `all-MiniLM-L6-v2` | An object with `encode(texts) -> embeddings` (a 2-D array) |
| `tokenizer` | `chunking.tokenizer`, `tokenizer_options` | `kanon-2-tokenizer` | A Hugging Face id (or vendored name) string, or a `(str) -> int` token counter |
| `spellfix-dictionary` | `spellfix.dict_name`, `dict_options` | `en_AU` | An object with `lookup(word)` returning truthy for a valid word |

A factory is called with the config's options as keyword arguments, unchanged.
The `ocr` factory also receives `lang`. Womblex adds no toggles of its own: the
engine's parameters are the options.

## Registering a model

Declare an entry point in the group `womblex.models.<slot>`. The entry-point
name is the name the config uses; the value is the factory.

```toml
[project.entry-points."womblex.models.layout"]
my-layout = "my_pkg.layout:make_analyzer"
```

Then select it:

```yaml
extraction:
  ocr:
    layout_model: my-layout
    layout_options: {threshold: 0.4}
```

Names are case-insensitive. A name already held by a built-in or another
package is refused, so a plugin cannot shadow a built-in. A config naming an
unregistered model is an error that lists the known names, and import paths
(`pkg.mod:fn`) are refused.

## Slot rules

**OCR.** A region engine returns `OCRPageResult.regions`; an engine that returns
page markdown sets `markdown` and `reading_order_native=True`, and its factory
declares the trait so the pipeline skips preprocessing and layout for it:

```python
def make_reader(lang: str = "eng", **options):
    return MyReader(**options)

make_reader.womblex_traits = {"markdown": True}
```

**Layout.** Regions must be sorted top-to-bottom, carry a well-formed box and a
confidence in 0 to 1, and use a womblex `block_type`: `paragraph`, `heading`,
`list_item`, `caption`, `header`, `footer`, `footnote`, `signature`, `figure`
or `table`. `check_layout_regions` in `womblex.ingest.interfaces.protocols` is
the conformance check; run it over your analyser's output in your own tests. A
non-conforming model is logged and the page falls back to full-page text. The
same analyser serves redaction detection when named under `redaction`.

**PII context.** Candidates score by cosine similarity against reference
contexts, so Womblex only needs `encode`. `pii.context_similarity_threshold`
(default 0.35) is calibrated to `all-MiniLM-L6-v2`: recalibrate it when you
change `pii.model`.

**Tokeniser.** The built-in `huggingface` model reaches any Hugging Face
tokeniser by option (`tokenizer_options: {name: org/tok}`). A bare Hugging Face
id in `chunking.tokenizer` is not a registered name.

**Spellfix dictionary.** The built-in `hunspell` model loads another Hunspell
directory by option (`dict_options: {name: en_GB}`).

## Model files

A package that ships weights offers a directory through the
`womblex.model_roots` entry-point group. The value is a directory or a callable
returning one:

```toml
[project.entry-points."womblex.model_roots"]
my-pkg = "my_pkg:models_dir"
```

Plugin roots are searched after `WOMBLEX_MODELS_DIR`, the bundled `_models/` and
the repo `models/`, so they supplement those roots and cannot shadow a bundled
artefact. Resolve files with `womblex.utils.models.resolve_local_model_path(name)`
inside the factory so they resolve offline and appear in the run's loaded-model
record.

## Minimal example

A layout plugin that treats the whole page as one paragraph:

```python
# my_pkg/layout.py
from womblex.ingest.interfaces.protocols import LayoutRegionResult


class WholePage:
    def __init__(self, **options):
        pass

    def analyze(self, img, conf_threshold=0.3):
        h, w = img.shape[:2]
        return [LayoutRegionResult((0, 0, w, h), "page", "paragraph", 1.0)]
```

```toml
[project.entry-points."womblex.models.layout"]
whole-page = "my_pkg.layout:WholePage"
```

With the package installed, `layout_model: whole-page` selects it.
