# Model Plugins

How an installed package supplies a model for a pipeline slot. The registry is
`utils/model_registry.py`.

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
`layout.options`, `model_options`, `tokenizer_options`, `dict_options`)
reach the factory as keyword arguments, unchanged. The enrichment and embedding
factories take no options key: they receive their config section instead.

## Slots

| Slot | Group | Config key | Factory receives | Factory returns |
|---|---|---|---|---|
| OCR engine | `womblex.models.ocr` | `extraction.ocr.engine` | `lang` plus `engine_options` | a reader with `read_page(img) -> OCRPageResult` |
| Layout analyser | `womblex.models.layout` | `layout.model` | `layout.options` | an object with `analyze(img, conf_threshold) -> list[LayoutRegion]` |
| PII context model | `womblex.models.pii-context` | `pii.model` | `model_options` | an object with `encode(texts) -> embeddings` |
| Chunk tokeniser | `womblex.models.tokenizer` | `chunking.tokenizer` | `tokenizer_options` | a Hugging Face id string, or a `(str) -> int` token counter |
| Spellfix dictionary | `womblex.models.spellfix-dictionary` | `spellfix.dict_name` | `dict_options` | an object with `lookup(word)`, as a Hunspell dictionary has |
| Enrichment provider | `womblex.models.enrichment` | `enrichment.model` | `config=` (`EnrichmentConfig`), `client=`, `token_counter=` | an `EnrichmentProvider` |
| Embedding provider | `womblex.models.embedding` | `embedding.model` | `config=` (`EmbeddingConfig`), `client=` | an `EmbeddingProvider` |

Slot names and the interfaces above are defined in the code that resolves
them: `ingest/paddle_ocr.py`, `ingest/interfaces/protocols.py`,
`pii/cleaner.py`, `process/chunker.py`, `process/spellfix.py`, `analyse/enrich.py`,
`analyse/embed.py`.

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
applies to redaction's exclusion zones too, since they read the layout step's regions.

### Pre-run check

Before a run (and at worker start and stage preflight), `utils/model_check.py`
checks every model the config names, at `processing.models_check`: `load`
(the default) builds the model through its factory; `smoke` also runs one
inference on a built-in input. Make a lazily-built model load at check time by
giving it a public `load()` method. A model that lives behind a service
(nothing local to load) should define `ping()`: it is called at `load` and
`smoke` alike and should raise unless the service is reachable, the credentials
are accepted and the model is served, at the smallest cost the service allows.
A model served by a registered deployment (`utils/deployment_registry.py`) is checked through it instead, once per deployment.
To report which variant of a model resolved
(as PaddleOCR reports v5 or the wheel's v4), set a `model_variant` string on the
object; it appears in the check result and the run record. A `smoke` check
needs, per slot: OCR, a non-empty reading of a rendered text line; layout,
regions that pass `check_layout_regions`; PII context, `encode` returning
`(1, dim)`; tokeniser, a non-empty chunking of a sentence; dictionary,
`lookup("the")` truthy; enrichment, one result for a sentence; embedding, one
non-empty vector. A failure names the slot, the model and the reason.

### Enrichment and embedding providers

An enrichment provider has a `name`, a `limits` (`ProviderLimits`) and
`enrich(texts) -> list[(EnrichmentResult, document | None)]`: one result per
text, its offsets into the text sent, independent of which other texts share
the request. Map labels onto the existing taxonomy inside the provider (a
`Person` of `type="natural"` is what PII masks as `PERSON`). The limits it
declares are what the stage packs and splits by: `max_texts_per_request`,
`token_ceiling` (`None` splits nothing; otherwise a longer text is split on blank
lines and the results offset-merged), `token_counter` (`count_batch(texts) -> list[int]`),
`needs_network`, and `returns_document`.

An embedding provider has a `name`, a `limits` (`EmbeddingLimits`:
`max_texts_per_request`, `needs_network`, and an optional `token_counter`
`plan` counts its tokens with) and `embed(texts) -> list[list[float]]`,
one vector per text in order. `embed` may take a `role` (`document` or `query`, or `None`), which the stage
passes from `embedding.task` (`retrieval/document` and `document` are the first role, `retrieval/query` and `query`
the second); a provider maps it to its own terms or ignores it, and its `docs/models.md` entry says which. A provider
whose `embed` takes no `role` is called without one. A provider that can return vectors of a requested width sets the
`dimensions` trait on its factory (`womblex_traits = {"dimensions": True}`); `embedding.dimensions` on one that has not is
refused before launch, as `POST /v1/runs`, `dispatch-workflow`, `run` and a worker all refuse.

Womblex keeps everything around the call: request packing, long-document
splitting, the table pass, the sidecars, the evidence reference and the
mention-to-chunk mapping. A provider changes who recognises entities or
computes vectors and nothing downstream.

Two factory traits answer the pre-launch gates before anything is built.
`returns_document` declares that `enrich` hands back a Document AI chunking
can reuse; without it nothing is persisted, `chunking.chunking_model` is set
aside, and chunks come from semchunk's local token splitting. `isaacus` gates the stage on an Isaacus deployment; leave it off a
provider Isaacus does not serve. A provider served by another deployment declares
it as `womblex_deployments` on its factory (with `womblex_vendor_model`), and the
stage is gated on and checked through that deployment instead. A provider with neither runs with
`client=None` and is checked like any local model.

```toml
[project.entry-points."womblex.models.enrichment"]
my-ner = "my_pkg.ner:make_provider"
```

```python
# my_pkg/ner.py
from womblex.analyse.enrich import ProviderLimits
from womblex.analyse.models import EnrichmentResult, Person, Span

class WordCounter:
    def count_batch(self, texts): return [len(t.split()) for t in texts]

class Provider:
    name = "my-ner"
    limits = ProviderLimits(max_texts_per_request=16, token_ceiling=4_000,
                            token_counter=WordCounter(), needs_network=False,
                            returns_document=False)
    def enrich(self, texts):
        return [(EnrichmentResult(text=t, type="other", persons=[
            Person(id="p0", name=Span(0, 4), type="natural", role="other",
                   mentions=[Span(0, 4)])]), None) for t in texts]

def make_provider(*, config, client=None, **_):
    return Provider()
```

### PII context model

`pii.context_similarity_threshold` is calibrated to the default model.
Recalibrate it whenever `pii.model` changes; scores from another model are not
comparable.

## Adding a provider: the checklist

A provider is a deployment (where a vendor's models are called from; `utils/deployment_registry.py`),
the models it serves in one or more slots, or both. The suite fails on a step left out of the
steps marked *checked*.

**A built-in vendor** lives in its own package under `providers/<vendor>/`:

1. The deployment, if it is new: `deployment.py` registers a `Deployment` (the variables it reads, network and billing,
   a free check taking the model ids a run uses on it, an optional billed `smoke(slot, model)`, `serves`, `detail`,
   `make_client`). *Checked:* every registered deployment meets the deployment conformance
   (`tests/test_deployment_registry.py`): variable names only, a network deployment is billed, a smoke call implies billed.
2. One adapter module per slot beside it, and the registration in the slot's own module
   (`analyse/enrich.py`, `analyse/embed.py`, `ingest/paddle_ocr.py`), naming `deployments=` and `vendor_model=`.
3. Conformance: nothing to write. `tests/test_provider_conformance.py` is parametrised over every registered model,
   so registering it tests it; a factory that needs options to be built goes in its `REQUIRED_OPTIONS`.
   *Checked:* the interface of its slot, and that every deployment it names is registered.
4. One entry in `docs/models.md` ("Hosted models") under the model's name or its vendor id, and, on a billed deployment,
   a **Paid:** line in `docs/functional_requirements.md` naming it. *Checked:* `tests/test_provider_docs.py`.
5. One CHANGELOG line.

The `providers/` row of the CLAUDE.md module table covers the package, so a new vendor adds no row. `docs/architecture.md`
and `docs/project-structure.md` list the new modules (`tests/test_doc_currency.py` holds CLAUDE.md's table to `src/`).
A womblex-benchmark pair is needed only when `SLOTS` or `DEFAULT_MODELS` change; a new non-default model needs none.

**A plugin package** does the same without Womblex's tree, and one package can supply a deployment and models in several slots:

```toml
[project.entry-points."womblex.deployments"]
my-vendor = "my_pkg.deployment:DEPLOYMENT"        # a Deployment named my-vendor

[project.entry-points."womblex.models.embedding"]
my-vendor-embed = "my_pkg.embedding:make_provider"  # factory.womblex_deployments = ("my-vendor",)

[project.entry-points."womblex.models.enrichment"]
my-vendor-ner = "my_pkg.ner:make_provider"
```

Installing it makes the deployment and every model selectable by name; the run record names the distribution and
version that supplied each. Test it as `tests/_fixture_vendor.py` does: present it through the entry-point groups and
run the conformance helpers over it.

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

A plugin root cannot shadow a built-in artefact, so give plugin files names of
their own.

A plugin that loads its weights from inside its own package rather than
through a models root (bundled wheel data, for instance) calls
`utils/models.record_loaded_path(name, path)` directly so those bytes still
reach the run's loaded-model record — the same path the built-in PaddleOCR reader uses for its
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
