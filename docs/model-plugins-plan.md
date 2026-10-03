# Model plugins — plan

*Status: proposed (2026-10). No merge has shipped. This document lists outstanding work only: each merge deletes its own entry (and any open decision it settles) in the same PR, alongside that merge's doc updates. The final merge moves the calls into `decisions.md` and retires this document.*

## Context
Repo users should be able to swap in their own models, local or API-backed, without forking womblex. A packaged group of CPU-friendly models stays as the default.

This plan covers only the model slots that can be swapped **without a Parquet schema change**. Slots that need one are listed under *Schema bucket* and are out of scope here.

| Slot | Today | Fixed contract the plugin must meet |
|---|---|---|
| OCR engine | `get_ocr_reader` in `ingest/paddle_ocr.py`: a hardcoded alias dict and if-chain; kwargs whitelisted | `OCRReader` protocol; output is either regions or markdown |
| Layout analyser | `get_layout_analyzer()`: a no-argument singleton with no config | `LayoutAnalyzer` protocol; emits the womblex `block_type` vocabulary |
| PII context model | Model name hardcoded in `pii/cleaner.py` | No output change |
| Chunk tokeniser | Already takes a HF id or a callable; must be available offline | Chunk schema unchanged |
| Spellfix dictionary | Already config (`dict_name`) | No change needed beyond model-root support |

## Design
- **Registry with entry points.** A new `womblex/plugins.py` provides `register`, `resolve` and `available` per slot. Plugins declare entry points in *their own* `pyproject.toml`, in the groups `womblex.ocr`, `womblex.layout` and `womblex.tokenizer`. Womblex reads them with `importlib.metadata`, so its own dependencies don't change.
- **Built-ins use the same registry.** Paddle, Mistral, Ollama and PP-DocLayout-M register in code. Nothing gets a special path.
- **The baseline group lives in one place.** A single `BASELINE` table (slot → plugin name) feeds the config defaults. Changing which models are in the group means editing that table and nothing else.
- **Packaging stays undecided without costing debt.** Plugins name models and never file paths; they always resolve through `utils/models.resolve_local_model_path`. A new `womblex.model_roots` entry-point group lets any installed package add a search root. Whatever packaging is chosen later (wheel, separate models package, or image only), only package-data or a root registration changes. Plugin code doesn't.
- **Config names registered plugins only.** Import paths are not accepted, so a config submitted to the `/v1` service API can't make the server import arbitrary code.
- **Pre-run check instead of runtime fallback.**
  - Each plugin exposes `check(options)`, returning ok or a reason.
  - `plugins.preflight` takes `off`, `load` (resolve and load the models) or `smoke` (also run one inference on a tiny built-in input).
  - A failed check stops the run before any document is processed, so a run's output never mixes models.
  - The existing in-slot load fallback (PaddleOCR v5 → the wheel's v4) stays. Preflight reports which variant resolved.
- **Provenance goes in footer keys, not columns.** The schemas are untouched.

## Merges
W is womblex. B is womblex-benchmark, a paired merge.

- **M0 (W). Split `config.py`, mechanical only.** It is 920 lines, over the 750 cap, and the later merges add fields. Move the model-slot settings into a sibling module and re-export them, so no imports change.
  - Docs: `project-structure.md`.
- **M1 (W). Registry core.**
  - Add `womblex/plugins.py`: slots, entry-point loading, built-in registration.
  - The plugin protocol carries slot, name, distribution, version and `check()`.
  - An unknown name raises an error listing the known names.
  - Add the `BASELINE` table, and the `womblex.model_roots` hook in `utils/models.py`.
  - Tests: a fake plugin injected through monkeypatched entry points; a fake model root; duplicate and unknown names.
  - Docs: a `decisions.md` entry explaining why a registry is justified despite the "no strategy patterns" rule; `architecture.md`, `project-structure.md` and the `CLAUDE.md` module table; CHANGELOG.
- **M2 (W). OCR slot.**
  - `get_ocr_reader` resolves through the registry. Built-ins register with their existing aliases.
  - `engine_options` is passed through unfiltered, so the plugin's factory signature validates it.
  - Each reader declares `output_shape` (`regions` or `markdown`). That replaces the hardcoded `LLM_OCR_ENGINES` / `is_llm_engine` set.
  - `readtext` becomes an optional, documented protocol method.
  - The reader cache is keyed by engine plus a frozen copy of the options.
  - Profiling's confidence sampling in `morphology.py` stays on Paddle, because its thresholds were calibrated on Paddle.
  - Gate: `content_digest` is unchanged on the vendored fixtures under the default engine.
  - Docs: `models.md`, `CLAUDE.md` module table (`paddle_ocr.py` row), CHANGELOG.
- **M3 (W). Layout slot.**
  - New config `extraction.ocr.layout.engine` plus `options`. Built-ins: `pp-doclayout-m` (default) and `none`.
  - `redact/stage.py` gets the same layout selection threaded through instead of calling the bare singleton.
  - The protocol exports the `block_type` vocabulary as a constant, and a conformance test checks plugins against it.
  - Docs: `models.md`, `architecture.md`, CHANGELOG.
- **M4 (W). PII context model and tokeniser.**
  - New config `pii.context_model` (default `all-MiniLM-L6-v2`), passed straight to sentence-transformers.
  - The `womblex.tokenizer` group registers named callable token counters. The offline `tokenizer_available` check stays.
  - Docs: `models.md` notes that `context_similarity_threshold` must be recalibrated when the model changes; CHANGELOG.
- **M5 (W). Preflight wiring.**
  - `womblex run` runs preflight before batch one.
  - A worker runs preflight at startup and refuses jobs whose plugins fail it, using the existing refused path.
  - At API submit, the check is whether the name is registered. The API host may not carry the models, so it doesn't load them.
  - The stage-contract preflight calls the plugin's `check()`.
  - The preflight result goes into the run record.
  - Docs: `service-api.md`, `deployment-images.md`, CHANGELOG.
- **M6 (W). Provenance.**
  - New footer key `womblex.plugins`: slot, name, distribution, version.
  - Plugins report their weights through `record_loaded_path`, so they appear in `loaded_models()`.
  - `build_run_record` drops the "OCR engine not recorded" entry from `partial`.
  - Docs: `contract.md`, CHANGELOG.
- **M7 (W). Distributed images.**
  - A Dockerfile build arg installs plugin packages into the worker and UI images. Their model roots register through the M1 hook.
  - Docs: `deployment-images.md`, CHANGELOG.
- **M8 (W). Plugin authoring guide.**
  - A new `docs/plugins.md`: each slot's protocol, the entry-point groups, `check()`, provenance obligations, and a minimal example.
  - Move this plan's calls into `decisions.md` and retire this document.
- **M9 (B). Benchmark.** Reports name the plugins that produced them, and suites can run against a named plugin. `docs/accuracy/` stays labelled as baseline.

## Sequencing with the permissive-deps plan
`permissive-deps-plan.md` is in flight and touches the same files: `strategies_scanned.py`, `redact/stage.py` and `morphology.py` (P3a, P3b), plus YOLO's removal (L2).
- M3 assumes L2 has landed. YOLO is not offered as a layout plugin.
- M2 and M3 rebase onto whichever of P3a / P3b has landed. Neither plan changes the other's seams.

## Open decisions
- **Baseline group membership.** Settled by editing the `BASELINE` table.
- **Baseline packaging.** In the wheel, a separate models package, or image only. Wheel package-data is a `pyproject.toml` change and needs approval.
- **Contract version.** Whether the additive `womblex.plugins` footer key bumps `CONTRACT_VERSION` from 1.0 to 1.1 (M6).

## Schema bucket (out of scope for this plan)
Each of these needs a Parquet schema change, so each needs its own plan.
1. **Enrichment provider.** The `*.enrichment_entities` / `*.enrichment_meta` / `*.graph_edges` parquets are shaped around the Kanon-2 ILGS Document.
2. **AI chunking with a non-Isaacus model.** `*.enrichment_doc.parquet` holds ILGS JSON, and semchunk 4 needs an Isaacus client.
3. **Graph-driven PII detection.** It depends on the ILGS entity types (`natural`, `address`).
4. **Link-stage candidates.** They come from `enrichment_entities`.
5. **`graph_refresh`.** It rebuilds edges over the ILGS-shaped sidecars.
6. **Enrichment token-budget tokeniser.** It is tied to the enricher's rate-limit accounting.
7. **Per-page or per-element OCR engine recording.** This needs an `ELEMENT_SCHEMA` column.
8. **Embedder provider**, plus any CPU embedding baseline.

## Conventions this plan holds to
- **Merge size.** At most 500 changed lines per merge. Every merge passes `uv run ruff check src/ tests/`, `uv run mypy src/` and `uv run python -m pytest tests/ -v`. Touched files stay under 750 lines.
- **Docs close every merge.** The same PR updates the docs it affects, adds a CHANGELOG entry, and deletes its own entry from this document.
- **Thin adapters.** A plugin's own parameters pass through `options`. Womblex adds no per-library toggles.
- **Default behaviour is identical.** With no plugin configured, `content_digest` is unchanged on every fixture.
