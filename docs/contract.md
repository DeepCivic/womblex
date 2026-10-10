# Consumer contract — Womblex

What a program reading Womblex output, or importing Womblex, may rely on.
Two surfaces: the on-disk data contract (Parquet) and the Python API. The
bundle that carries both to another system is in [egress.md](egress.md); the
column-level schemas are in [extraction.md](extraction.md).

## Contract version

`womblex.contract_version` (currently `1.5`, `store/contract.CONTRACT_VERSION`)
is in the footer of every pipeline Parquet and in `egress_manifest.json`. It is
versioned apart from the package: a release that changes no schema leaves it
alone.

| Change to a schema | Bump | Obligation |
|---|---|---|
| Additive column | minor | Readers back-fill older files (`_MANIFEST_NULL_BACKFILL` style) |
| Rename or removal | major | Reader shim as for `_CHUNKS_BACKFILL`, kept for the old major |

A file with no `contract_version` key was written before the contract existed
and is read as `1.0`-compatible.

## Files, join keys and sensitivity

`womblex.sensitivity` is set per file role (`store/contract.ROLE_SENSITIVITY`,
the single source). An unknown role reads as `raw`.

| Role (`*.<role>.parquet`) | Joins on | Sensitivity |
|---|---|---|
| `elements` | `(source_hash, elem_order)` | raw |
| `table_cells`, `form_fields` | `(source_hash, parent_elem_order)` to `elements`; `table_cells.bbox` (1.3) locates a cell on the page, null where the producer has no cell geometry | raw |
| `_manifest` / run-root `manifest` | `source_hash` | none |
| `normalised_text`, `spellfix_text`, `spellfix_corrections` | `(source_hash, elem_order)` | raw |
| `chunks` | `(source_hash, chunk_index)`; table chunks anchor on `elem_order` | raw |
| `embeddings`, `chunk_quality` | `(source_hash, chunk_index)` | none |
| `enrichment_entities` | `(source_hash, entity_id)`; carries `chunk_index` to `chunks` (narrative mentions only; table mentions are `-1`). `text_layer` names the text `mention_start` / `mention_end` index: the narrative under its element-text layer, or `table_markdown` with the table named by `elem_order` or `sheet` (1.4; null on older files, read as narrative). `mention_text` is the text of that mention, which can differ from the entity's `name` (1.5; null on older files) | raw |
| `graph_edges` | `source_hash` + `source_id` / `target_id` to `enrichment_entities.entity_id` | raw |
| `enrichment_doc` | `source_hash` (one row per document) | raw |
| `enrichment_meta` | `source_hash`; `table_count` is tables sent to the enricher on their own, null where none were (1.4) | none |
| `entity_links` | `(source_hash, mention_start, mention_end)` to `enrichment_entities`; its `entity_id` is the reference-register id | raw |
| `pii_spans` | `(source_hash, chunk_index)`; `entity_id` to `enrichment_entities` | raw |
| `clean_text` | `(source_hash, chunk_index)`; `mask_status` is `masked`, `no_entity` or `not_masked` (verbatim, no candidate source covered the chunk); files written before contract 1.2 read back as `masked` where `n_masked` is above zero, null otherwise | masked |
| `money_spans`, `money_columns` | `source_hash` plus the locus anchor (`start_char` / `elem_order` / `parent_elem_order`) | none |
| `layout_regions` | `(source_hash, page)`; boxes are normalised like element `bbox` ([layout.md](layout.md)) | none |
| `redactions`, `source_index` | `source_hash` | none |
| `provenance` | `source_hash` | raw |

`none` is a claim about text, not derivation: embeddings come from unmasked
chunks but carry no text.

## Determinism

For a given `source_hash` + `womblex.version` + `config_digest` +
`womblex.models` + `womblex.slot_models`, extraction content and row order are
stable, so the manifest's `content_digest` matches. File bytes and
`extracted_at_iso` are not guaranteed. A mismatch is explained by the stamped
version, config and model digests and never blocks output.

## Model provenance

The run stamp is five footer keys (`womblex.run_id`, `version`, `commit`,
`config_digest`, `stage`), plus `womblex.preset` (the config's `dataset.name`)
when the config names one. `commit` is `unavailable:<reason>` when neither the
work tree nor a build stamp answers. A downstream sidecar inherits `run_id`,
`config_digest` and `preset` from the batch it annotates, preferring the
elements shard or manifest, then any sibling; a sidecar whose siblings carry no
run, or disagree, is written unstamped. Further keys name what produced a file:
`womblex.models` names loaded local model *artefacts* by digest
(`store/run_stamp.read_footer_models`); `womblex.slot_models` names which
swappable-slot model (`docs/model-plugins.md`) each slot actually built, by
slot, name, and the distribution and version that supplied it
(`read_footer_slot_models`); `womblex.model_check` records the pre-run model
check's result (`read_footer_model_check`). Each is written only when it has a
value — a process that loaded, built or checked nothing writes no key — and
all are additive metadata, so a reader that ignores them reads the file
unchanged. No column is added for any.

## Safe to hand onward

Only files whose footer says `sensitivity=masked` or `none`. Anything `raw`,
or carrying no `sensitivity` key, stays inside the trust boundary.

## Python API

`womblex.__all__` is the stable API, pinned by `tests/test_public_api.py`:
`__version__`; `extract_text`; the `run_*` operations; the `*_shards` stage functions;
`build_bundle`; `write_run_manifest`; `read_results` (reads the elements
role); `CONTRACT_VERSION`. Names resolve lazily, so `import womblex` does not
load the extraction stack. Anything not in `__all__` is internal and may change
in any release.

### Deprecation policy

A name in `__all__`, or a signature it exposes, is removed or changed
incompatibly only after one minor release in which it emits a
`DeprecationWarning` naming the replacement. The removal is recorded in the
CHANGELOG under `Removed`.
