"""The Pipeline Composer's read model (docs/ui-plan.md merge 9).

Two static sources, neither run-scoped — unlike every other screen, the
composer edits configuration and shows the pipeline's shape, it does not
read a run's artefacts:

- `cloud/stage_contracts.py`'s `STAGE_CONTRACTS` is already the pipeline's
  DAG as data. `get_stage_graph()` renders it as nodes/edges instead of the
  frontend hand-coding one — the plan's §3 "Do not hand-code the DAG in the
  frontend" rule.
- `config.py`'s `WomblexConfig` is already a validated Pydantic model. The
  form is its JSON Schema; validation and the YAML download both go through
  the same `WomblexConfig(**raw)` construction `load_config` uses, so the
  console can never accept a config the CLI would reject.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, get_args

import yaml
from pydantic import BaseModel, ValidationError

from womblex.cloud.stage_contracts import (
    DISCOVERY_SUFFIXES,
    PRODUCER_OF,
    STAGE_CONTRACTS,
    STAGE_NAMES,
)
from womblex.config import DatasetConfig, PathsConfig, WomblexConfig
from womblex.store.remote import is_remote_uri
from womblex.ui.deps import UISettings

#: Extraction is not itself a `StageContract` — it runs inside
#: `process_batch`, not `run-stage` — but every stage's element-stream
#: inputs come from it, so the graph needs a source node for those edges to
#: point at.
EXTRACT_NODE = "extract"

#: Which `WomblexConfig` section configures each stage. Not derivable from
#: `StageContract` — contracts name parquet suffixes, not config fields — so it
#: is declared here, beside the config models, rather than in the frontend
#: where it would drift silently (`tests/test_ui.py` asserts every name is a
#: real field). It is what lets a node in the composer's graph carry its own
#: enabled toggle instead of the operator hunting for the matching section.
#: `graph-refresh` is absent because it has none: it re-derives edges from what
#: `enrich` and `chunk` already wrote. `extract` is absent because it is
#: configured by `detection` + `extraction` + `redaction` together, so no one
#: section is *the* one.
CONFIG_SECTION: dict[str, str] = {
    "normalise": "normalise",
    "spellfix": "spellfix",
    "chunk": "chunking",
    "money": "money",
    "enrich": "enrichment",
    "embed": "embedding",
    "link": "linking",
    "pii": "pii",
    "quality": "quality",
}

#: A config at schema defaults, used only to resolve `StageContract`'s
#: config-derived `conditional_inputs` / `outputs` for the structural graph.
#: `dataset` / `paths` have no defaults of their own (they name a real run)
#: so they're filled with placeholders that nothing here reads.
_DEFAULT_CONFIG = WomblexConfig(
    dataset=DatasetConfig(name="composer"),
    paths=PathsConfig(input_root=Path("."), output_root=Path("."), checkpoint_dir=Path(".")),
)


def _producer(suffix: str) -> str | None:
    """The stage (or `EXTRACT_NODE`) that writes *suffix*, if known."""
    if suffix in DISCOVERY_SUFFIXES:
        return EXTRACT_NODE
    return PRODUCER_OF.get(suffix)


def get_stage_graph() -> dict[str, Any]:
    """The pipeline DAG `STAGE_CONTRACTS` implies, at schema defaults.

    Edges come from `required_inputs` only — the hard ordering guardrail the
    plan's §3 names ("ensuring extraction precedes chunking"). Conditional
    inputs ride along on each node instead: they are config-derived, so an
    edge for one would hold only for whatever config the form happens to
    have, and a graph that reshapes as an operator edits a batch size reads
    worse than a fixed one annotated with what a stage *might* also read.

    One edge per ordered pair, carrying every suffix that justifies it:
    "must run after" is a relation two stages either stand in or do not, and
    emitting `chunk`'s three extraction sidecars separately would have every
    renderer dedupe them back to avoid parallel arrows.
    """
    nodes: list[dict[str, Any]] = [
        {
            "id": EXTRACT_NODE,
            "scope": None,
            "mutation": None,
            "needs_isaacus_api": False,
            "checkpoint_dirname": None,
            "required_inputs": [],
            "conditional_inputs": [],
            "outputs": list(DISCOVERY_SUFFIXES),
            "config_section": None,
        }
    ]
    # (producer, consumer) -> the suffixes justifying it, insertion-ordered.
    edges: dict[tuple[str, str], list[str]] = {}
    for name in STAGE_NAMES:
        contract = STAGE_CONTRACTS[name]
        conditional = [
            {"suffix": c.suffix, "reason": c.reason, "strict": c.strict}
            for c in contract.conditional_inputs(_DEFAULT_CONFIG)
        ]
        nodes.append(
            {
                "id": name,
                "scope": contract.scope.value,
                "mutation": contract.mutation.value,
                "needs_isaacus_api": contract.needs_isaacus_api,
                "checkpoint_dirname": contract.checkpoint_dirname,
                "required_inputs": list(contract.required_inputs),
                "conditional_inputs": conditional,
                "outputs": list(contract.outputs(_DEFAULT_CONFIG)),
                "config_section": CONFIG_SECTION.get(name),
            }
        )
        for suffix in contract.required_inputs:
            producer = _producer(suffix)
            if producer is not None:
                edges.setdefault((producer, name), []).append(suffix)
    return {
        "nodes": nodes,
        "edges": [
            {"from": producer, "to": consumer, "suffixes": suffixes}
            for (producer, consumer), suffixes in edges.items()
        ],
    }


def _nested_model(annotation: Any) -> type[BaseModel] | None:
    """The `BaseModel` in *annotation*, seeing through `X | None`.

    Returns `None` for a plain container — `normalise.substitutions` is a
    free-form `dict[str, str]` of letterhead replacements, and recursing into
    it would report every one of an operator's own substitution keys as an
    unrecognised config field.
    """
    for candidate in get_args(annotation) or (annotation,):
        if isinstance(candidate, type) and issubclass(candidate, BaseModel):
            return candidate
    return None


def unknown_keys(
    raw: dict[str, Any], model: type[BaseModel] = WomblexConfig, prefix: str = ""
) -> list[str]:
    """Dotted paths in *raw* that no field of *model* claims.

    Pydantic ignores unrecognised keys, which for a *config editor* is the
    worst failure mode available: `chunkng:` for `chunking:` validates clean
    and then vanishes from the rendered YAML, leaving a file that does not
    do what the operator typed and no signal anything was dropped. Naming
    them keeps that visible without making the composer stricter than the
    CLI (see `validate_config`).
    """
    found: list[str] = []
    for key, value in raw.items():
        field = model.model_fields.get(key)
        if field is None:
            found.append(f"{prefix}{key}")
            continue
        nested = _nested_model(field.annotation)
        if nested is not None and isinstance(value, dict):
            found.extend(unknown_keys(value, nested, f"{prefix}{key}."))
    return found


def get_config_schema() -> dict[str, Any]:
    """`WomblexConfig`'s JSON Schema — the composer form's field list, straight
    from Pydantic. No hand-typed mirror of `config.py` to fall out of sync.

    `paths` is stripped: it names the deployment's ingest/output locations,
    not something the operator retypes per run (docs/ui-ingest-plan.md §3).
    `validate_config`/`render_yaml` inject it back before construction.
    """
    schema = WomblexConfig.model_json_schema()
    schema.get("properties", {}).pop("paths", None)
    required = schema.get("required")
    if isinstance(required, list) and "paths" in required:
        required.remove("paths")
    return schema


#: `pathlib.Path` mangles an object-store URI (`Path("s3://foo")` collapses
#: to `s3:/foo`), so an object-store field is filled with this placeholder
#: instead and named in a YAML header comment (`render_yaml`) rather than
#: written where it would lie.
_PATHS_PLACEHOLDER = "."


def _deployment_paths(settings: UISettings) -> tuple[dict[str, str], list[str]]:
    """This deployment's `paths` section, plus env vars to note instead of a
    mangled object-store URI. A local folder is written verbatim."""
    env_vars: list[str] = []

    if settings.ingest_uri and is_remote_uri(settings.ingest_uri):
        input_root = _PATHS_PLACEHOLDER
        env_vars.append("$WOMBLEX_INGEST_URI")
    elif settings.ingest_uri:
        input_root = settings.ingest_uri
    else:
        input_root = str(settings.output_root) if settings.output_root else _PATHS_PLACEHOLDER

    if settings.store_uri and is_remote_uri(settings.store_uri):
        output_root = _PATHS_PLACEHOLDER
        env_vars.append("$WOMBLEX_STORE_URI")
    elif settings.store_uri:
        output_root = settings.store_uri
    else:
        output_root = str(settings.output_root) if settings.output_root else _PATHS_PLACEHOLDER

    paths = {
        "input_root": input_root,
        "output_root": output_root,
        "checkpoint_dir": _PATHS_PLACEHOLDER,
    }
    return paths, env_vars


def validate_config(raw: dict[str, Any], settings: UISettings) -> dict[str, Any]:
    """Try to build a `WomblexConfig` from *raw*; report Pydantic's own errors.

    *raw* carries no `paths` (the schema does not offer it); this deployment's
    locations are injected before constructing the model, the same
    `WomblexConfig(**raw)` construction `load_config` uses.

    `unknown_keys` is reported *beside* `valid` rather than folded into it,
    for exactly that reason: the CLI loads a config with a stray key without
    complaint, so failing it here would make the composer reject configs that
    run fine. It is a warning the operator has almost certainly made a typo,
    not a verdict on the config.
    """
    unknown = unknown_keys(raw)
    paths, _ = _deployment_paths(settings)
    try:
        WomblexConfig(**{**raw, "paths": paths})
    except ValidationError as e:
        errors = e.errors(include_url=False, include_context=False, include_input=False)
        return {"valid": False, "errors": errors, "unknown_keys": unknown}
    return {"valid": True, "errors": [], "unknown_keys": unknown}


def render_yaml(raw: dict[str, Any], settings: UISettings) -> str:
    """Validate *raw* and render it back as YAML, in the shape `load_config` reads.

    Round-trips through the validated model (`model_dump(mode="json")`)
    rather than dumping the posted dict verbatim, so the download always
    reflects Pydantic-applied defaults and coercions — what a fresh
    `womblex run --config <this file>` would actually see, not whatever the
    browser happened to send. `paths` comes from this deployment's ingest/
    output locations, injected the same way `validate_config` does.

    Dropped keys are recorded as a YAML comment at the top of the file, since
    the downloaded file is what gets committed and mailed around. Comments
    are inert to `yaml.safe_load`, so `load_config` reads it unchanged.

    When ingest/output are object-store URIs, `paths.input_root`/
    `output_root` in the body are a placeholder rather than a mangled
    `s3:/…` path — a second header comment names the env vars actually
    driving storage instead ("storage is env, not YAML").

    Raises `pydantic.ValidationError` on an invalid config; the route
    translates that to a 422 with the same error shape `validate_config`
    returns.
    """
    paths, env_vars = _deployment_paths(settings)
    config = WomblexConfig(**{**raw, "paths": paths})
    body = str(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))
    unknown = unknown_keys(raw)

    header_lines: list[str] = []
    if unknown:
        header_lines += [
            f"# WARNING: {len(unknown)} submitted key(s) are not in the Womblex config",
            "# schema and were dropped from this file (likely typos):",
            *(f"#   {key}" for key in unknown),
        ]
    if env_vars:
        header_lines += [
            "# paths.input_root/output_root above are placeholders — this deployment's "
            "storage is configured via " + " / ".join(env_vars) + ", not written here.",
        ]
    if not header_lines:
        return body
    return "\n".join([*header_lines, "", body])
