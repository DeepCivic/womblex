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
from typing import Any

import yaml
from pydantic import ValidationError

from womblex.cloud.stage_contracts import (
    DISCOVERY_SUFFIXES,
    PRODUCER_OF,
    STAGE_CONTRACTS,
    STAGE_NAMES,
)
from womblex.config import DatasetConfig, PathsConfig, WomblexConfig

#: Extraction is not itself a `StageContract` — it runs inside
#: `process_batch`, not `run-stage` — but every stage's element-stream
#: inputs come from it, so the graph needs a source node for those edges to
#: point at.
EXTRACT_NODE = "extract"

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
    inputs ride along on each node instead of adding edges of their own:
    they are `Callable[[WomblexConfig], ...]`, so an edge for one would only
    be true for whatever config the composer form happens to hold at that
    moment, and a graph that changes shape as an operator edits a batch size
    would be harder to read than a fixed structure annotated with what a
    stage might additionally read.
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
        }
    ]
    edges: list[dict[str, str]] = []
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
            }
        )
        for suffix in contract.required_inputs:
            producer = _producer(suffix)
            if producer is not None:
                edges.append({"from": producer, "to": name, "suffix": suffix})
    return {"nodes": nodes, "edges": edges}


def get_config_schema() -> dict[str, Any]:
    """`WomblexConfig`'s JSON Schema — the composer form's field list, straight
    from Pydantic. No hand-typed mirror of `config.py` to fall out of sync."""
    return WomblexConfig.model_json_schema()


def validate_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Try to build a `WomblexConfig` from *raw*; report Pydantic's own errors.

    Uses `WomblexConfig(**raw)` — the same construction `load_config` uses —
    so a config the composer accepts is one the CLI accepts too, and there is
    no separate composer-side notion of validity.
    """
    try:
        WomblexConfig(**raw)
    except ValidationError as e:
        errors = e.errors(include_url=False, include_context=False, include_input=False)
        return {"valid": False, "errors": errors}
    return {"valid": True, "errors": []}


def render_yaml(raw: dict[str, Any]) -> str:
    """Validate *raw* and render it back as YAML, in the shape `load_config` reads.

    Round-trips through the validated model (`model_dump(mode="json")`)
    rather than dumping the posted dict verbatim, so the download always
    reflects Pydantic-applied defaults and coercions — what a fresh
    `womblex run --config <this file>` would actually see, not whatever the
    browser happened to send.

    Raises `pydantic.ValidationError` on an invalid config; the route
    translates that to a 422 with the same error shape `validate_config`
    returns.
    """
    config = WomblexConfig(**raw)
    return str(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))
