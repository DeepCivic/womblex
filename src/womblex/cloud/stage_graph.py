"""The pipeline's shape as nodes and edges, derived from `STAGE_CONTRACTS`.

The contracts already are the pipeline's DAG as data; this renders them so a
caller drawing the pipeline never hand-codes one. Edges come from each
stage's `required_inputs` only, the hard ordering; config-derived inputs ride
on the node, since an edge for one would hold only for one config.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from womblex.cloud.stage_contracts import DISCOVERY_SUFFIXES, PRODUCER_OF, STAGE_CONTRACTS, STAGE_NAMES
from womblex.config import DatasetConfig, PathsConfig, WomblexConfig

#: Extraction is not a `StageContract` (it runs inside `process_batch`), but
#: every element-stream input comes from it, so it is the graph's source node.
EXTRACT_NODE = "extract"

#: The `WomblexConfig` section that configures each stage. Contracts name
#: parquet suffixes, not config fields, so this is declared. `graph-refresh`
#: and `layout` have no section with an `enabled` toggle; `extract` is
#: configured by `detection`, `extraction` and `redaction` together.
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

#: Schema defaults, used only to resolve the contracts' config-derived inputs and outputs.
_DEFAULT_CONFIG = WomblexConfig(
    dataset=DatasetConfig(name="stage-graph"),
    paths=PathsConfig(input_root=Path("."), output_root=Path("."), checkpoint_dir=Path(".")),
)


def _producer(suffix: str) -> str | None:
    if suffix in DISCOVERY_SUFFIXES:
        return EXTRACT_NODE
    return PRODUCER_OF.get(suffix)


def stage_graph() -> dict[str, Any]:
    """`{"nodes": […], "edges": […]}` at schema defaults; one edge per ordered pair, with its suffixes."""
    nodes: list[dict[str, Any]] = [{
        "id": EXTRACT_NODE, "scope": None, "mutation": None, "needs_isaacus_api": False,
        "checkpoint_dirname": None, "required_inputs": [], "conditional_inputs": [],
        "outputs": list(DISCOVERY_SUFFIXES), "config_section": None,
    }]
    edges: dict[tuple[str, str], list[str]] = {}
    for name in STAGE_NAMES:
        contract = STAGE_CONTRACTS[name]
        nodes.append({
            "id": name,
            "scope": contract.scope.value,
            "mutation": contract.mutation.value,
            "needs_isaacus_api": contract.needs_isaacus_api,
            "checkpoint_dirname": contract.checkpoint_dirname,
            "required_inputs": list(contract.required_inputs),
            "conditional_inputs": [
                {"suffix": c.suffix, "reason": c.reason, "strict": c.strict}
                for c in contract.conditional_inputs(_DEFAULT_CONFIG)
            ],
            "outputs": list(contract.outputs(_DEFAULT_CONFIG)),
            "config_section": CONFIG_SECTION.get(name),
        })
        for suffix in contract.required_inputs:
            producer = _producer(suffix)
            if producer is not None:
                edges.setdefault((producer, name), []).append(suffix)
    return {
        "nodes": nodes,
        "edges": [{"from": p, "to": c, "suffixes": s} for (p, c), s in edges.items()],
    }
