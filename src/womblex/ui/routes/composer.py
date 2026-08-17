"""``/api/composer`` — the Pipeline Composer's graph, form schema, presets and
config validation (docs/ui-plan.md merge 9).

Reads no run state — `config.py`, `cloud/stage_contracts.py` and the preset
registry are static, so unlike every other route here this one takes no
`UISettings` dependency. `POST /validate` and `POST /yaml` are not a second
writable surface (feedback stays the console's only one, plan §4): neither
touches a run or the filesystem, they just build a `WomblexConfig` in memory
and hand back what came of it. `GET /presets` serves named pre-configured
pipelines (e.g. `DEFAULT-Isaacus`) the form loads as starting points.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Response
from pydantic import ValidationError

from womblex.ui import composer, presets

router = APIRouter(prefix="/api/composer", tags=["composer"])


@router.get("/graph")
def get_graph() -> dict:
    """The pipeline DAG `STAGE_CONTRACTS` implies — nodes, required-input edges."""
    return composer.get_stage_graph()


@router.get("/presets")
def get_presets() -> dict:
    """Named pre-configured pipelines the form offers as starting points.

    Each is a *partial* `WomblexConfig` (no `dataset` / `paths` — the operator
    still supplies the run's identity and paths). `DEFAULT-Isaacus` is the
    reference extract → chunk → enrich → build_graph → money shape.
    """
    return {"presets": presets.list_presets()}


@router.get("/presets/{name}")
def get_preset(name: str) -> dict:
    """One preset by name; 404 if there is no such preset."""
    preset = presets.get_preset(name)
    if preset is None:
        raise HTTPException(status_code=404, detail=f"no such preset: {name!r}")
    return preset.as_dict()


@router.get("/schema")
def get_schema() -> dict:
    """`WomblexConfig`'s JSON Schema, for the composer form to render fields from."""
    return composer.get_config_schema()


@router.post("/validate")
def post_validate(raw: dict[str, Any]) -> dict:
    """Whether *raw* builds a valid `WomblexConfig`, plus Pydantic's own errors.

    `unknown_keys` lists submitted keys the schema does not claim. These do
    not make a config invalid — the CLI ignores them too — but they are the
    one mistake a config editor must never swallow silently, since a typo'd
    key validates clean and then vanishes from the rendered YAML.
    """
    return composer.validate_config(raw)


@router.post("/yaml")
def post_yaml(raw: dict[str, Any]) -> Response:
    """Validate *raw* and return it as a downloadable YAML file.

    422 with Pydantic's own errors on an invalid config — the same shape
    `/validate` reports — there is no separate "download anyway" path for a
    config the CLI would reject. Keys the schema does not claim are dropped
    (as `load_config` would drop them) but named in a comment header, so the
    file itself records what it lost.
    """
    try:
        text = composer.render_yaml(raw)
    except ValidationError as e:
        detail = e.errors(include_url=False, include_context=False, include_input=False)
        raise HTTPException(status_code=422, detail=detail) from e
    return Response(
        content=text,
        media_type="application/yaml",
        headers={"Content-Disposition": "attachment; filename=womblex.yaml"},
    )
