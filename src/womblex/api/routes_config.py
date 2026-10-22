"""`/v1/stages` and `/v1/config/*`: the pipeline's shape and a config's schema, validation and YAML.

Static reads, not run-scoped: a caller composing a config before it submits
one. The bodies are the library's (:mod:`womblex.cloud.stage_graph`,
:mod:`womblex.config_tools`); these routes only map them onto HTTP.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import ValidationError

from womblex import config_tools
from womblex.api.auth import Caller
from womblex.api.models import ConfigValidation, StageGraph
from womblex.cloud.deployment import DeploymentSettings
from womblex.cloud.stage_graph import stage_graph


def config_router(settings: DeploymentSettings, can_read: Callable[..., Caller]) -> APIRouter:
    router = APIRouter(tags=["config"], dependencies=[Depends(can_read)])

    @router.get("/v1/stages", response_model=StageGraph)
    def get_stages() -> StageGraph:
        return StageGraph(**stage_graph())

    @router.get("/v1/config/schema")
    def get_schema() -> dict[str, Any]:
        """`WomblexConfig`'s JSON Schema; ``paths`` is the deployment's and is left out."""
        return config_tools.config_schema()

    @router.post("/v1/config/validate", response_model=ConfigValidation)
    def validate(raw: dict[str, Any]) -> ConfigValidation:
        return ConfigValidation(**config_tools.validate_config(raw, settings))

    @router.post("/v1/config/yaml", response_class=Response)
    def to_yaml(raw: dict[str, Any]) -> Response:
        """The config as a YAML file `womblex run --config` reads; 422 with the validation errors otherwise."""
        try:
            text = config_tools.render_yaml(raw, settings)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=config_tools.validation_errors(e)) from e
        return Response(content=text, media_type="application/yaml",
                        headers={"Content-Disposition": "attachment; filename=womblex.yaml"})

    return router
