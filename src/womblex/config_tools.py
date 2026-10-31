"""A config as a caller composes it: its JSON Schema, its validation, its YAML.

Each goes through the `WomblexConfig(**raw)` construction `load_config` uses,
so nothing composed here can be accepted that the CLI would reject. `paths`
is the deployment's, never the caller's: the schema omits it and validation
and rendering inject `deployment_paths(settings)`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import yaml
from pydantic import ValidationError

from womblex.cloud.deployment import deployment_paths
from womblex.config import WomblexConfig

if TYPE_CHECKING:
    from womblex.cloud.deployment import DeploymentSettings


def config_schema() -> dict[str, Any]:
    """`WomblexConfig`'s JSON Schema without `paths`."""
    schema = WomblexConfig.model_json_schema()
    schema.get("properties", {}).pop("paths", None)
    required = schema.get("required")
    if isinstance(required, list) and "paths" in required:
        required.remove("paths")
    return schema


def validation_errors(e: ValidationError) -> list[dict[str, Any]]:
    """Pydantic's errors without URLs, context or the posted input."""
    return [dict(err) for err in e.errors(include_url=False, include_context=False, include_input=False)]


def validate_config(raw: dict[str, Any], settings: DeploymentSettings) -> dict[str, Any]:
    """``{"valid": bool, "errors": [...]}`` for *raw* under this deployment's paths."""
    paths, _ = deployment_paths(settings)
    try:
        WomblexConfig(**{**raw, "paths": paths})
    except ValidationError as e:
        return {"valid": False, "errors": validation_errors(e)}
    return {"valid": True, "errors": []}


def render_yaml(raw: dict[str, Any], settings: DeploymentSettings) -> str:
    """*raw* validated and rendered as the YAML `load_config` reads, defaults applied.

    An object-store location is a placeholder in `paths`, and a header comment
    names the environment variable that sets it. Raises `ValidationError`.
    """
    paths, env_vars = deployment_paths(settings)
    config = WomblexConfig(**{**raw, "paths": paths})
    body = str(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))
    if not env_vars:
        return body
    header = ("# paths.input_root/output_root above are placeholders — this deployment's "
              "storage is configured via " + " / ".join(env_vars) + ", not written here.")
    return f"{header}\n\n{body}"
