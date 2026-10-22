"""`/v1/presets`: the built-in and saved presets a run or workflow starts from.

Reads need ``read``. Saved presets are the deployment's, shared by every
caller, so saving and deleting need ``admin``. The bodies are
:mod:`womblex.presets`; these routes map them onto HTTP.
"""
from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError

from womblex import presets
from womblex.api.auth import Caller
from womblex.api.models import PresetDeleted, PresetList, PresetOut, PresetSave
from womblex.cloud.deployment import DeploymentSettings
from womblex.config_tools import validation_errors
from womblex.store.remote_manifest import StoreUnreachable


def presets_router(
    settings: DeploymentSettings, can_read: Callable[..., Caller], can_admin: Callable[..., Caller],
) -> APIRouter:
    router = APIRouter(tags=["presets"])

    def unreachable(e: StoreUnreachable) -> HTTPException:
        return HTTPException(status_code=503, detail=f"preset store unreachable: {e}")

    @router.get("/v1/presets", response_model=PresetList, dependencies=[Depends(can_read)])
    def list_presets() -> PresetList:
        try:
            return PresetList(presets=[PresetOut(**p) for p in presets.list_all_presets(settings)])
        except StoreUnreachable as e:
            raise unreachable(e) from e

    @router.get("/v1/presets/{name}", response_model=PresetOut, dependencies=[Depends(can_read)])
    def get_preset(name: str) -> PresetOut:
        try:
            found = presets.get_any_preset(settings, name)
        except StoreUnreachable as e:
            raise unreachable(e) from e
        if found is None:
            raise HTTPException(status_code=404, detail=f"no such preset: {name}")
        return PresetOut(**found.as_dict())

    @router.post("/v1/presets", response_model=PresetOut, status_code=201, dependencies=[Depends(can_admin)])
    def save_preset(body: PresetSave) -> PresetOut:
        """Save or replace a preset; 400 on an unsafe name or a config that would not load."""
        try:
            saved = presets.save_preset(settings, name=body.name, description=body.description,
                                        formats=tuple(body.formats), config=body.config)
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=validation_errors(e)) from e
        except presets.PresetsNotWritable as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        except StoreUnreachable as e:
            raise unreachable(e) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return PresetOut(**saved.as_dict())

    @router.delete("/v1/presets/{name}", response_model=PresetDeleted, dependencies=[Depends(can_admin)])
    def delete_preset(name: str) -> PresetDeleted:
        """404 when no saved preset has *name*; a built-in is code and is never deleted."""
        try:
            removed = presets.delete_saved_preset(settings, name)
        except presets.PresetsNotWritable as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        except StoreUnreachable as e:
            raise unreachable(e) from e
        if not removed:
            raise HTTPException(status_code=404, detail=f"no such saved preset: {name}")
        return PresetDeleted(deleted=name)

    return router
