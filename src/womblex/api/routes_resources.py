"""`/v1/resources`: what this deployment is wired to, and a live check of each piece.

``admin``: it describes the deployment, not a caller's runs. Credentials are
masked by :mod:`womblex.cloud.resources`; a check answers with the reason it
failed rather than an error status.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from fastapi import APIRouter, Depends

from womblex.api.auth import Caller
from womblex.api.models import Reachability, Resources
from womblex.cloud import resources
from womblex.cloud.deployment import DeploymentSettings
from womblex.cloud.reachability import queue_reachable


def resources_router(settings: DeploymentSettings, can_admin: Callable[..., Caller]) -> APIRouter:
    router = APIRouter(tags=["ops"], dependencies=[Depends(can_admin)])

    @router.get("/v1/resources", response_model=Resources)
    def get_resources() -> Resources:
        """The store, ingest location, queue and Isaacus deployment, read without a network call."""
        return Resources(**resources.deployment_resources(settings))

    @router.post("/v1/resources/test/{target}", response_model=Reachability)
    def test(target: Literal["store", "ingest", "queue"]) -> Reachability:
        if target == "store":
            ok, error = resources.check_store(settings)
        elif target == "ingest":
            ok, error = resources.check_ingest(settings)
        else:
            ok, error = queue_reachable(settings.db_dsn)
        return Reachability(target=target, reachable=ok, error=error)

    return router
