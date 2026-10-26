"""`/v1/collections`: the collections a caller's stored workflows feed, and adding to or rerunning one.

Owner-scoped like `/v1/runs`: a caller sees and acts on collections its own stored workflows feed
(Requirement 34); another owner's is 404, and ``admin`` sees all. Reading is ``read``; adding and
rerunning are ``submit``. Both act on a stored workflow and neither creates, edits nor removes one.
A collection fed by several workflows needs the workflow named (409 otherwise). A refusal is 409
with the workflow refusal's stable code, before any queue row. The bodies are
:mod:`womblex.cloud.collections`.
"""
from __future__ import annotations

from collections.abc import Callable

import sqlalchemy.exc
from fastapi import APIRouter, Depends, HTTPException

from womblex.api.auth import Caller
from womblex.api.models import (
    CollectionAction,
    CollectionAdded,
    CollectionDetail,
    CollectionList,
    CollectionRerun,
    CollectionSummary,
)
from womblex.cloud import collections as coll
from womblex.cloud.deployment import DeploymentSettings
from womblex.cloud.jobs import RunOwnedError
from womblex.cloud.workflow_runner import WorkflowRefusedError
from womblex.cloud.workflow_store import StoredWorkflow
from womblex.presets import get_any_preset
from womblex.store.remote_manifest import StoreUnreachable


def collections_router(
    settings: DeploymentSettings, can_read: Callable[..., Caller], can_submit: Callable[..., Caller],
) -> APIRouter:
    router = APIRouter(tags=["collections"])

    def visible(caller: Caller) -> dict[str, list[StoredWorkflow]]:
        try:
            return coll.stored_collections(settings.db_dsn, caller.owner)
        except (sqlalchemy.exc.SQLAlchemyError, OSError) as e:
            raise HTTPException(status_code=503, detail="job queue unreachable") from e

    def feeding(caller: Caller, name: str) -> list[StoredWorkflow]:
        if (found := visible(caller).get(name)) is None:
            raise HTTPException(status_code=404, detail=f"collection not found: {name}")
        return found

    def chosen(caller: Caller, name: str, body: CollectionAction) -> StoredWorkflow:
        candidates = feeding(caller, name)
        try:
            stored = coll.pick_workflow(candidates, body.workflow)
        except coll.AmbiguousWorkflow as e:
            raise HTTPException(status_code=409, detail={
                "code": "ambiguous_workflow", "message": str(e), "workflows": e.names}) from e
        if stored is None:
            raise HTTPException(status_code=404, detail=f"workflow not found in collection {name}: {body.workflow}")
        return stored

    def resolve(preset: str):  # type: ignore[no-untyped-def]
        return get_any_preset(settings, preset)

    def refused(e: WorkflowRefusedError | RunOwnedError) -> HTTPException:
        code = e.code.value if isinstance(e, WorkflowRefusedError) else "run_owned"
        return HTTPException(status_code=409, detail={"code": code, "message": str(e)})

    @router.get("/v1/collections", response_model=CollectionList)
    def list_collections(caller: Caller = Depends(can_read)) -> CollectionList:  # noqa: B008
        found = visible(caller)
        return CollectionList(collections=[
            CollectionSummary(name=n, workflows=sorted(w.id for w in ws)) for n, ws in sorted(found.items())])

    @router.get("/v1/collections/{name}", response_model=CollectionDetail)
    def get_collection(name: str, caller: Caller = Depends(can_read)) -> CollectionDetail:  # noqa: B008
        workflows = feeding(caller, name)
        try:
            documents, current, written_by = coll.collection_documents(settings.store_uri or "", name)
        except StoreUnreachable as e:
            raise HTTPException(status_code=503, detail=f"run store unreachable: {e}") from e
        return CollectionDetail(name=name, workflows=sorted(w.id for w in workflows),
                                documents=documents, current_documents=current, written_by=written_by)

    @router.post("/v1/collections/{name}/add", response_model=CollectionAdded, status_code=202)
    def add_to_collection(
        name: str, body: CollectionAction | None = None, caller: Caller = Depends(can_submit),  # noqa: B008
    ) -> CollectionAdded:
        stored = chosen(caller, name, body or CollectionAction())
        try:
            firing = coll.add_documents(settings, stored, resolve)
        except (WorkflowRefusedError, RunOwnedError) as e:
            raise refused(e) from e
        return CollectionAdded(name=name, workflow=stored.id, firing=firing)

    @router.post("/v1/collections/{name}/rerun", response_model=CollectionRerun, status_code=202)
    def rerun_collection(
        name: str, body: CollectionAction | None = None, caller: Caller = Depends(can_submit),  # noqa: B008
    ) -> CollectionRerun:
        stored = chosen(caller, name, body or CollectionAction())
        try:
            sent = coll.rerun_documents(settings, stored, resolve)
        except (WorkflowRefusedError, RunOwnedError) as e:
            raise refused(e) from e
        return CollectionRerun(name=name, workflow=stored.id, stages=list(sent.stages),
                               newly_dispatched=sent.newly_enqueued)

    return router
