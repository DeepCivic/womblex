"""`POST /v1/hooks/{workflow}/{hook}`: an inbound webhook, and `GET /v1/workflows/{id}/inbound-events` (Requirement 35).

Authentication is the hook's own HMAC over the raw body (`X-Womblex-Signature`), not the bearer
token: the sender is another system holding the hook's key. ``--insecure-no-auth`` does not skip
it; only ``WOMBLEX_INSECURE_INBOUND=1`` does. A body over ``inbound.MAX_BODY_BYTES`` is 413 before
any check. The body of the work is :mod:`womblex.cloud.inbound`.
"""
from __future__ import annotations

from collections.abc import Callable

import sqlalchemy.exc
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from womblex.api.auth import Caller
from womblex.api.models import InboundEventList, InboundEventOut
from womblex.cloud import inbound
from womblex.cloud.deployment import DeploymentSettings
from womblex.cloud.inbound_store import InboundEvents
from womblex.cloud.jobs import schema_missing
from womblex.cloud.webhook import SIGNATURE_HEADER
from womblex.cloud.workflow_store import WorkflowStore
from womblex.presets import get_any_preset


def inbound_router(settings: DeploymentSettings, can_read: Callable[..., Caller]) -> APIRouter:
    router = APIRouter(tags=["hooks"])

    @router.post("/v1/hooks/{workflow}/{hook}", status_code=202, response_model=None)
    async def receive(workflow: str, hook: str, request: Request) -> JSONResponse:
        too_large = inbound.refuse(413, "body_too_large", f"the body is over {inbound.MAX_BODY_BYTES} bytes")
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > inbound.MAX_BODY_BYTES:
            return JSONResponse(status_code=too_large.status, content=too_large.body)
        raw = bytearray()
        async for chunk in request.stream():  # a body without a truthful length is cut off as it arrives
            raw += chunk
            if len(raw) > inbound.MAX_BODY_BYTES:
                return JSONResponse(status_code=too_large.status, content=too_large.body)
        answer = await run_in_threadpool(
            inbound.handle_event, settings, workflow, hook, bytes(raw), request.headers.get(SIGNATURE_HEADER),
            lambda name: get_any_preset(settings, name),
        )
        return JSONResponse(status_code=answer.status, content=answer.body)

    @router.get("/v1/workflows/{workflow}/inbound-events", response_model=InboundEventList)
    def list_events(
        workflow: str, limit: int = Query(100, ge=1, le=1000), caller: Caller = Depends(can_read),  # noqa: B008
    ) -> InboundEventList:
        """The workflow's inbound events, newest first, owner-scoped: another owner's workflow is 404."""
        try:
            with WorkflowStore(settings.db_dsn) as wf_store:
                stored = wf_store.get(workflow)
            if stored is not None and caller.owner is not None and stored.owner != caller.owner:
                stored = None
            if stored is None:
                raise HTTPException(status_code=404, detail=f"workflow not found: {workflow}")
            with InboundEvents(settings.db_dsn) as events:
                rows = events.list(workflow, limit=limit)
        except sqlalchemy.exc.DBAPIError as e:
            if not schema_missing(e):
                raise HTTPException(status_code=503, detail="job queue unreachable") from e
            raise HTTPException(status_code=404, detail=f"workflow not found: {workflow}") from e
        prefix = f"{workflow}:"
        return InboundEventList(workflow=workflow, events=[
            InboundEventOut(event_id=r.event_id.removeprefix(prefix + r.hook + ":"), hook=r.hook, event=r.event,
                            received_at=r.received_at,
                            outcome=None if r.outcome is None else {**r.outcome["body"], "http_status": r.outcome["http_status"]})
            for r in rows])

    return router
