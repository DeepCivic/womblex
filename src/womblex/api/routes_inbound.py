"""`POST /v1/hooks/{workflow}/{hook}`: an inbound webhook (Requirement 35).

Authentication is the hook's own HMAC over the raw body (`X-Womblex-Signature`), not the bearer
token: the sender is another system holding the hook's key. ``--insecure-no-auth`` does not skip
it; only ``WOMBLEX_INSECURE_INBOUND=1`` does. A body over ``inbound.MAX_BODY_BYTES`` is 413 before
any check. The body of the work is :mod:`womblex.cloud.inbound`.
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from womblex.cloud import inbound
from womblex.cloud.deployment import DeploymentSettings
from womblex.cloud.webhook import SIGNATURE_HEADER
from womblex.presets import get_any_preset


def inbound_router(settings: DeploymentSettings) -> APIRouter:
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

    return router
