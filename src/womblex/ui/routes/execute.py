"""``/api/execute`` — the Execution Controls (docs/ui-plan.md merge 11).

The one writable-to-a-run surface in the console. ``GET /status`` is a
cheap, network-free read of whether this deployment can dispatch work (and
if not, why); ``POST /enqueue`` plans an extraction run into the queue. Both
delegate to :mod:`womblex.ui.execute`, which enforces the ``--audit-only``
switch and the store+queue requirement (plan §4) before touching anything —
this router only maps its :class:`~womblex.ui.execute.ExecutionDisabled`
reasons onto HTTP status codes.

Per-stage dispatch and the run/log feed are the queue's own views the
Dashboard already serves (``/api/dashboard``); nothing is duplicated here.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from womblex.ui import execute
from womblex.ui.deps import UISettings, get_settings

router = APIRouter(prefix="/api/execute", tags=["execute"])

#: `ExecutionDisabled.reason` -> HTTP status. An audit-only deployment is a
#: deliberate choice (403 Forbidden); a missing store/queue is a wiring gap on
#: an otherwise willing console (409 Conflict — the request cannot be served in
#: the console's current state). Kept as data so a new reason must be mapped
#: explicitly rather than defaulting to a misleading 403.
_REASON_STATUS: dict[str, int] = {
    "execute_disabled": 403,
    "no_store": 409,
    "no_ingest": 409,
    "no_queue": 409,
}


class EnqueueRequest(BaseModel):
    """The configure-and-run form: which documents, at what run id and batching.

    ``input_prefix`` is ingest-relative (the enqueue lists
    ``<ingest_uri>/<prefix>``), matching ``womblex enqueue --input-prefix``.
    Optional: omitted means the whole configured ingest root, which is the
    normal case (docs/ui-ingest-plan.md §2 "no prefix field on any screen").
    ``run_id`` omitted mints a fresh timestamped id; supplying an existing
    one resumes it, since enqueue is idempotent on ``(run_id, batch_num)``.
    """

    input_prefix: str | None = None
    run_id: str | None = None
    batch_size: int = Field(50, ge=1)
    max_attempts: int = Field(3, ge=1)


@router.get("/status")
def get_status(settings: UISettings = Depends(get_settings)) -> dict:  # noqa: B008 - FastAPI DI idiom
    """Whether the console can dispatch work, and which piece is missing if not."""
    return execute.execution_status(settings).as_dict()


@router.get("/ingest")
def get_ingest(settings: UISettings = Depends(get_settings)) -> dict:  # noqa: B008
    """Reachability + document count of the configured ingest location.

    Feeds the composer's "N documents ready" line before it enqueues.
    """
    return execute.ingest_preflight(settings)


@router.post("/enqueue")
def post_enqueue(
    body: EnqueueRequest, settings: UISettings = Depends(get_settings),  # noqa: B008
) -> dict:
    """Plan an extraction run into the queue; workers the platform brings up run it.

    403 when the console is audit-only, 409 when no store, ingest location
    or queue is configured (the states :class:`ExecutionDisabled`
    distinguishes), 400 on bad input (no documents under the prefix, an
    unsafe run_id).
    """
    try:
        result = execute.enqueue_extraction(
            settings,
            input_prefix=body.input_prefix,
            run_id=body.run_id,
            batch_size=body.batch_size,
            max_attempts=body.max_attempts,
        )
    except execute.ExecutionDisabled as e:
        raise HTTPException(status_code=_REASON_STATUS[e.reason], detail=e.detail) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return result.as_dict()
