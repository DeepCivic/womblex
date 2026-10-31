"""`/v1/queue`: the job queue in one read, for a caller watching its runs drain.

``read``, scoped to the caller's own runs (``admin`` sees every run). The
body is :mod:`womblex.cloud.queue_snapshot`; a queue that cannot be read is 503.
"""
from __future__ import annotations

import logging
from collections.abc import Callable

from fastapi import APIRouter, Depends, HTTPException, Query

from womblex.api.auth import Caller
from womblex.api.models import QueueSnapshot
from womblex.cloud.queue_snapshot import DEFAULT_STALE_AFTER, DEFAULT_THROUGHPUT_WINDOW, queue_snapshot

logger = logging.getLogger(__name__)


def queue_router(db_dsn: str, can_read: Callable[..., Caller]) -> APIRouter:
    router = APIRouter(tags=["runs"])

    @router.get("/v1/queue", response_model=QueueSnapshot)
    def get_queue(
        run_id: str | None = None,
        stale_after: float = Query(DEFAULT_STALE_AFTER, gt=0),
        window_seconds: float = Query(DEFAULT_THROUGHPUT_WINDOW, gt=0),
        job_limit: int = Query(200, ge=1, le=1000),
        caller: Caller = Depends(can_read),  # noqa: B008
    ) -> QueueSnapshot:
        try:
            snapshot = queue_snapshot(db_dsn, run_id=run_id, owner=caller.owner, stale_after=stale_after,
                                      window_seconds=window_seconds, job_limit=job_limit)
        except Exception as e:
            logger.warning("api: queue unreadable: %s", e)
            raise HTTPException(status_code=503, detail="job queue unreachable") from e
        return QueueSnapshot(run_id=run_id, stale_after_seconds=stale_after, **snapshot)

    return router
