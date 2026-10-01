"""FastAPI app factory for the `/v1` service API (service plan B4a; submission and `/files` follow).

Separate from the console: it binds one shared store and queue for its
lifetime, authenticates every `/v1` route with a service token
(:mod:`womblex.api.auth`), and scopes runs to the caller that submitted them —
a run another client owns is a 404, and ``admin`` sees all. A run is known by
the queue, so the CLI- and console-submitted runs (no owner) are admin-only.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from fastapi import Depends, FastAPI, HTTPException, Response

from womblex.api.auth import Caller, ClientRegistry, caller_dependency, require_scope
from womblex.api.models import (
    Health,
    Ready,
    RunList,
    RunManifest,
    RunMetrics,
    RunStatus,
)
from womblex.cloud.dispatch import QUEUE_CONNECT_TIMEOUT
from womblex.cloud.queue import JobQueue, RunSummary
from womblex.ui import readers as ui_readers
from womblex.ui import resources
from womblex.ui.deps import UISettings

logger = logging.getLogger(__name__)

#: How many runs one caller's listing returns.
RUN_LIMIT = 1000


def run_state(counts: dict[str, int]) -> str:
    """Collapse a status rollup into one state; any failed job fails the run."""
    if counts.get("failed"):
        return "failed"
    if counts.get("running") or (counts.get("pending") and counts.get("done")):
        return "running"
    return "pending" if counts.get("pending") else "done"


def _status(run: RunSummary) -> RunStatus:
    return RunStatus(
        run_id=run.run_id, state=run_state(run.counts), counts=run.counts,  # type: ignore[arg-type]
        total=run.total, created_at=run.created_at, updated_at=run.updated_at,
    )


def create_api_app(
    *,
    store_uri: str,
    db_dsn: str,
    registry: ClientRegistry | None,
) -> FastAPI:
    """Build the API, bound to one store and queue for its lifetime.

    ``registry=None`` is the ``--insecure-no-auth`` development mode; the
    ``serve`` command, not this factory, refuses an empty registry.
    """
    settings = UISettings(output_root=None, store_uri=store_uri, db_dsn=db_dsn)
    app = FastAPI(title="Womblex API", version="1")
    get_caller = caller_dependency(registry)
    can_read = require_scope(get_caller, "read")

    @contextmanager
    def queue() -> Iterator[JobQueue]:
        try:
            q = JobQueue(db_dsn, connect_timeout=QUEUE_CONNECT_TIMEOUT)
        except Exception as e:
            logger.warning("api: queue unreachable: %s", e)
            raise HTTPException(status_code=503, detail="job queue unreachable") from e
        with q:
            yield q

    def find_run(q: JobQueue, caller: Caller, run_id: str) -> RunSummary:
        found = q.runs(caller.owner, run_id=run_id, limit=1)
        if found:
            return found[0]
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}")

    @app.get("/v1/health", response_model=Health, tags=["ops"])
    def health() -> Health:
        return Health()

    @app.get("/v1/ready", response_model=Ready, tags=["ops"])
    def ready(response: Response) -> Ready:
        store_ok = bool(resources.test_store(settings)["reachable"])
        queue_ok = bool(resources.test_queue(settings)["reachable"])
        ok = store_ok and queue_ok
        if not ok:
            response.status_code = 503
        return Ready(ready=ok, store=store_ok, queue=queue_ok)

    @app.get("/v1/runs", response_model=RunList, tags=["runs"])
    def list_runs(caller: Caller = Depends(can_read)) -> RunList:  # noqa: B008
        with queue() as q:
            return RunList(runs=[_status(r) for r in q.runs(caller.owner, limit=RUN_LIMIT)])

    @app.get("/v1/runs/{run_id}", response_model=RunStatus, tags=["runs"])
    def get_run(run_id: str, caller: Caller = Depends(can_read)) -> RunStatus:  # noqa: B008
        with queue() as q:
            return _status(find_run(q, caller, run_id))

    @app.get("/v1/runs/{run_id}/manifest", response_model=RunManifest, tags=["runs"])
    def get_manifest(run_id: str, caller: Caller = Depends(can_read)) -> RunManifest:  # noqa: B008
        with queue() as q:
            find_run(q, caller, run_id)
        try:
            rows = ui_readers.get_manifest_rows(settings, run_id)
        except ui_readers.StoreUnreachable as e:
            raise HTTPException(status_code=503, detail=f"run store unreachable: {e}") from e
        return RunManifest(run_id=run_id, documents=rows or [])

    @app.get("/v1/runs/{run_id}/metrics", response_model=RunMetrics, tags=["runs"])
    def get_metrics(run_id: str, caller: Caller = Depends(can_read)) -> RunMetrics:  # noqa: B008
        with queue() as q:
            find_run(q, caller, run_id)
            return RunMetrics(
                run_id=run_id,
                stats=q.stats(run_id, owner=caller.owner),
                workers=[w.__dict__ for w in q.workers(run_id, owner=caller.owner)],
                throughput=q.throughput(run_id, owner=caller.owner).__dict__,
            )

    return app
