"""The job queue in one read: counts, jobs, workers, stale jobs and throughput.

One connection for every view, scoped to a run and an owner when given. A
reachable database without the DBOS tables yet is an empty queue, not a
fault: nothing has been enqueued there. Any other failure raises.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from womblex.cloud.dispatch import QUEUE_CONNECT_TIMEOUT
from womblex.cloud.reachability import is_missing_schema

#: A ``running`` job not updated for this many seconds is listed as stale; DBOS recovers it when its executor restarts.
DEFAULT_STALE_AFTER = 900.0
#: The trailing window throughput is measured over, in seconds.
DEFAULT_THROUGHPUT_WINDOW = 3600.0


def empty_snapshot(window_seconds: float = DEFAULT_THROUGHPUT_WINDOW) -> dict[str, Any]:
    return {
        "stats": {}, "total": 0, "summary": None, "jobs": [], "workers": [], "stale": [],
        "throughput": {"window_seconds": window_seconds, "completed": 0, "per_minute": 0.0, "last_completed_at": None},
    }


def queue_snapshot(
    db_dsn: str | None,
    *,
    run_id: str | None = None,
    owner: str | None = None,
    stale_after: float = DEFAULT_STALE_AFTER,
    window_seconds: float = DEFAULT_THROUGHPUT_WINDOW,
    job_limit: int = 200,
) -> dict[str, Any]:
    """Every queue view for *run_id* (or all runs) and *owner* (or every owner).

    ``summary`` is the run's status sentence when *run_id* names one. ``stale``
    carries whole rows: the oldest updates are the first to fall outside
    ``job_limit``, so ids alone would not resolve.
    """
    from womblex.cloud.jobs import RunBoard

    with RunBoard(db_dsn, connect_timeout=QUEUE_CONNECT_TIMEOUT) as board:
        try:
            stats = board.stats(run_id, owner=owner)
            found = board.runs(owner, run_id=run_id, limit=1) if run_id else []
            jobs = board.list_jobs(run_id, limit=job_limit, owner=owner)
            workers = board.workers(run_id, owner=owner)
            stale = board.stale_jobs(stale_after, run_id, owner=owner)
            throughput = board.throughput(run_id, window_seconds=window_seconds, owner=owner)
        except Exception as e:
            if is_missing_schema(e):
                return empty_snapshot(window_seconds)
            raise
    return {
        "stats": stats,
        "total": sum(stats.values()),
        "summary": found[0].sentence if found else None,
        "jobs": [asdict(j) for j in jobs],
        "workers": [asdict(w) for w in workers],
        "stale": [asdict(j) for j in stale],
        "throughput": asdict(throughput),
    }
