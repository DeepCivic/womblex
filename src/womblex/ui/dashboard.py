"""The Dashboard's read model (docs/ui-plan.md merge 8).

Two sources, both already written by the pipeline:

- **The job queue**, when a DSN is configured. Queue counts are exact, the
  job list is ``womblex_jobs`` itself, the fleet is ``locked_by`` on running
  rows, and throughput is derived from ``updated_at`` — no new columns.
- **Per-stage checkpoints**, always. Every shard stage writes its
  ``CheckpointState`` to a dot-directory *inside the run*
  (``<run>/.chunk-checkpoint/`` and friends), which ``STAGE_CONTRACTS``
  already names via ``checkpoint_dirname``. That makes stage progress
  readable from the run the console is already pointed at, in both
  deployments, with no configuration of its own.

Separate from :mod:`womblex.ui.readers` because none of this is a parquet
sidecar read: the queue is Postgres and the checkpoints are JSON.
"""
from __future__ import annotations

import logging
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, cast

from womblex.cloud.stage_contracts import STAGE_CONTRACTS
from womblex.store.checkpoint import CHECKPOINT_GLOB, CheckpointProgress, read_checkpoints
from womblex.store.feedback_output import is_safe_run_id
from womblex.ui.deps import UISettings

if TYPE_CHECKING:
    from womblex.store.remote import RemoteStore

logger = logging.getLogger(__name__)

#: Default staleness threshold, in seconds. A ``running`` row locked longer
#: than this is what ``requeue_stale`` recovers, so the dashboard names the
#: same rows a worker's ``--stale-timeout`` would act on.
DEFAULT_STALE_AFTER = 900.0

#: Default trailing window for the throughput tile, in seconds.
DEFAULT_THROUGHPUT_WINDOW = 3600.0

#: Seconds to wait for the queue connection before reporting it unreachable.
#: The dashboard is polled, so an unbounded connect to a routable-but-dead
#: host would pin a request thread per poll until the OS gave up.
QUEUE_CONNECT_TIMEOUT = 5.0

#: Stage name -> the dot-directory its checkpoint lands in, taken from the
#: contracts rather than re-typed here so a renamed directory cannot drift.
#: Stages with no checkpoint of their own (``quality``, whose scope is the
#: whole run) are absent by construction.
CHECKPOINT_DIRNAMES: dict[str, str] = {
    name: contract.checkpoint_dirname
    for name, contract in STAGE_CONTRACTS.items()
    if contract.checkpoint_dirname is not None
}


def get_dashboard(
    settings: UISettings,
    *,
    run_id: str | None = None,
    stale_after: float = DEFAULT_STALE_AFTER,
    window_seconds: float = DEFAULT_THROUGHPUT_WINDOW,
    job_limit: int = 200,
) -> dict:
    """Queue state and per-stage progress, scoped to *run_id* when given.

    ``queue`` is ``None`` whenever there is no queue to read — no DSN
    configured, or the connection failed — and ``queue_error`` says which.
    That is a normal local deployment, not a fault, so the checkpoint half
    of the payload is unaffected by it.
    """
    queue, queue_error = _queue_section(
        settings, run_id, stale_after=stale_after, window_seconds=window_seconds,
        job_limit=job_limit,
    )
    return {
        "run_id": run_id,
        "stale_after_seconds": stale_after,
        "queue": queue,
        "queue_error": queue_error,
        "stages": _stage_progress(settings, run_id) if run_id else [],
    }


def _queue_section(
    settings: UISettings,
    run_id: str | None,
    *,
    stale_after: float,
    window_seconds: float,
    job_limit: int,
) -> tuple[dict | None, str | None]:
    """Read every queue view in one connection, or explain why there is none.

    A queue that cannot be reached is reported rather than raised: the
    dashboard's other half still renders, and an operator seeing "queue
    unreachable" next to live checkpoint progress has more to go on than a
    500.
    """
    if not settings.db_dsn:
        return None, None
    try:
        from womblex.cloud.queue import JobQueue

        with JobQueue(settings.db_dsn, connect_timeout=QUEUE_CONNECT_TIMEOUT) as queue:
            stats = queue.stats(run_id)
            jobs = queue.list_jobs(run_id, limit=job_limit)
            workers = queue.workers(run_id)
            stale = queue.stale_jobs(stale_after, run_id)
            throughput = queue.throughput(run_id, window_seconds=window_seconds)
    except Exception as e:
        logger.warning("dashboard: queue unavailable: %s", e)
        return None, str(e)
    return {
        "stats": stats,
        "total": sum(stats.values()),
        "jobs": [asdict(j) for j in jobs],
        "workers": [asdict(w) for w in workers],
        # Whole rows, not ids: `jobs` is capped by `job_limit` ordered on
        # `updated_at DESC`, and a stale row's `updated_at` is by definition
        # among the oldest — so it is the first thing to fall outside that
        # window, and ids alone would be unresolvable exactly when they
        # matter. Bounded by the running set, so it stays small.
        "stale": [asdict(j) for j in stale],
        "throughput": asdict(throughput),
    }, None


def _stage_progress(settings: UISettings, run_id: str) -> list[dict]:
    """Per-stage checkpoint progress for one run, in pipeline order.

    Only stages with a checkpoint on disk appear — an absent directory means
    the stage has not run for this run, which the Corpus Inspector's stage
    presence already reports from the sidecars themselves.

    ``run_id`` arrives as a *query* parameter here, not a path segment, so
    nothing has already rejected ``../``: unlike the other console reads,
    this one must contain the join itself. A run_id that is not a single
    path segment reads as "no such run", which is the same answer the
    remote branch's ``list_dirs`` check already gives — the two deployments
    must not disagree about what is reachable.
    """
    if not is_safe_run_id(run_id):
        return []
    if settings.is_remote:
        found = _remote_checkpoints(cast(str, settings.store_uri), run_id)
    else:
        run_dir = cast(Path, settings.output_root) / run_id
        found = {
            stage: read_checkpoints(run_dir / dirname)
            for stage, dirname in CHECKPOINT_DIRNAMES.items()
        }
    return [
        {"stage": stage, **asdict(progress)}
        for stage, entries in found.items()
        for progress in entries
    ]


def _remote_checkpoints(store_uri: str, run_id: str) -> dict[str, list[CheckpointProgress]]:
    """Stage each run's checkpoint JSONs into a temp dir and read them locally.

    Checkpoints are a few KB apiece, so this stays the same
    stage-then-reuse-the-local-reader pattern :mod:`womblex.ui.readers` uses
    for manifests — one definition of what a checkpoint file means.
    """
    from womblex.store.remote import RemoteStore

    store: RemoteStore = RemoteStore.from_uri(store_uri)
    if run_id not in store.list_dirs("runs"):
        return {}
    found: dict[str, list[CheckpointProgress]] = {}
    with tempfile.TemporaryDirectory(prefix="womblex-ui-") as tmp:
        for stage, dirname in CHECKPOINT_DIRNAMES.items():
            keys = store.list_files(f"runs/{run_id}/{dirname}", CHECKPOINT_GLOB)
            if not keys:
                continue
            stage_dir = Path(tmp) / dirname
            stage_dir.mkdir()
            store.download_to_dir(keys, stage_dir)
            found[stage] = read_checkpoints(stage_dir)
    return found
