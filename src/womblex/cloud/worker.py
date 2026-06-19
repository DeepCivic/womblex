"""Worker loop: claim a batch, stage it, run the pipeline, publish shards.

Each iteration claims one batch from the queue, pulls its input documents from
object storage into a throwaway scratch dir, runs the shared
``womblex.batch.process_batch`` body (identical to ``womblex run``), then pushes
the resulting ``batch-NNNN.*.parquet`` shards back. A per-job failure marks the
row (with retry) and moves on — one bad document never stops the fleet, the same
contract the local runner honours.
"""

from __future__ import annotations

import logging
import os
import socket
import tempfile
import time
from pathlib import Path

from womblex.batch import process_batch
from womblex.cloud.queue import Job, JobQueue
from womblex.config import WomblexConfig
from womblex.store.remote import RemoteStore

logger = logging.getLogger(__name__)


def default_worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def _process_job(job: Job, config: WomblexConfig, store: RemoteStore) -> None:
    """Stage inputs, run the batch, publish shards. Raises on failure."""
    with tempfile.TemporaryDirectory(prefix="womblex-job-") as tmp:
        root = Path(tmp)
        inputs_dir = root / "inputs"
        shards_dir = root / "shards"
        shards_dir.mkdir(parents=True, exist_ok=True)

        files = store.download_to_dir(job.input_keys, inputs_dir)
        outcome = process_batch(files, config, batch_num=job.batch_num, shard_dir=shards_dir)
        store.upload_glob(shards_dir, f"batch-{job.batch_num:04d}.*", job.shard_prefix)
        logger.info(
            "[batch %d] %d ok, %d failed -> %s",
            job.batch_num, outcome.batch.succeeded, outcome.batch.failed, job.shard_prefix,
        )


def run_worker(
    dsn: str,
    store_uri: str,
    config: WomblexConfig,
    *,
    worker_id: str | None = None,
    run_id: str | None = None,
    poll_interval: float = 5.0,
    once: bool = False,
    idle_timeout: float | None = None,
    stale_timeout: float | None = None,
) -> int:
    """Run the claim→process→publish loop until drained or interrupted.

    Returns the number of jobs completed by this worker.

    ``once`` processes at most one job then exits (handy for one-shot
    container/CronJob execution). ``idle_timeout`` exits after that many
    seconds with no work (auto-scale-to-zero). ``stale_timeout`` requeues
    ``running`` rows orphaned by crashed workers before each claim.
    """
    worker_id = worker_id or default_worker_id()
    queue = JobQueue(dsn)
    store = RemoteStore.from_uri(store_uri)
    logger.info("worker %s started (store=%s, run=%s)", worker_id, store_uri, run_id or "ALL")

    completed = 0
    idle_since: float | None = None
    try:
        while True:
            if stale_timeout:
                queue.requeue_stale(stale_timeout)

            job = queue.claim(worker_id, run_id)
            if job is None:
                if once:
                    break
                now = time.monotonic()
                idle_since = idle_since or now
                if idle_timeout is not None and now - idle_since >= idle_timeout:
                    logger.info("worker %s idle for %.0fs — exiting", worker_id, idle_timeout)
                    break
                time.sleep(poll_interval)
                continue

            idle_since = None
            logger.info("worker %s claimed job %d (batch %d, attempt %d)",
                        worker_id, job.id, job.batch_num, job.attempts)
            try:
                _process_job(job, config, store)
                queue.complete(job.id)
                completed += 1
            except Exception as e:  # one bad batch must not kill the worker
                logger.exception("job %d (batch %d) failed", job.id, job.batch_num)
                queue.fail(job.id, repr(e))

            if once:
                break
    except KeyboardInterrupt:  # pragma: no cover - interactive
        logger.info("worker %s interrupted; %d job(s) done", worker_id, completed)
    finally:
        queue.close()
    return completed


__all__ = ["default_worker_id", "run_worker"]
