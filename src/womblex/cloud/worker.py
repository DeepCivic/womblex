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
from womblex.store.remote import RemoteStore, same_location
from womblex.utils.run_log import capture_batch_log

logger = logging.getLogger(__name__)


def default_worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def _same_ingest(a: str, b: str) -> bool:
    """Whether two ingest roots name the same location.

    Normalised, not compared as strings: an enqueue and a worker configured
    from different places (a flag here, a compose env var there) routinely
    differ by a trailing slash while naming the same bucket and prefix. An
    unparseable root on either side falls back to an exact match rather than
    raising — the refusal path must not itself throw.
    """
    if a == b:
        return True
    try:
        return same_location(a, b)
    except (ValueError, ImportError):
        return False


def _log_key(job: Job) -> str:
    """Where this batch's log lands, a ``logs/`` sibling of its shards.

    ``shard_prefix`` is ``runs/<run_id>/documents``; the log goes to
    ``runs/<run_id>/logs/batch-NNNN.log``, so the reader can list a run's
    logs without knowing the batch numbering ahead of time.
    """
    run_prefix = job.shard_prefix.rsplit("/", 1)[0]
    return f"{run_prefix}/logs/batch-{job.batch_num:04d}.log"


def _process_job(job: Job, config: WomblexConfig, store: RemoteStore, ingest: RemoteStore) -> None:
    """Stage inputs, run the batch, publish shards **and the batch log**.

    The log is captured for the whole run and published in a ``finally``, so a
    failed batch still leaves its ``batch-NNNN.log`` in the store — that is the
    case the operator most needs it. A failed *upload* of the log never masks
    the original error: it is logged and swallowed.
    """
    with tempfile.TemporaryDirectory(prefix="womblex-job-") as tmp:
        root = Path(tmp)
        inputs_dir = root / "inputs"
        shards_dir = root / "shards"
        shards_dir.mkdir(parents=True, exist_ok=True)
        log_path = root / f"batch-{job.batch_num:04d}.log"

        try:
            with capture_batch_log(log_path):
                files = ingest.download_to_dir(job.input_keys, inputs_dir)
                outcome = process_batch(
                    files, config, batch_num=job.batch_num, shard_dir=shards_dir,
                )
                # Glob off the shard path the batch reported, so the naming
                # scheme lives only in womblex.batch.
                store.upload_glob(shards_dir, f"{outcome.shard_path.stem}.*", job.shard_prefix)
                logger.info(
                    "[batch %d] %d ok, %d failed -> %s",
                    job.batch_num, outcome.batch.succeeded, outcome.batch.failed,
                    job.shard_prefix,
                )
        finally:
            # Outside the capture (the file is now closed and complete) but
            # inside the temp dir; publish on success and failure alike.
            try:
                store.upload_file(log_path, _log_key(job))
            except Exception:  # publishing the log must never mask the batch's own error
                logger.exception("[batch %d] failed to publish batch log", job.batch_num)


def run_worker(
    dsn: str,
    store_uri: str,
    config: WomblexConfig,
    *,
    ingest_uri: str | None = None,
    worker_id: str | None = None,
    run_id: str | None = None,
    poll_interval: float = 5.0,
    once: bool = False,
    idle_timeout: float | None = None,
    stale_timeout: float | None = None,
) -> int:
    """Run the claim→process→publish loop until drained or interrupted.

    Returns the number of jobs completed by this worker.

    ``ingest_uri`` names a second store to download source documents from;
    ``None`` keeps today's single-store behaviour (inputs and outputs share
    ``store_uri``). ``once`` processes at most one job then exits (handy for
    one-shot container/CronJob execution). ``idle_timeout`` exits after that
    many seconds with no work (auto-scale-to-zero). ``stale_timeout`` requeues
    ``running`` rows orphaned by crashed workers before each claim.
    """
    worker_id = worker_id or default_worker_id()
    queue = JobQueue(dsn)
    store = RemoteStore.from_uri(store_uri)
    ingest = RemoteStore.from_uri(ingest_uri) if ingest_uri else store
    worker_ingest_root = ingest_uri or store_uri
    logger.info(
        "worker %s started (store=%s, ingest=%s, run=%s)",
        worker_id, store_uri, worker_ingest_root, run_id or "ALL",
    )

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

            logger.info("worker %s claimed job %d (batch %d, attempt %d)",
                        worker_id, job.id, job.batch_num, job.attempts)
            if job.ingest_root and not _same_ingest(job.ingest_root, worker_ingest_root):
                error = (
                    f"ingest root mismatch: job enqueued against "
                    f"{job.ingest_root!r}, this worker reads from "
                    f"{worker_ingest_root!r}"
                )
                logger.error("job %d (batch %d) refused: %s", job.id, job.batch_num, error)
                # Released, not failed: the batch is fine, this worker is the
                # wrong one for it. Failing here would burn the retry budget —
                # and, since the row returns to pending, re-claim it in a tight
                # loop until the job died. A refusal is "no work for me", so
                # it backs off and ages towards idle_timeout like an empty
                # claim rather than holding a mis-wired worker up forever.
                queue.release(job.id, error)
                if once:
                    break
                idle_since = idle_since or time.monotonic()
                time.sleep(poll_interval)
                continue

            idle_since = None
            try:
                _process_job(job, config, store, ingest)
                queue.complete(job.id)
                completed += 1
            except Exception as e:  # one bad batch must not kill the worker
                logger.exception("job %d (batch %d) failed", job.id, job.batch_num)
                queue.fail(
                    job.id,
                    f"{type(e).__name__}: {e} — worker ingest root {worker_ingest_root}",
                )

            if once:
                break
    except KeyboardInterrupt:  # pragma: no cover - interactive
        logger.info("worker %s interrupted; %d job(s) done", worker_id, completed)
    finally:
        queue.close()
    return completed


__all__ = ["default_worker_id", "run_worker"]
