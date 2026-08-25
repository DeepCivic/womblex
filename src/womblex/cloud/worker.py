"""Worker loop: claim a job, run it, publish what it produced.

Each iteration claims one row from the queue. An **extraction batch** pulls its
input documents from object storage into a throwaway scratch dir, runs the
shared ``womblex.batch.process_batch`` body (identical to ``womblex run``), then
pushes the resulting ``batch-NNNN.*.parquet`` shards back. A **stage** job needs
no staging of its own — its inputs are already in the store — so it hands the
run's shard prefix to ``stage_runner.run_stage_remote``, which stages one base
at a time and publishes each stage's sidecars.

The same worker serves both: execution stays on the fleet, and a dispatcher
(the CLI, or the console) only ever writes rows. A per-job failure marks the row
(with retry) and moves on — one bad document never stops the fleet, the same
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


def _output_prefix(job: Job) -> str:
    """The run's output dir — ``shard_prefix`` (``…/documents``) less its leaf."""
    return job.shard_prefix.rsplit("/", 1)[0]


def _log_name(job: Job) -> str:
    return f"stage-{job.stage}" if job.kind == "stage" else f"batch-{job.batch_num:04d}"


def _log_key(job: Job) -> str:
    """Where this job's log lands, a ``logs/`` sibling of the run's shards.

    ``shard_prefix`` is ``runs/<run_id>/documents``; the log goes to
    ``runs/<run_id>/logs/batch-NNNN.log`` (or ``logs/stage-<name>.log``), so the
    reader can list a run's logs without knowing the batch numbering ahead of
    time.
    """
    return f"{_output_prefix(job)}/logs/{_log_name(job)}.log"


class StageJobFailed(Exception):
    """A stage job ran but its summary reported a non-zero exit."""


class StageNotReady(Exception):
    """Every base of a stage job is blocked on an upstream input that is absent.

    Distinct from :class:`StageJobFailed` because nothing went wrong: the stage
    was claimed before the sidecar it reads existed. Treated like the
    ingest-root refusal — released, not failed — so the attempt is not consumed.
    """


def _process_job(job: Job, config: WomblexConfig, store: RemoteStore, ingest: RemoteStore) -> None:
    """Run one claimed job, capturing and publishing its log either way.

    The log is captured for the whole run and published in a ``finally``, so a
    failed job still leaves its ``batch-NNNN.log`` / ``stage-<name>.log`` in the
    store — that is the case the operator most needs it. A failed *upload* of
    the log never masks the original error: it is logged and swallowed.
    """
    with tempfile.TemporaryDirectory(prefix="womblex-job-") as tmp:
        root = Path(tmp)
        log_path = root / f"{_log_name(job)}.log"
        try:
            with capture_batch_log(log_path):
                if job.kind == "stage":
                    _run_stage(job, config, store)
                else:
                    _run_batch(job, config, store, ingest, root)
        finally:
            # Outside the capture (the file is now closed and complete) but
            # inside the temp dir; publish on success and failure alike.
            try:
                store.upload_file(log_path, _log_key(job))
            except Exception:  # publishing the log must never mask the job's own error
                logger.exception("[%s] failed to publish job log", job.label)


def _run_batch(
    job: Job, config: WomblexConfig, store: RemoteStore, ingest: RemoteStore, root: Path,
) -> None:
    """Stage this batch's inputs, extract them, publish the shards."""
    inputs_dir = root / "inputs"
    shards_dir = root / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    files = ingest.download_to_dir(job.input_keys, inputs_dir)
    outcome = process_batch(files, config, batch_num=job.batch_num, shard_dir=shards_dir)
    # Glob off the shard path the batch reported, so the naming scheme lives
    # only in womblex.batch.
    store.upload_glob(shards_dir, f"{outcome.shard_path.stem}.*", job.shard_prefix)
    logger.info(
        "[batch %d] %d ok, %d failed -> %s",
        job.batch_num, outcome.batch.succeeded, outcome.batch.failed, job.shard_prefix,
    )


def _run_stage(job: Job, config: WomblexConfig, store: RemoteStore) -> None:
    """Run one downstream stage over the run's shard prefix.

    The same call ``womblex run-stage --store`` makes, with the same
    preconditions — the queue is a second dispatcher for it, not a second
    implementation. Checkpoints are staged: the claim gate lets only one stage
    of a run run at a time, so there is no concurrent runner to clobber them,
    and a crashed stage resumes where it stopped instead of re-doing the run.

    A non-zero summary is raised rather than returned, so the row records the
    failure and retries — a stage that published nothing must not read as done.
    Two shapes of non-zero, though: a stage whose every base is blocked on an
    absent upstream sidecar raises :class:`StageNotReady` instead, because the
    caller must release it rather than spend an attempt on work that has not
    become runnable yet.
    """
    from womblex.cloud.stage_contracts import STAGE_CONTRACTS
    from womblex.cloud.stage_runner import (
        checkpoint_prefix_for,
        prepare_stage_context,
        run_stage_remote,
    )

    contract = STAGE_CONTRACTS[job.stage or ""]
    ctx = prepare_stage_context(contract, config)
    summary = run_stage_remote(
        contract, store, job.shard_prefix, config, ctx=ctx,
        checkpoint_prefix=checkpoint_prefix_for(contract, _output_prefix(job)),
    )
    summary.log()
    if summary.exit_code == 0:
        return
    detail = (
        f"stage {contract.name}: {summary.failed} failed, "
        f"{summary.not_ready} not-ready of {summary.bases} base(s)"
    )
    blocked_only = (
        not summary.failed
        and not summary.discovery_failed
        and summary.bases
        and summary.not_ready == summary.bases
    )
    if blocked_only:
        missing = ", ".join(sorted(summary.not_ready_missing)) or "an upstream sidecar"
        raise StageNotReady(f"{detail} — every base awaits {missing}")
    raise StageJobFailed(detail)


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

    Serves both row kinds: extraction batches and, once a run's batches have
    settled, the downstream stage jobs a dispatcher enqueued for it. Returns the
    number of jobs completed by this worker.

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

            logger.info("worker %s claimed job %d (%s, attempt %d)",
                        worker_id, job.id, job.label, job.attempts)
            # Stage jobs read the store, never the ingest, so the guard is
            # batch-only — `ingest_root` is NULL on a stage row regardless.
            if job.ingest_root and not _same_ingest(job.ingest_root, worker_ingest_root):
                error = (
                    f"ingest root mismatch: job enqueued against "
                    f"{job.ingest_root!r}, this worker reads from "
                    f"{worker_ingest_root!r}"
                )
                logger.error("job %d (%s) refused: %s", job.id, job.label, error)
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
            except StageNotReady as e:
                # Released, not failed, for the same reason as the ingest
                # mismatch above: the stage is fine, it is just early. Failing
                # here spends an attempt per poll on a stage whose upstream has
                # not published yet, so a slow-draining run exhausts the retry
                # budget and lands the stage terminally failed — the operator
                # then sees "failed" for a pipeline that was merely in order.
                logger.warning("job %d (%s) released: %s", job.id, job.label, e)
                queue.release(job.id, str(e))
                if once:
                    break
                idle_since = time.monotonic()
                time.sleep(poll_interval)
                continue
            except Exception as e:  # one bad job must not kill the worker
                logger.exception("job %d (%s) failed", job.id, job.label)
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


__all__ = ["StageJobFailed", "StageNotReady", "default_worker_id", "run_worker"]
