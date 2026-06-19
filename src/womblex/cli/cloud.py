"""Cloud CLI subcommands: ``enqueue`` (plan batches), ``worker`` (process),
``jobs`` (status).

Distributed counterpart to ``womblex run``. ``enqueue`` lists source documents
in an object store, splits them into batches, and writes one queue row each;
``worker`` claims and processes those rows; ``jobs`` reports progress. The queue
replaces the local JSON checkpoint, so resuming is just re-running ``enqueue``
(idempotent) and starting workers.

Inputs and outputs live under one object-store base URI (``--store``), shared by
enqueue and every worker. Outputs land at ``<store>/<output-prefix>/documents/``
as the usual ``batch-NNNN.*.parquet`` shards — point ``womblex manifest`` /
``chunk --shards`` at that dir (synced down) exactly as for a local run.
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path

from womblex.cli._shared import SUPPORTED_EXTENSIONS, Command

logger = logging.getLogger("womblex")


def _resolve_dsn(args: argparse.Namespace) -> str | None:
    return args.dsn or os.environ.get("WOMBLEX_DB_DSN") or os.environ.get("DATABASE_URL")


def _resolve_store(args: argparse.Namespace) -> str | None:
    return args.store or os.environ.get("WOMBLEX_STORE_URI")


# --- enqueue -----------------------------------------------------------------


def _register_enqueue(p: argparse.ArgumentParser) -> None:
    p.add_argument("--store", help="Object-store base URI (or $WOMBLEX_STORE_URI), e.g. s3://womblex")
    p.add_argument("--input-prefix", required=True, help="Store-relative dir holding source documents")
    p.add_argument("--config", type=Path, help="Config YAML (sources processing.batch_size)")
    p.add_argument("--run-id", default=None, help="Run identifier (default: auto timestamp)")
    p.add_argument(
        "--output-prefix", default=None,
        help="Store-relative dir for outputs (default: runs/<run_id>). "
             "Shards land under <output-prefix>/documents/.",
    )
    p.add_argument("--batch-size", type=int, default=None, help="Docs per batch (overrides config)")
    p.add_argument("--max-attempts", type=int, default=3, help="Retries per batch before 'failed'")
    p.add_argument("--dsn", default=None, help="Postgres DSN (or $WOMBLEX_DB_DSN / $DATABASE_URL)")
    p.add_argument("--create-schema", action="store_true", help="Create the jobs table first")


def cmd_enqueue(args: argparse.Namespace) -> int:
    from womblex.cloud.queue import JobQueue, JobSpec
    from womblex.store.remote import RemoteStore
    from womblex.store.retention import generate_run_id

    dsn = _resolve_dsn(args)
    store_uri = _resolve_store(args)
    if not dsn:
        logger.error("No Postgres DSN (pass --dsn or set WOMBLEX_DB_DSN / DATABASE_URL)")
        return 1
    if not store_uri:
        logger.error("No store URI (pass --store or set WOMBLEX_STORE_URI)")
        return 1

    batch_size = args.batch_size
    if batch_size is None:
        if args.config is not None:
            from womblex.config import load_config

            batch_size = load_config(args.config).processing.batch_size
        else:
            batch_size = 100
    if batch_size < 1:
        logger.error("batch-size must be >= 1")
        return 1

    run_id = args.run_id or generate_run_id()
    output_prefix = (args.output_prefix or f"runs/{run_id}").strip("/")
    shard_prefix = f"{output_prefix}/documents"

    store = RemoteStore.from_uri(store_uri)
    all_keys = store.list_files(args.input_prefix, "*")
    keys = sorted(k for k in all_keys if Path(k).suffix.lower() in SUPPORTED_EXTENSIONS)
    if not keys:
        logger.error("No supported documents under %s/%s", store_uri, args.input_prefix)
        return 1

    specs = [
        JobSpec(
            batch_num=batch_idx,
            input_keys=keys[i : i + batch_size],
            shard_prefix=shard_prefix,
            max_attempts=args.max_attempts,
        )
        for batch_idx, i in enumerate(range(0, len(keys), batch_size), start=1)
    ]

    with JobQueue(dsn) as queue:
        if args.create_schema:
            queue.ensure_schema()
        inserted = queue.enqueue(run_id, specs)

    logger.info(
        "run_id=%s: %d document(s) -> %d batch(es), %d newly enqueued. Shards -> %s/%s",
        run_id, len(keys), len(specs), inserted, store_uri, shard_prefix,
    )
    logger.info("Start workers with: womblex worker --store %s --run-id %s --config <cfg>",
                store_uri, run_id)
    return 0


# --- worker ------------------------------------------------------------------


def _register_worker(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=Path, required=True, help="Config YAML (pipeline settings)")
    p.add_argument("--store", help="Object-store base URI (or $WOMBLEX_STORE_URI)")
    p.add_argument("--dsn", default=None, help="Postgres DSN (or $WOMBLEX_DB_DSN / $DATABASE_URL)")
    p.add_argument("--run-id", default=None, help="Only claim jobs for this run (default: any)")
    p.add_argument("--worker-id", default=None, help="Worker identity in locks (default: host:pid)")
    p.add_argument("--poll-interval", type=float, default=5.0, help="Seconds between empty polls")
    p.add_argument("--once", action="store_true", help="Process at most one job then exit")
    p.add_argument("--idle-timeout", type=float, default=None, help="Exit after N idle seconds")
    p.add_argument("--stale-timeout", type=float, default=None,
                   help="Requeue 'running' jobs locked longer than N seconds (crash recovery)")


def cmd_worker(args: argparse.Namespace) -> int:
    from womblex.cloud.worker import run_worker
    from womblex.config import load_config

    dsn = _resolve_dsn(args)
    store_uri = _resolve_store(args)
    if not dsn:
        logger.error("No Postgres DSN (pass --dsn or set WOMBLEX_DB_DSN / DATABASE_URL)")
        return 1
    if not store_uri:
        logger.error("No store URI (pass --store or set WOMBLEX_STORE_URI)")
        return 1

    config = load_config(args.config)
    completed = run_worker(
        dsn, store_uri, config,
        worker_id=args.worker_id,
        run_id=args.run_id,
        poll_interval=args.poll_interval,
        once=args.once,
        idle_timeout=args.idle_timeout,
        stale_timeout=args.stale_timeout,
    )
    logger.info("worker exiting: %d job(s) completed", completed)
    return 0


# --- jobs --------------------------------------------------------------------


def _register_jobs(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dsn", default=None, help="Postgres DSN (or $WOMBLEX_DB_DSN / $DATABASE_URL)")
    p.add_argument("--run-id", default=None, help="Limit to one run (default: all runs)")
    p.add_argument("--create-schema", action="store_true", help="Create the jobs table if missing")


def cmd_jobs(args: argparse.Namespace) -> int:
    from womblex.cloud.queue import JobQueue

    dsn = _resolve_dsn(args)
    if not dsn:
        logger.error("No Postgres DSN (pass --dsn or set WOMBLEX_DB_DSN / DATABASE_URL)")
        return 1

    with JobQueue(dsn) as queue:
        if args.create_schema:
            queue.ensure_schema()
        stats = queue.stats(args.run_id)

    scope = f"run {args.run_id}" if args.run_id else "all runs"
    if not stats:
        logger.info("No jobs (%s)", scope)
        return 0
    total = sum(stats.values())
    parts = ", ".join(f"{status}={count}" for status, count in sorted(stats.items()))
    logger.info("Jobs (%s): %s (total %d)", scope, parts, total)
    return 0


# --- finalize ----------------------------------------------------------------


def _register_finalize(p: argparse.ArgumentParser) -> None:
    p.add_argument("--store", help="Object-store base URI (or $WOMBLEX_STORE_URI)")
    p.add_argument("--run-id", required=True, help="Run to finalise")
    p.add_argument(
        "--output-prefix", default=None,
        help="Store-relative outputs dir (default: runs/<run_id>). "
             "Reads <output-prefix>/documents/*._manifest.parquet.",
    )
    p.add_argument("--dsn", default=None,
                   help="Optional Postgres DSN — warn if jobs are still unfinished")


def cmd_finalize(args: argparse.Namespace) -> int:
    """Consolidate a distributed run's shard manifests into one run manifest.

    ``womblex run`` writes ``<run_root>/manifest.parquet`` at the end of a local
    run; a distributed run has no single end, so this is the explicit
    finalisation step. Run it once the fleet has drained: it downloads every
    ``*._manifest.parquet`` shard, consolidates them, and uploads
    ``<output-prefix>/manifest.parquet`` back to the store. Idempotent — safe to
    re-run as more batches land.
    """
    from womblex.store.remote import RemoteStore
    from womblex.store.run_manifest import RUN_MANIFEST_FILENAME, write_run_manifest

    store_uri = _resolve_store(args)
    if not store_uri:
        logger.error("No store URI (pass --store or set WOMBLEX_STORE_URI)")
        return 1

    output_prefix = (args.output_prefix or f"runs/{args.run_id}").strip("/")
    shard_prefix = f"{output_prefix}/documents"
    store = RemoteStore.from_uri(store_uri)

    dsn = _resolve_dsn(args)
    if dsn:
        from womblex.cloud.queue import JobQueue

        with JobQueue(dsn) as queue:
            stats = queue.stats(args.run_id)
        unfinished = stats.get("pending", 0) + stats.get("running", 0)
        if unfinished:
            logger.warning(
                "run %s has %d unfinished job(s) (%s) — finalising shards present so far",
                args.run_id, unfinished, stats,
            )
        if stats.get("failed"):
            logger.warning("run %s has %d failed job(s); their docs are absent",
                           args.run_id, stats["failed"])

    manifest_keys = store.list_files(shard_prefix, "*._manifest.parquet")
    if not manifest_keys:
        logger.error("No *._manifest.parquet under %s/%s", store_uri, shard_prefix)
        return 1

    with tempfile.TemporaryDirectory(prefix="womblex-finalize-") as tmp:
        docs = Path(tmp) / "documents"
        store.download_to_dir(manifest_keys, docs)
        local_manifest = write_run_manifest(docs)
        store.upload_file(local_manifest, f"{output_prefix}/{RUN_MANIFEST_FILENAME}")

    logger.info(
        "Finalised run %s: %d shard manifest(s) -> %s/%s/%s",
        args.run_id, len(manifest_keys), store_uri, output_prefix, RUN_MANIFEST_FILENAME,
    )
    return 0


COMMANDS = [
    Command("enqueue", "Plan batches into the cloud job queue", _register_enqueue, cmd_enqueue),
    Command("worker", "Process batches from the cloud job queue", _register_worker, cmd_worker),
    Command("jobs", "Show cloud job-queue status", _register_jobs, cmd_jobs),
    Command("finalize", "Consolidate a distributed run's manifest in object storage",
            _register_finalize, cmd_finalize),
]
