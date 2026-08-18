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
from womblex.utils.isaacus_client import make_isaacus_client

logger = logging.getLogger("womblex")

#: Stages the runner can execute, spelled here so `womblex --help` does not
#: import the contract table (and pyarrow with it). `test_stage_runner.py`
#: asserts this equals `stage_contracts.STAGE_NAMES`, so it cannot drift.
#: `manifest` is absent by design — `womblex finalize` already covers it.
RUN_STAGE_CHOICES = (
    "normalise", "spellfix", "chunk", "money", "enrich",
    "embed", "link", "pii", "graph-refresh", "quality",
)


def _resolve_dsn(args: argparse.Namespace) -> str | None:
    return args.dsn or os.environ.get("WOMBLEX_DB_DSN") or os.environ.get("DATABASE_URL")


def _resolve_store(args: argparse.Namespace) -> str | None:
    return args.store or os.environ.get("WOMBLEX_STORE_URI")


def _resolve_ingest(args: argparse.Namespace) -> str | None:
    """The configured source-document location, or ``None`` to fall back to ``--store``."""
    return args.ingest or os.environ.get("WOMBLEX_INGEST_URI")


# --- enqueue -----------------------------------------------------------------


def _register_enqueue(p: argparse.ArgumentParser) -> None:
    p.add_argument("--store", help="Object-store base URI (or $WOMBLEX_STORE_URI), e.g. s3://womblex")
    p.add_argument(
        "--ingest", default=None,
        help="Object-store base URI for source documents (or $WOMBLEX_INGEST_URI). "
             "Must be disjoint from --store's runs/ output. Defaults to --store, "
             "matching --input-prefix under it, for back-compatibility.",
    )
    p.add_argument(
        "--input-prefix", default=None,
        help="Store-relative dir holding source documents. Optional with --ingest "
             "(default: the whole ingest root); required without it.",
    )
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
    from womblex.store.remote import RemoteStore, assert_disjoint_locations
    from womblex.store.retention import generate_run_id

    dsn = _resolve_dsn(args)
    store_uri = _resolve_store(args)
    ingest_uri = _resolve_ingest(args)
    if not dsn:
        logger.error("No Postgres DSN (pass --dsn or set WOMBLEX_DB_DSN / DATABASE_URL)")
        return 1
    if not store_uri:
        logger.error("No store URI (pass --store or set WOMBLEX_STORE_URI)")
        return 1
    if ingest_uri:
        try:
            assert_disjoint_locations(ingest_uri, store_uri)
        except ValueError as e:
            logger.error(str(e))
            return 1
    elif not args.input_prefix:
        logger.error(
            "No source documents location (pass --ingest, or --input-prefix under --store)"
        )
        return 1

    batch_size = args.batch_size
    if batch_size is None:
        if args.config is not None:
            from womblex.config import load_config

            batch_size = load_config(args.config).processing.batch_size
        else:
            from womblex.config import ProcessingConfig

            batch_size = ProcessingConfig().batch_size
    if batch_size < 1:
        logger.error("batch-size must be >= 1")
        return 1

    run_id = args.run_id or generate_run_id()
    output_prefix = (args.output_prefix or f"runs/{run_id}").strip("/")
    shard_prefix = f"{output_prefix}/documents"

    ingest_store = RemoteStore.from_uri(ingest_uri) if ingest_uri else RemoteStore.from_uri(store_uri)
    input_prefix = args.input_prefix or ""
    all_keys = ingest_store.list_files(input_prefix, "*")
    keys = sorted(k for k in all_keys if Path(k).suffix.lower() in SUPPORTED_EXTENSIONS)
    if not keys:
        logger.error("No supported documents under %s/%s", ingest_uri or store_uri, input_prefix)
        return 1

    specs = [
        JobSpec(
            batch_num=batch_idx,
            input_keys=keys[i : i + batch_size],
            shard_prefix=shard_prefix,
            max_attempts=args.max_attempts,
            ingest_root=ingest_uri,
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
    p.add_argument(
        "--ingest", default=None,
        help="Object-store base URI to read source documents from (or "
             "$WOMBLEX_INGEST_URI). Defaults to --store.",
    )
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
    ingest_uri = _resolve_ingest(args)
    if not dsn:
        logger.error("No Postgres DSN (pass --dsn or set WOMBLEX_DB_DSN / DATABASE_URL)")
        return 1
    if not store_uri:
        logger.error("No store URI (pass --store or set WOMBLEX_STORE_URI)")
        return 1

    config = load_config(args.config)
    completed = run_worker(
        dsn, store_uri, config,
        ingest_uri=ingest_uri,
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


# --- run-stage ---------------------------------------------------------------


def _register_run_stage(p: argparse.ArgumentParser) -> None:
    p.add_argument("--stage", required=True, choices=RUN_STAGE_CHOICES,
                   help="Downstream shard stage to run. `manifest` is absent by "
                        "design — `womblex finalize` already does it.")
    target = p.add_mutually_exclusive_group()
    target.add_argument("--store", help="Object-store base URI (or $WOMBLEX_STORE_URI)")
    target.add_argument("--shards", type=Path, default=None,
                        help="Local shard dir — run the same contract without a store.")
    p.add_argument("--run-id", default=None, help="Run to operate on (required with --store)")
    p.add_argument(
        "--output-prefix", default=None,
        help="Store-relative outputs dir (default: runs/<run_id>). "
             "Reads and writes <output-prefix>/documents/.",
    )
    p.add_argument("--config", type=Path, default=None,
                   help="Config YAML. Conditional inputs and declared outputs are "
                        "resolved from it (e.g. chunking_model, text_source).")
    p.add_argument("--dsn", default=None,
                   help="Optional Postgres DSN — warn if extraction jobs are still draining")
    p.add_argument("--force", action="store_true",
                   help="Re-run bases whose declared outputs already exist")
    p.add_argument("--stage-checkpoints", action="store_true",
                   help="Stage the stage's checkpoint dir in/out of the store. "
                        "Single-invocation per run — concurrent runners would clobber it.")
    p.add_argument("--dataset", default="runner", help="Checkpoint dataset name. Default: 'runner'.")


def _runner_config(config_path: Path | None):  # type: ignore[no-untyped-def]
    """Load the config, or a defaults instance when ``--config`` is omitted.

    The runner never reads ``dataset`` / ``paths`` — it operates on a shard
    prefix — but ``WomblexConfig`` requires them, so absent a config file we
    supply placeholders and let every stage section fall back to its defaults.
    """
    from womblex.config import WomblexConfig, load_config

    if config_path is not None:
        return load_config(config_path)
    # Validated from a mapping rather than passed as keywords: the fields are
    # typed as their models, so keyword dicts are a type error even though
    # pydantic coerces them at runtime.
    return WomblexConfig.model_validate({
        "dataset": {"name": "runner"},
        "paths": {"input_root": ".", "output_root": ".", "checkpoint_dir": "."},
    })


def cmd_run_stage(args: argparse.Namespace) -> int:
    """Run one downstream shard stage against object storage, a batch at a time.

    The generalisation of ``finalize``: that command downloads one sidecar class
    (``*._manifest.parquet``), calls an unchanged library function against a temp
    ``Path``, and uploads the result. This does the same for the per-batch stages,
    driven by the declarations in :mod:`womblex.cloud.stage_contracts`.

    Idempotent — completed bases skip on their published outputs, so re-running
    as more batches land processes only the new ones. Stage *ordering* is the
    caller's: ``embed`` needs ``chunk`` to have run, ``pii`` is terminal after
    ``enrich`` and ``embed``.
    """
    from womblex.cloud.stage_contracts import STAGE_CONTRACTS, RunContext
    from womblex.cloud.stage_runner import (
        checkpoint_prefix_for,
        run_stage_local,
        run_stage_remote,
    )

    contract = STAGE_CONTRACTS[args.stage]
    config = _runner_config(args.config)

    if contract.preflight is not None:
        try:
            contract.preflight(config)
        except Exception as e:
            logger.error("%s preflight failed: %s", args.stage, e)
            return 1

    ctx = RunContext()
    if contract.needs_isaacus_api:
        from womblex.utils.availability import isaacus_available

        if not isaacus_available():
            # `chunk_shards` would otherwise warn, write nothing and return
            # cleanly — a remote no-op that looks like success.
            logger.error(
                "%s needs Isaacus (isaacus SDK + ISAACUS_API_KEY, or "
                "ISAACUS_SAGEMAKER_ENDPOINTS for a private deployment); none is "
                "resolvable. Refusing to run rather than publishing nothing.",
                args.stage,
            )
            return 1
    if contract.needs_client:
        try:
            ctx.client = make_isaacus_client(models=contract.models(config))
        except ImportError as e:
            logger.error("Isaacus SDK not usable (reinstall womblex): %s", e)
            return 1
        except Exception as e:
            # Logs the exception, not the key — the rule trips on "API_KEY" in the literal.
            # nosemgrep: python.lang.security.audit.logging.logger-credential-leak.python-logger-credential-disclosure
            logger.error(
                "Could not construct Isaacus client (check ISAACUS_API_KEY, or "
                "ISAACUS_SAGEMAKER_ENDPOINTS for a private deployment): %s", e,
            )
            return 1

    if args.shards is not None:
        if not args.shards.is_dir():
            logger.error("--shards path is not a directory: %s", args.shards)
            return 1
        logger.info("run-stage %s --shards %s", args.stage, args.shards)
        summary = run_stage_local(contract, args.shards, config, ctx=ctx)
        summary.log()
        return summary.exit_code

    store_uri = _resolve_store(args)
    if not store_uri:
        logger.error("No target (pass --shards, or --store / $WOMBLEX_STORE_URI)")
        return 1
    if not args.run_id:
        logger.error("--run-id is required with --store")
        return 1

    from womblex.store.remote import RemoteStore

    output_prefix = (args.output_prefix or f"runs/{args.run_id}").strip("/")
    shard_prefix = f"{output_prefix}/documents"
    store = RemoteStore.from_uri(store_uri)

    _warn_if_draining(args, args.run_id)

    ckpt_prefix = checkpoint_prefix_for(contract, output_prefix) if args.stage_checkpoints else None
    logger.info("run-stage %s: %s/%s (scope=%s, mutation=%s)",
                args.stage, store_uri, shard_prefix,
                contract.scope.value, contract.mutation.value)
    summary = run_stage_remote(
        contract, store, shard_prefix, config,
        ctx=ctx, force=args.force,
        checkpoint_prefix=ckpt_prefix, checkpoint_dataset=args.dataset,
    )
    summary.log()
    return summary.exit_code


def _warn_if_draining(args: argparse.Namespace, run_id: str) -> None:
    """Advisory only — running against a still-draining fleet yields partial outputs."""
    dsn = _resolve_dsn(args)
    if not dsn:
        return
    from womblex.cloud.queue import JobQueue

    with JobQueue(dsn) as queue:
        stats = queue.stats(run_id)
    unfinished = stats.get("pending", 0) + stats.get("running", 0)
    if unfinished:
        logger.warning(
            "run %s has %d unfinished extraction job(s) (%s) — this stage will "
            "cover only the batches present so far. Re-run when the fleet drains.",
            run_id, unfinished, stats,
        )
    if stats.get("failed"):
        logger.warning("run %s has %d failed job(s); their docs are absent",
                       run_id, stats["failed"])


COMMANDS = [
    Command("enqueue", "Plan batches into the cloud job queue", _register_enqueue, cmd_enqueue),
    Command("worker", "Process batches from the cloud job queue", _register_worker, cmd_worker),
    Command("jobs", "Show cloud job-queue status", _register_jobs, cmd_jobs),
    Command("finalize", "Consolidate a distributed run's manifest in object storage",
            _register_finalize, cmd_finalize),
    Command("run-stage", "Run a downstream shard stage against object storage",
            _register_run_stage, cmd_run_stage),
]
