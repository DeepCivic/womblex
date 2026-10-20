"""``retry-stage``: re-run a failed stage of a distributed run from its first failed unit."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from womblex.cli._shared import Command
from womblex.cli.cloud import _DSN_HELP, _resolve_dsn, _resolve_store

logger = logging.getLogger("womblex")


def _register_retry_stage(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run-id", required=True, help="Run the stage belongs to")
    p.add_argument("--stage", required=True, help="Stage to retry")
    p.add_argument("--config", type=Path, required=True,
                   help="Config YAML — the paths and models the retry's pre-launch checks use "
                        "(a stage dispatched by a workflow keeps that workflow's overlay)")
    p.add_argument("--store", default=None,
                   help="Object-store base URI (or $WOMBLEX_STORE_URI). When set, the stage's "
                        "required inputs are checked against what the run holds.")
    p.add_argument("--dsn", default=None, help=_DSN_HELP)


def cmd_retry_stage(args: argparse.Namespace) -> int:
    """Fork a failed stage from its first failed unit; exit 1 if the retry was refused.

    Every refusal is before the fork. The retry runs on the fleet, as the stage did; the
    original stays listed `failed`, naming it, and the run's rollup counts the retry.
    """
    from womblex.cloud.dbos_app import ensure_schema
    from womblex.cloud.jobs import RunBoard, RunOwnedError
    from womblex.cloud.retry import RetryRefusedError, retry_stage
    from womblex.config import load_config
    from womblex.store.remote import RemoteStore

    dsn = _resolve_dsn(args)
    store_uri = _resolve_store(args)
    ensure_schema(dsn)
    try:
        with RunBoard(dsn) as board:
            retry = retry_stage(
                board, args.run_id, args.stage, config=load_config(args.config),
                store=RemoteStore.from_uri(store_uri) if store_uri else None, dsn=dsn,
            )
    except (RetryRefusedError, RunOwnedError) as e:
        logger.error("retry refused: %s", e)
        return 1
    if retry.start_step is None:
        logger.info("%s retried as %s over its affected units %s (after %s)", args.stage, retry.retry_id,
                    ", ".join(retry.bases) if retry.bases is not None else "(all)", retry.forked_from)
    else:
        logger.info("%s retried as %s, from step %d of %s", args.stage, retry.retry_id,
                    retry.start_step, retry.forked_from)
    # The coordinator wrote the run manifest before this retry; it does not run again.
    logger.info("Once %s succeeds, rewrite the run manifest:\n  womblex finalize --run-id %s%s",
                retry.retry_id, args.run_id, f" --store {store_uri}" if store_uri else "")
    if retry.workflow:  # nothing redelivers automatically (Phase 3: delivery follows the retry rules)
        where = f" --store {store_uri}" if store_uri else ""
        logger.info("Once the run's stages have succeeded, redeliver workflow %s, replacing what was delivered:\n"
                    "  womblex deliver <%s's workflow file> --run-id %s%s",
                    retry.workflow, retry.workflow, args.run_id, where)
    if retry.later:
        where = f" --config {args.config}" + (f" --store {store_uri}" if store_uri else "")
        logger.info(
            "%s ran on the partial output of %s. Once %s succeeds, re-run each in this order:\n%s",
            ", ".join(retry.later), args.stage, retry.retry_id,
            "\n".join(f"  womblex retry-stage --run-id {args.run_id} --stage {s}{where}" for s in retry.later),
        )
    return 0


COMMANDS = [
    Command("retry-stage", "Retry a failed stage of a distributed run from its first failed unit",
            _register_retry_stage, cmd_retry_stage),
]
