"""``jobs`` CLI subcommand: cloud job-queue status over the DBOS system database."""

from __future__ import annotations

import argparse
import logging

from womblex.cli._shared import Command
from womblex.cli.cloud import _DSN_HELP, _resolve_dsn

logger = logging.getLogger("womblex")


def _register_jobs(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dsn", default=None, help=_DSN_HELP)
    p.add_argument("--run-id", default=None, help="Limit to one run (default: all runs)")
    p.add_argument("--create-schema", action="store_true",
                   help="Create the DBOS system tables first (enqueue always does; `jobs` "
                        "only reads, so use this to provision an empty database)")


def cmd_jobs(args: argparse.Namespace) -> int:
    from womblex.cloud.dbos_app import ensure_schema
    from womblex.cloud.jobs import RunBoard

    dsn = _resolve_dsn(args)
    if args.create_schema:
        ensure_schema(dsn)
    with RunBoard(dsn) as board:
        stats = board.stats(args.run_id)
        summaries = board.runs(run_id=args.run_id, limit=10_000)
        orphaned = [r.run_id for r in summaries if r.orphaned]

    scope = f"run {args.run_id}" if args.run_id else "all runs"
    if not stats:
        logger.info("No jobs (%s)", scope)
        return 0
    total = sum(stats.values())
    parts = ", ".join(f"{status}={count}" for status, count in sorted(stats.items()))
    logger.info("Jobs (%s): %s (total %d)", scope, parts, total)
    if args.run_id:
        for r in summaries:
            logger.info("%s", r.sentence)
    if orphaned:
        logger.warning("Orphaned (a firing queued extraction, then failed before its stages): %s", ", ".join(orphaned))
    return 0


COMMANDS = [
    Command("jobs", "Show cloud job-queue status", _register_jobs, cmd_jobs),
]
