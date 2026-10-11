"""Workflow CLI subcommand: ``run-workflow`` — run a stored workflow's steps over a local shard dir."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from womblex.cli._shared import Command, add_models_check_argument

logger = logging.getLogger("womblex")


def _register_run_workflow(p: argparse.ArgumentParser) -> None:
    p.add_argument("workflow_id", help="Stored workflow to run")
    p.add_argument("--shards", type=Path, required=True,
                   help="Local shard dir the steps run over (extraction has already produced it).")
    p.add_argument("--dsn", default=None,
                   help="Database the workflow is stored in: the DBOS system database URL "
                        "(or $WOMBLEX_DB_DSN / $DATABASE_URL). Default: the local SQLite file.")
    p.add_argument("--presets-dir", type=Path, default=None,
                   help="Directory of operator-saved presets (or $WOMBLEX_UI_PRESETS_DIR). "
                        "Built-in presets always resolve.")
    p.add_argument("--ingest", default=None,
                   help="`layout` step only: a local corpus root, when it is not where the manifest recorded it.")
    p.add_argument("--force", action="store_true",
                   help="Re-run bases whose declared outputs already exist")
    add_models_check_argument(p)


def cmd_run_workflow(args: argparse.Namespace) -> int:
    """Run a stored workflow's steps in order; exit 1 if it was refused or any step failed."""
    from womblex.cloud.workflow_runner import WorkflowRefusedError, run_workflow_local
    from womblex.cloud.workflow_store import WorkflowStore
    from womblex.ui import readers
    from womblex.ui.deps import UISettings

    if not args.shards.is_dir():
        logger.error("--shards path is not a directory: %s", args.shards)
        return 1
    dsn = args.dsn or os.environ.get("WOMBLEX_DB_DSN") or os.environ.get("DATABASE_URL")
    try:
        with WorkflowStore(dsn) as store:
            stored = store.get(args.workflow_id)
    except Exception as e:
        logger.error("could not read workflow %r: %s", args.workflow_id, e)
        return 1
    if stored is None:
        logger.error("no stored workflow %r", args.workflow_id)
        return 1

    presets_dir = args.presets_dir
    if presets_dir is None and "WOMBLEX_UI_PRESETS_DIR" in os.environ:
        presets_dir = Path(os.environ["WOMBLEX_UI_PRESETS_DIR"])
    settings = UISettings(output_root=args.shards, store_uri=None, presets_dir=presets_dir)
    try:
        result = run_workflow_local(
            stored, args.shards,
            resolve_preset=lambda name: readers.get_any_preset(settings, name),
            source_root=args.ingest, force=args.force, models_check=args.models_check,
        )
    except WorkflowRefusedError as e:
        logger.error("%s", e)
        return 1
    result.log()
    return result.exit_code


COMMANDS = [
    Command("run-workflow", "Run a stored workflow's steps over a local shard directory",
            _register_run_workflow, cmd_run_workflow),
]
