"""Verify CLI subcommand: ``verify-shards`` (directory-level shard audit)."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from womblex.cli._shared import Command

logger = logging.getLogger("womblex")


def _resolve_shard_dir(arg: Path) -> Path | None:
    """Accept either a run root (`output/run-…/`) or a shard dir.

    Returns the path to use, or None if neither shape matches.
    """
    if not arg.is_dir():
        return None
    documents = arg / "documents"
    if documents.is_dir():
        return documents
    # Already a shard dir if any manifest sibling exists
    if any(arg.glob("*._manifest.parquet")):
        return arg
    return None


def _register_verify_shards(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "run_dir", type=Path,
        help="Run root (containing 'documents/') or a shard directory.",
    )
    p.add_argument(
        "--compare-to", type=Path, default=None, action="append",
        help="Additional run/shard dir to diff against. May be repeated.",
    )
    p.add_argument(
        "--input-dir", type=Path, default=None,
        help="Source-file directory; enables source-count comparison.",
    )
    p.add_argument(
        "--format", choices=["text", "json"], default="text",
        help="Output format (json only valid for single-run audit).",
    )


def cmd_verify_shards(args: argparse.Namespace) -> int:
    """Audit shard directory integrity; optionally diff against other runs."""
    from womblex.store.shard_audit import (
        audit_shard_directory,
        format_audit_diff,
        format_audit_json,
        format_audit_text,
    )

    primary = _resolve_shard_dir(args.run_dir)
    if primary is None:
        logger.error("not a run or shard dir: %s", args.run_dir)
        return 1

    primary_report = audit_shard_directory(primary, input_dir=args.input_dir)

    if args.compare_to:
        if args.format == "json":
            logger.error("--format json is not supported with --compare-to")
            return 1
        reports = {args.run_dir.name: primary_report}
        for other in args.compare_to:
            other_shards = _resolve_shard_dir(other)
            if other_shards is None:
                logger.error("not a run or shard dir: %s", other)
                return 1
            reports[other.name] = audit_shard_directory(
                other_shards, input_dir=args.input_dir,
            )
        print(format_audit_diff(reports))
    elif args.format == "json":
        print(format_audit_json(primary_report))
    else:
        print(format_audit_text(primary_report))

    corrupted = primary_report.scan.corrupted_batches
    if corrupted:
        logger.warning(
            "verify-shards: %d corrupted batch(es) in %s",
            len(corrupted), primary,
        )
        return 2
    return 0


COMMANDS = [
    Command(
        "verify-shards",
        "Audit shard directory integrity; optionally diff across runs",
        _register_verify_shards,
        cmd_verify_shards,
    ),
]
