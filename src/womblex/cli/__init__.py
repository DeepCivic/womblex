"""Command-line interface for womblex.

Topic modules under ``womblex.cli`` each expose a ``COMMANDS`` list of
``Command`` records. ``main()`` aggregates them, wires up argparse subparsers,
and dispatches to the matching handler.

Usage:

    womblex run                  --config configs/example.yaml
    womblex extract              document.pdf -o output/
    womblex chunk                --shards <dir> [--config configs/example.yaml]
    womblex chunk                --config configs/example.yaml
    womblex enrich               --shards <dir> [--config configs/example.yaml]
    womblex graph-refresh        --shards <dir>   (offline mention→chunk edge rebuild)
    womblex embed                --shards <dir> [--config configs/example.yaml]
    womblex manifest             --shards <dir> [-o PATH]
    womblex money                --shards <dir> [--config configs/example.yaml]
    womblex normalise            --shards <dir> [--config configs/example.yaml]
    womblex redact               --shards <dir> --pdfs <dir> [--checkpoint PATH]
    womblex redact               --config configs/example.yaml
    womblex annotate-redactions  <shards> <pdfs> [--checkpoint PATH]   (deprecated alias)
    womblex validate-redactions  --labels DIR --pdfs DIR [--report PATH]
    womblex score                --labels DIR --shards DIR
    womblex ground-truth         --shards <dir> --out <dir> --config configs/ground-truth.yaml
    womblex verify-shards        <run-or-shard-dir> [--compare-to <other>]
    womblex resolve-source       <run-or-shard-dir> [--root DIR] [--hash H]
    womblex profile              <file> [--sample-rows N]
    womblex ingest-gnaf          <input> -o output/gnaf
    womblex ingest-geo           <input> -o output/geo
    womblex ingest-abn           <input> -o output/abn
    womblex serve                [--store URI --dsn DSN] [--port 8081]
    womblex api-token            --client NAME [--scope SCOPE ...]
    womblex seed-demo            [--output-root DIR | --store URI]
"""
from __future__ import annotations

import argparse
import logging
import sys
import traceback

from womblex.cli import (
    api_token,
    cloud,
    demo,
    embed,
    ground_truth,
    ingest,
    jobs,
    link,
    money,
    normalise,
    pii,
    pipeline,
    profile,
    quality,
    redact,
    retry,
    score,
    serve,
    spellfix,
    verify,
    workflow,
)
from womblex.cli._shared import setup_logging

logger = logging.getLogger("womblex")


ALL_COMMANDS = [
    *pipeline.COMMANDS,
    *cloud.COMMANDS,
    *jobs.COMMANDS,
    *retry.COMMANDS,
    *redact.COMMANDS,
    *link.COMMANDS,
    *embed.COMMANDS,
    *ground_truth.COMMANDS,
    *normalise.COMMANDS,
    *spellfix.COMMANDS,
    *quality.COMMANDS,
    *money.COMMANDS,
    *pii.COMMANDS,
    *ingest.COMMANDS,
    *score.COMMANDS,
    *profile.COMMANDS,
    *verify.COMMANDS,
    *api_token.COMMANDS,
    *serve.COMMANDS,
    *demo.COMMANDS,
    *workflow.COMMANDS,
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="womblex", description="Document extraction pipeline")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    parser.add_argument(
        "--log-format", choices=("text", "json"), default=None,
        help="Log format (default: $WOMBLEX_LOG_FORMAT, else text); json emits one object per line",
    )
    sub = parser.add_subparsers(dest="command")

    dispatch = {}
    for cmd in ALL_COMMANDS:
        sub_parser = sub.add_parser(cmd.name, help=cmd.help)
        cmd.register(sub_parser)
        dispatch[cmd.name] = cmd.handler

    args = parser.parse_args(argv)
    setup_logging(args.verbose, args.log_format)

    if args.command in dispatch:
        try:
            return dispatch[args.command](args)
        except Exception as e:
            if (report := _config_failure(e)) is None:
                raise
            logger.error("%s", report)
            return 1
    parser.print_help()
    return 0


def _config_failure(e: Exception) -> str | None:
    """`load_config`'s own failure (missing file, bad YAML, invalid config) as one line; None for any other error."""
    import yaml
    from pydantic import ValidationError

    from womblex.config import load_config

    if not isinstance(e, OSError | yaml.YAMLError | ValidationError):
        return None
    frame = next((f for f, _ in traceback.walk_tb(e.__traceback__) if f.f_code is load_config.__code__), None)
    if frame is None:
        return None
    if isinstance(e, ValidationError):
        detail = f"{e.error_count()} validation error(s): " + "; ".join(
            f"{'.'.join(map(str, err['loc']))}: {err['msg']}" if err["loc"] else err["msg"] for err in e.errors())
    else:
        detail = str(e)
    return f"config {frame.f_locals['path']} does not load: " + " ".join(s.strip() for s in detail.splitlines())


if __name__ == "__main__":
    sys.exit(main())
