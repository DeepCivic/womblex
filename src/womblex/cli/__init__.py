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
    womblex normalise            --shards <dir> [--config configs/example.yaml]
    womblex redact               --shards <dir> --pdfs <dir> [--checkpoint PATH]
    womblex redact               --config configs/example.yaml
    womblex annotate-redactions  <shards> <pdfs> [--checkpoint PATH]   (deprecated alias)
    womblex validate-redactions  --labels DIR --pdfs DIR [--report PATH]
    womblex score                --labels DIR --shards DIR
    womblex verify-shards        <run-or-shard-dir> [--compare-to <other>]
    womblex profile              <file> [--sample-rows N]
    womblex ingest-gnaf          <input> -o output/gnaf
    womblex ingest-geo           <input> -o output/geo
    womblex ingest-abn           <input> -o output/abn
"""
from __future__ import annotations

import argparse
import logging
import sys

from womblex.cli import (
    cloud,
    embed,
    ingest,
    link,
    normalise,
    pii,
    pipeline,
    profile,
    quality,
    redact,
    score,
    spellfix,
    verify,
)
from womblex.cli._shared import setup_logging

logger = logging.getLogger("womblex")


ALL_COMMANDS = [
    *pipeline.COMMANDS,
    *cloud.COMMANDS,
    *redact.COMMANDS,
    *link.COMMANDS,
    *embed.COMMANDS,
    *normalise.COMMANDS,
    *spellfix.COMMANDS,
    *quality.COMMANDS,
    *pii.COMMANDS,
    *ingest.COMMANDS,
    *score.COMMANDS,
    *profile.COMMANDS,
    *verify.COMMANDS,
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="womblex", description="Document extraction pipeline")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    sub = parser.add_subparsers(dest="command")

    dispatch = {}
    for cmd in ALL_COMMANDS:
        sub_parser = sub.add_parser(cmd.name, help=cmd.help)
        cmd.register(sub_parser)
        dispatch[cmd.name] = cmd.handler

    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    if args.command in dispatch:
        return dispatch[args.command](args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
