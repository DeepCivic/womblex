"""CLI: ``stats`` — statistical figure annotation over an extraction shard dir.

Per-stage (``--shards``) command mirroring ``womblex money``: reads
``*.elements.parquet`` + ``*.table_cells.parquet`` and writes
``*.stat_spans.parquet`` per batch (reference periods, the measures a corpus
vocabulary names, the figures stated beside them, and calculation / source /
gap / forecast cues). Annotation only — element and chunk text are untouched.
Offline (no API). See ``docs/stats-extraction.md``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from womblex.cli._shared import Command

logger = logging.getLogger("womblex")


def _register_stats(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--shards", type=Path, required=True,
        help="Shard dir (`<run_id>/documents/`) with `*.elements.parquet`; "
             "writes `*.stat_spans.parquet` in place.",
    )
    p.add_argument(
        "--config", type=Path, default=None,
        help="Optional YAML to source the `stats:` section (and processing.text_source). "
             "Defaults to StatsConfig() if omitted.",
    )
    p.add_argument(
        "--vocabulary", type=Path, default=None,
        help="YAML file of `measures` and `cues`; outranks the config's `stats.vocabulary`.",
    )
    p.add_argument(
        "--text-source", type=str, default=None,
        choices=["elements", "normalised", "spellfix"],
        help="Override the element-text layer narrative offsets index.",
    )
    p.add_argument("--checkpoint-dir", type=Path, default=None,
                   help="Per-stage checkpoint root (default: `<shard_dir>/../.stats-checkpoint/`).")
    p.add_argument("--dataset", type=str, default="stats",
                   help="Checkpoint dataset name. Default: 'stats'.")
    p.add_argument("--no-resume", action="store_true",
                   help="Clear stats-stage checkpoint before running (re-annotate every batch).")
    p.add_argument("--no-verify-resume", action="store_true",
                   help="Skip the resume-time `*.stat_spans.parquet` integrity scan.")


def cmd_stats(args: argparse.Namespace) -> int:
    """Annotate statistical figures over a shard directory (per-stage, offline)."""
    from pydantic import ValidationError

    from womblex.config import StatsConfig, load_config
    from womblex.process.stats_stage import stats_shards
    from womblex.process.stats_vocab import load_vocabulary
    from womblex.process.text_overlay import MissingOverlayError
    from womblex.store.checkpoint import CheckpointManager
    from womblex.store.shard_audit import reconcile_stage_checkpoint_with_shards
    from womblex.store.stats_output import STAT_SPANS_SUFFIX

    shard_dir: Path = args.shards
    if not shard_dir.is_dir():
        logger.error("--shards path is not a directory: %s", shard_dir)
        return 1
    if not any(shard_dir.glob("*.elements.parquet")):
        logger.error(
            "--shards directory has no `*.elements.parquet` — run extraction "
            "first: %s", shard_dir,
        )
        return 1

    text_source = "elements"
    if args.config:
        cfg = load_config(args.config)
        config, text_source = cfg.stats, cfg.processing.text_source
    else:
        # Naming the stage on the command line is asking for it; `enabled` gates dispatch.
        config = StatsConfig(enabled=True)
    updates: dict[str, object] = {}
    if args.vocabulary:
        updates["vocabulary"] = args.vocabulary
    if args.text_source:
        text_source = args.text_source
        updates["text_source"] = args.text_source
    if updates:
        config = config.model_copy(update=updates)
    if not config.enabled:
        logger.info("stats stage disabled in config; nothing to do.")
        return 0
    # A vocabulary that cannot be read refuses the stage before any batch is opened.
    try:
        load_vocabulary(config)
    except (OSError, ValidationError, ValueError) as e:
        logger.error("stats --shards: refused: %s", e)
        return 1

    checkpoint_root = args.checkpoint_dir or shard_dir.parent / ".stats-checkpoint"
    ckpt = CheckpointManager(checkpoint_root, f"{args.dataset}_stats")
    if args.no_resume:
        ckpt.clear()
    else:
        ckpt.load()
        if not args.no_verify_resume:
            dropped = reconcile_stage_checkpoint_with_shards(
                ckpt, shard_dir, suffix=STAT_SPANS_SUFFIX)
            if dropped:
                logger.warning("Resume integrity scan: dropped %d doc(s) with corrupted "
                               "stat_spans shards; they will be re-annotated.", len(dropped))

    logger.info(
        "stats --shards: dir=%s text_source=%s narrative=%s tables=%s vocabulary=%s",
        shard_dir, config.text_source or text_source, config.narrative, config.tables,
        config.vocabulary,
    )
    try:
        result = stats_shards(shard_dir, config, text_source=text_source, checkpoint_mgr=ckpt)
    except MissingOverlayError as e:
        logger.error("stats --shards: refused: %s", e)
        return 1
    logger.info(
        "Done: %d batches written, %d spans annotated (%d figures)",
        result.batches_written, result.spans_written, result.figures_written,
    )
    return 0


COMMANDS = [
    Command("stats", "Annotate statistical figures over a shard dir (per-stage, offline)",
            _register_stats, cmd_stats),
]
