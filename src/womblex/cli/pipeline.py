"""Pipeline CLI subcommands: ``run`` (full pipeline), ``extract`` (single file),
``chunk`` (extract + chunk a directory)."""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from womblex.cli._shared import Command, discover_files, format_eta

logger = logging.getLogger("womblex")


# --- run ---------------------------------------------------------------------


def _register_run(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=Path, required=True, help="Path to config YAML")
    p.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    p.add_argument("--limit", type=int, default=None, help="Max documents to process")
    p.add_argument("--skip", type=int, default=0, help="Skip first N documents")
    p.add_argument("--batch-size", type=int, default=None, help="Override config batch size")


def cmd_run(args: argparse.Namespace) -> int:
    """Run all pipeline stages using a config file."""
    from womblex.config import load_config
    from womblex.operations import (
        BatchResult,
        run_chunking,
        run_extraction,
        run_pii_cleaning,
        run_redaction,
        write_batch_parquet,
    )
    from womblex.store.checkpoint import CheckpointManager
    from womblex.store.output import ShardVerificationError, verify_shard_persistence

    config = load_config(args.config)
    logger.info("Loaded config: %s", config.dataset.name)

    input_root = config.paths.input_root
    if not input_root.exists():
        logger.error("Input directory does not exist: %s", input_root)
        return 1

    all_files = discover_files(input_root, args.limit, args.skip)
    logger.info("Found %d documents to process", len(all_files))
    if not all_files:
        logger.error("No supported files found in %s", input_root)
        return 1

    output_root = config.paths.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    shard_dir = output_root / "documents"
    shard_dir.mkdir(parents=True, exist_ok=True)
    cumulative_shard_size = sum(s.stat().st_size for s in shard_dir.glob("*.parquet"))

    checkpoint_mgr = CheckpointManager(config.paths.checkpoint_dir, config.dataset.name)
    if args.resume:
        checkpoint_mgr.load()
        all_files = checkpoint_mgr.filter_unprocessed(all_files)
        logger.info("Resuming: %d documents remaining", len(all_files))
    else:
        checkpoint_mgr.clear()

    if not all_files:
        logger.info("All documents already processed")
        return 0

    batch_size = args.batch_size or config.processing.batch_size
    total_files = len(all_files)
    total_succeeded = 0
    total_failed = 0
    start_time = time.time()

    stages = ["extraction"]
    if config.redaction.enabled:
        stages.append(f"redaction({config.redaction.mode})")
    if config.pii.enabled:
        stages.append("pii")
    if config.chunking.enabled:
        stages.append("chunking")
    logger.info(
        "Starting pipeline: %d documents, batch_size=%d, stages=[%s]",
        total_files, batch_size, ", ".join(stages),
    )

    for batch_num, i in enumerate(range(0, total_files, batch_size), start=1):
        batch_files = all_files[i : i + batch_size]
        batch_start = time.time()
        logger.info(
            "[Batch %d] Processing %d documents (%d-%d of %d)...",
            batch_num, len(batch_files), i + 1, min(i + batch_size, total_files), total_files,
        )

        batch_results = run_extraction(batch_files, config)
        if config.redaction.enabled:
            batch_results = run_redaction(batch_results, config)
        if config.chunking.enabled:
            batch_results = run_chunking(batch_results, config)
        if config.pii.enabled:
            batch_results = run_pii_cleaning(batch_results, config)

        batch = BatchResult(results=batch_results)
        total_succeeded += batch.succeeded
        total_failed += batch.failed

        shard_path = shard_dir / f"batch-{batch_num:04d}.parquet"
        rows_to_write = sum(
            1 for r in batch.results if r.status == "completed" and r.extraction is not None
        )
        write_batch_parquet(batch, shard_path)

        if rows_to_write > 0:
            try:
                cumulative_shard_size = verify_shard_persistence(
                    shard_path, rows_to_write, cumulative_shard_size,
                )
            except ShardVerificationError as e:
                logger.error("[Batch %d] integrity check failed: %s", batch_num, e)
                raise

        doc_ids = [r.doc_id for r in batch.results]
        checkpoint_mgr.update(doc_ids, batch.succeeded, batch.failed, batch_num)

        batch_elapsed = time.time() - batch_start
        docs_done = i + len(batch_files)
        docs_remaining = total_files - docs_done
        total_elapsed = time.time() - start_time
        avg_per_doc = total_elapsed / docs_done if docs_done > 0 else 0
        logger.info(
            "[Batch %d] Complete: %d ok, %d failed (%.1fs, ETA: %s)",
            batch_num, batch.succeeded, batch.failed, batch_elapsed,
            format_eta(avg_per_doc * docs_remaining),
        )

    total_elapsed = time.time() - start_time
    logger.info(
        "Done in %s: %d succeeded, %d failed. Output: %s",
        format_eta(total_elapsed), total_succeeded, total_failed, output_root,
    )
    return 0


# --- extract -----------------------------------------------------------------


def _register_extract(p: argparse.ArgumentParser) -> None:
    p.add_argument("file", help="Path to document")
    p.add_argument("-o", "--output", default="output/", help="Output directory")
    p.add_argument(
        "--format",
        choices=["txt", "parquet"],
        default="txt",
        help="Output format: txt (one .txt per unit) or parquet (single .parquet file)",
    )


def cmd_extract(args: argparse.Namespace) -> int:
    """Extract text from a single document (extraction stage only)."""
    from womblex.config import (
        ChunkingConfig,
        DatasetConfig,
        ExtractionConfig,
        PathsConfig,
        RedactionConfig,
        WomblexConfig,
    )
    from womblex.operations import run_extraction

    path = Path(args.file)
    if not path.exists():
        logger.error("File not found: %s", path)
        return 1

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = WomblexConfig(
        dataset=DatasetConfig(name="single"),
        paths=PathsConfig(
            input_root=path.parent,
            output_root=output_dir,
            checkpoint_dir=output_dir / ".checkpoints",
        ),
        extraction=ExtractionConfig(),
        chunking=ChunkingConfig(enabled=False),
        redaction=RedactionConfig(enabled=False),
    )

    results = run_extraction([path], config)

    if args.format == "txt":
        completed = [r for r in results if r.status == "completed" and r.extraction]
        if len(completed) != 1:
            logger.error(
                "--format txt requires exactly 1 extraction unit, got %d. "
                "Use --format parquet for multi-unit files (e.g. spreadsheets).",
                len(completed),
            )
            return 1
        r = completed[0]
        out_path = output_dir / f"{r.doc_id}.txt"
        out_path.write_text(r.extraction.full_text, encoding="utf-8")
        logger.info("  %s -> %s (%d chars)", r.doc_id, out_path, len(r.extraction.full_text))
    else:
        from womblex.store.output import write_results

        rows = [
            (r.doc_id, str(r.path), r.extraction)
            for r in results
            if r.status == "completed" and r.extraction
        ]
        if rows:
            out_path = output_dir / f"{path.stem}.parquet"
            write_results(rows, out_path)
            logger.info("Wrote %d unit(s) to %s", len(rows), out_path)

    for r in results:
        if r.error:
            logger.error("  %s: %s", r.doc_id, r.error)

    ok = sum(1 for r in results if r.status == "completed")
    err = sum(1 for r in results if r.status == "error")
    logger.info("Extracted %d unit(s) (%d ok, %d errors)", len(results), ok, err)
    return 0 if err == 0 else 1


# --- chunk -------------------------------------------------------------------


def _register_chunk(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=Path, required=True, help="Path to config YAML")
    p.add_argument("--limit", type=int, default=None, help="Max documents to process")


def cmd_chunk(args: argparse.Namespace) -> int:
    """Chunk a directory of documents (extraction + chunking stages)."""
    from womblex.config import load_config
    from womblex.operations import BatchResult, run_chunking, run_extraction, write_batch_parquet

    config = load_config(args.config)
    if not config.chunking.enabled:
        logger.warning("Chunking is disabled in config. Set chunking.enabled: true to enable.")
        return 1

    input_root = config.paths.input_root
    if not input_root.exists():
        logger.error("Input directory does not exist: %s", input_root)
        return 1

    all_files = discover_files(input_root, args.limit)
    if not all_files:
        logger.error("No supported files found in %s", input_root)
        return 1

    output_root = config.paths.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    logger.info("Extraction stage: %d documents", len(all_files))
    results = run_extraction(all_files, config)

    logger.info("Chunking stage")
    results = run_chunking(results, config)

    batch = BatchResult(results=results)
    write_batch_parquet(batch, output_root / "documents.parquet")

    total_chunks = sum(len(r.chunks) for r in results if r.status == "completed")
    logger.info(
        "Done: %d ok, %d failed, %d total chunks",
        batch.succeeded, batch.failed, total_chunks,
    )
    return 0


COMMANDS = [
    Command("run", "Run all pipeline stages from a config file", _register_run, cmd_run),
    Command("extract", "Extract text from a single document", _register_extract, cmd_extract),
    Command("chunk", "Extract and chunk documents", _register_chunk, cmd_chunk),
]
