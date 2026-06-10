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
    p.add_argument(
        "--no-verify-resume", action="store_true",
        help="Skip the resume-time shard integrity scan (advanced — only use "
             "if you've already verified shards yourself).",
    )
    p.add_argument("--limit", type=int, default=None, help="Max documents to process")
    p.add_argument("--skip", type=int, default=0, help="Skip first N documents")
    p.add_argument("--batch-size", type=int, default=None, help="Override config batch size")
    p.add_argument(
        "--run-id", type=str, default=None,
        help=(
            "Identifier for this run instance (overrides dataset.run_id in config). "
            "Outputs land under <output_root>/<run_id>/documents/. "
            "If omitted, config value or an auto-generated timestamp is used."
        ),
    )


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
    from womblex.store.retention import apply_retention, generate_run_id, most_recent_run
    from womblex.store.shard_audit import reconcile_checkpoint_with_shards
    from womblex.utils.availability import isaacus_available

    config = load_config(args.config)
    logger.info("Loaded config: %s", config.dataset.name)

    # Pre-flight composition check: `run` has no enrichment stage, so graph-driven
    # (post_enrichment) PII can never satisfy its precondition here and would raise
    # PreconditionError mid-run. Fail fast with guidance instead of crashing later.
    if config.pii.enabled and config.pii.pipeline_point == "post_enrichment":
        logger.error(
            "pii.pipeline_point='post_enrichment' is unsupported in `womblex run` "
            "(this path has no enrichment stage). Use the per-stage flow "
            "(`womblex enrich --shards` then `womblex pii --shards`), or set "
            "pii.pipeline_point to 'post_chunk' / 'post_extraction'."
        )
        return 1

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

    # Resolve run_id: CLI > config > (resume: most-recent existing) > auto-generated
    explicit_run_id = args.run_id or config.dataset.run_id
    if explicit_run_id:
        run_id = explicit_run_id
    elif args.resume:
        prev = most_recent_run(output_root)
        if prev is None:
            logger.error("--resume given but no existing run dir found under %s", output_root)
            return 1
        run_id = prev.name
        logger.info("--resume: picked most-recent run %s", run_id)
    else:
        run_id = generate_run_id()
    logger.info("run_id: %s", run_id)

    run_root = output_root / run_id
    shard_dir = run_root / "documents"
    shard_dir.mkdir(parents=True, exist_ok=True)
    cumulative_shard_size = sum(s.stat().st_size for s in shard_dir.glob("*.parquet"))

    # Apply retention before processing (only on fresh runs — never on resume,
    # which would risk purging newer runs the user is intentionally bypassing).
    if not args.resume:
        purged = apply_retention(
            output_root,
            config.paths.checkpoint_dir,
            current_run_id=run_id,
            policy=config.processing.retention.policy,
            keep=config.processing.retention.keep,
        )
        if purged:
            logger.info(
                "retention(%s, keep=%d): purged %d old run(s)",
                config.processing.retention.policy,
                config.processing.retention.keep,
                len(purged),
            )

    checkpoint_root = config.paths.checkpoint_dir / run_id
    checkpoint_mgr = CheckpointManager(checkpoint_root, config.dataset.name)
    # Offset the batch counter when resuming so the shard path
    # (batch-NNNN.parquet) continues from where the prior invocation left
    # off — otherwise resumed batches overwrite earlier shards.
    batch_num_offset = 0
    if args.resume:
        checkpoint_mgr.load()
        if not args.no_verify_resume:
            dropped = reconcile_checkpoint_with_shards(checkpoint_mgr, shard_dir)
            if dropped:
                logger.warning(
                    "Resume integrity scan: dropped %d doc(s) with corrupted shards; "
                    "they will be re-extracted.", len(dropped),
                )
                # Recompute cumulative size after archive renames so the
                # post-write shrink check doesn't trip on the archived files.
                cumulative_shard_size = sum(
                    s.stat().st_size for s in shard_dir.glob("*.parquet")
                )
        batch_num_offset = checkpoint_mgr.state.last_batch
        all_files = checkpoint_mgr.filter_unprocessed(all_files)
        logger.info("Resuming: %d documents remaining (next batch=%d)", len(all_files), batch_num_offset + 1)
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

    chunk_will_skip = config.chunking.enabled and not isaacus_available()
    stages = ["extraction"]
    if config.redaction.enabled:
        stages.append(f"redaction({config.redaction.mode})")
    if config.pii.enabled:
        stages.append("pii")
    if config.chunking.enabled:
        stages.append("chunking(skipped: no Isaacus)" if chunk_will_skip else "chunking")
    logger.info(
        "Starting pipeline: %d documents, batch_size=%d, stages=[%s]",
        total_files, batch_size, ", ".join(stages),
    )
    if chunk_will_skip:
        logger.warning(
            "chunking.enabled but the Isaacus API is unavailable (needs the isaacus "
            "SDK + ISAACUS_API_KEY) — the chunk stage will be skipped for every batch; "
            "no chunks will be written."
        )

    for batch_idx, i in enumerate(range(0, total_files, batch_size), start=1):
        batch_num = batch_idx + batch_num_offset
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

    from womblex.store.run_manifest import write_run_manifest

    manifest_path = write_run_manifest(shard_dir)

    total_elapsed = time.time() - start_time
    logger.info(
        "Done in %s: %d succeeded, %d failed. Output: %s (manifest: %s)",
        format_eta(total_elapsed), total_succeeded, total_failed, output_root, manifest_path,
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
        assert r.extraction is not None  # a completed unit always carries extraction
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
    p.add_argument(
        "--shards", type=Path,
        help=(
            "Per-stage mode: chunk an existing shard directory "
            "(``<run_id>/documents/`` containing ``*.elements.parquet``). "
            "Writes ``*.chunks.parquet`` siblings in place."
        ),
    )
    p.add_argument(
        "--config", type=Path,
        help=(
            "Config YAML. With --shards: sources chunking settings "
            "(tokenizer, chunk_size, chunking_model for AI chunking) and "
            "processing.text_source. Without --shards: E2E composition mode "
            "(extraction + chunking)."
        ),
    )
    p.add_argument(
        "--checkpoint-dir", type=Path, default=None,
        help=(
            "Per-stage checkpoint root (default: ``<shard_dir>/../.chunk-checkpoint/``). "
            "Only used with --shards."
        ),
    )
    p.add_argument(
        "--dataset", type=str, default="chunk",
        help="Checkpoint dataset name (only used with --shards). Default: 'chunk'.",
    )
    p.add_argument(
        "--no-resume", action="store_true",
        help="Clear chunk-stage checkpoint before running (re-chunk every batch).",
    )
    p.add_argument(
        "--no-verify-resume", action="store_true",
        help=(
            "Skip the resume-time *.chunks.parquet integrity scan (advanced — "
            "only use if you've already verified the chunks files yourself)."
        ),
    )
    p.add_argument("--limit", type=int, default=None, help="Max documents to process (--config only)")


def cmd_chunk(args: argparse.Namespace) -> int:
    """Chunk a shard directory (per-stage) or a config-described corpus (E2E)."""
    if args.shards is not None:
        return _cmd_chunk_shards(args)
    if args.config is not None:
        return _cmd_chunk_config(args)
    logger.error("chunk requires --shards (per-stage) and/or --config (E2E)")
    return 1


def _cmd_chunk_shards(args: argparse.Namespace) -> int:
    from womblex.config import ChunkingConfig, load_config
    from womblex.process.chunk_stage import chunk_shards
    from womblex.store.checkpoint import CheckpointManager
    from womblex.store.shard_audit import reconcile_chunk_checkpoint_with_shards

    shard_dir: Path = args.shards
    if not shard_dir.is_dir():
        logger.error("--shards path is not a directory: %s", shard_dir)
        return 1
    if not any(shard_dir.glob("*._manifest.parquet")):
        logger.error("--shards directory has no `*._manifest.parquet`: %s", shard_dir)
        return 1

    if args.config is not None:
        cfg = load_config(args.config)
        chunking_config = cfg.chunking
        text_source = cfg.processing.text_source
    else:
        chunking_config = ChunkingConfig()
        text_source = "elements"

    checkpoint_root = args.checkpoint_dir or shard_dir.parent / ".chunk-checkpoint"
    ckpt = CheckpointManager(checkpoint_root, f"{args.dataset}_chunk")
    if args.no_resume:
        ckpt.clear()
    else:
        ckpt.load()
        if not args.no_verify_resume:
            dropped = reconcile_chunk_checkpoint_with_shards(ckpt, shard_dir)
            if dropped:
                logger.warning(
                    "Resume integrity scan: dropped %d doc(s) with corrupted "
                    "chunks shards; they will be re-chunked.", len(dropped),
                )

    logger.info(
        "chunk --shards: dir=%s tokenizer=%s chunk_size=%d processes=%d text_source=%s",
        shard_dir, chunking_config.tokenizer, chunking_config.chunk_size,
        chunking_config.processes, text_source,
    )
    result = chunk_shards(shard_dir, chunking_config, text_source=text_source, checkpoint_mgr=ckpt)
    logger.info(
        "Done: %d batches written, %d docs chunked, %d total chunks",
        result.batches_written, result.docs_chunked, result.total_chunks,
    )
    return 0


def _cmd_chunk_config(args: argparse.Namespace) -> int:
    """E2E composition: extract + chunk in one pass (back-compat path)."""
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


# --- manifest ----------------------------------------------------------------


def _register_manifest(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--shards", type=Path, required=True,
        help="Shard dir (`<run_id>/documents/`) whose `*._manifest.parquet` "
             "sidecars are consolidated.",
    )
    p.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output parquet path (default: `<run_root>/manifest.parquet`).",
    )


def cmd_manifest(args: argparse.Namespace) -> int:
    """Consolidate per-batch manifests into one run-level documents table."""
    from womblex.store.run_manifest import write_run_manifest

    shard_dir: Path = args.shards
    if not shard_dir.is_dir():
        logger.error("--shards path is not a directory: %s", shard_dir)
        return 1
    if not any(shard_dir.glob("*._manifest.parquet")):
        logger.error("--shards directory has no `*._manifest.parquet`: %s", shard_dir)
        return 1
    write_run_manifest(shard_dir, args.output)
    return 0


COMMANDS = [
    Command("run", "Run all pipeline stages from a config file", _register_run, cmd_run),
    Command("extract", "Extract text from a single document", _register_extract, cmd_extract),
    Command("chunk", "Extract and chunk documents", _register_chunk, cmd_chunk),
    Command(
        "manifest", "Consolidate per-batch manifests into a run-level documents table",
        _register_manifest, cmd_manifest,
    ),
]
