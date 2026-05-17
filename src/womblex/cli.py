"""Command-line interface for womblex.


Usage:

    womblex run     --config configs/example.yaml          # all stages

    womblex extract document.pdf -o output/               # single file

    womblex chunk   --config configs/example.yaml          # chunking stage

    womblex redact  --config configs/example.yaml          # redaction stage
"""

from __future__ import annotations

import argparse
import logging

import sys
import time

from pathlib import Path


logger = logging.getLogger("womblex")


SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".xlsx", ".xls", ".docx"}


def _setup_logging(verbose: bool = False) -> None:

    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def _format_eta(seconds: float) -> str:

    if seconds < 60:
        return f"{seconds:.0f}s"

    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"

    hours = int(seconds // 3600)

    mins = int((seconds % 3600) // 60)

    return f"{hours}h {mins}m"


def _discover_files(input_root: Path, limit: int | None = None, skip: int = 0) -> list[Path]:
    """Discover supported documents in input directory."""
    files = sorted(
        (f for f in input_root.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS),
        key=lambda p: p.name,
    )

    if skip:
        files = files[skip:]

    if limit:
        files = files[:limit]
    return files


def cmd_run(args: argparse.Namespace) -> int:
    """Run all pipeline stages using a config file."""

    from womblex.config import load_config

    from womblex.operations import (
        run_extraction,
        run_redaction,
        run_chunking,
        run_pii_cleaning,
        write_batch_parquet,
        BatchResult,
    )

    from womblex.store.checkpoint import CheckpointManager
    from womblex.store.output import verify_shard_persistence, ShardVerificationError

    config = load_config(args.config)

    logger.info("Loaded config: %s", config.dataset.name)

    input_root = config.paths.input_root

    if not input_root.exists():
        logger.error("Input directory does not exist: %s", input_root)

        return 1

    all_files = _discover_files(input_root, args.limit, args.skip)

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
        total_files,
        batch_size,
        ", ".join(stages),
    )

    for batch_num, i in enumerate(range(0, total_files, batch_size), start=1):
        batch_files = all_files[i : i + batch_size]

        batch_start = time.time()

        logger.info(
            "[Batch %d] Processing %d documents (%d-%d of %d)...",
            batch_num,
            len(batch_files),
            i + 1,
            min(i + batch_size, total_files),
            total_files,
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
            batch_num,
            batch.succeeded,
            batch.failed,
            batch_elapsed,
            _format_eta(avg_per_doc * docs_remaining),
        )

    total_elapsed = time.time() - start_time
    logger.info(
        "Done in %s: %d succeeded, %d failed. Output: %s",
        _format_eta(total_elapsed),
        total_succeeded,
        total_failed,
        output_root,
    )

    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    """Extract text from a single document (extraction stage only)."""

    from womblex.config import (
        ChunkingConfig,
        DatasetConfig,
        ExtractionConfig,
        PathsConfig,
        WomblexConfig,
        RedactionConfig,
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

        rows = [(r.doc_id, str(r.path), r.extraction) for r in results if r.status == "completed" and r.extraction]

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


def cmd_chunk(args: argparse.Namespace) -> int:
    """Chunk a directory of documents (extraction + chunking stages)."""

    from womblex.config import load_config

    from womblex.operations import run_chunking, run_extraction, write_batch_parquet, BatchResult

    config = load_config(args.config)

    if not config.chunking.enabled:
        logger.warning("Chunking is disabled in config. Set chunking.enabled: true to enable.")

        return 1

    input_root = config.paths.input_root

    if not input_root.exists():
        logger.error("Input directory does not exist: %s", input_root)

        return 1

    all_files = _discover_files(input_root, args.limit)

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
        batch.succeeded,
        batch.failed,
        total_chunks,
    )

    return 0


def cmd_redact(args: argparse.Namespace) -> int:
    """Run the redaction stage over a document directory."""

    from womblex.config import load_config

    from womblex.operations import run_extraction, run_redaction, write_batch_parquet, BatchResult

    config = load_config(args.config)

    if not config.redaction.enabled:
        logger.warning("Redaction is disabled in config. Set redaction.enabled: true to enable.")

        return 1

    input_root = config.paths.input_root

    if not input_root.exists():
        logger.error("Input directory does not exist: %s", input_root)

        return 1

    all_files = _discover_files(input_root, args.limit)

    if not all_files:
        logger.error("No supported files found in %s", input_root)

        return 1

    output_root = config.paths.output_root

    output_root.mkdir(parents=True, exist_ok=True)

    logger.info("Extraction stage: %d documents", len(all_files))

    results = run_extraction(all_files, config)

    logger.info("Redaction stage: mode=%s", config.redaction.mode)
    results = run_redaction(results, config)

    batch = BatchResult(results=results)

    write_batch_parquet(batch, output_root / "documents.parquet")

    redacted_count = sum(
        bool(
            r.status == "completed"
            and r.extraction
            and r.extraction.redaction_report
            and r.extraction.redaction_report.total > 0
        )
        for r in results
    )
    logger.info(
        "Done: %d ok, %d failed, %d documents with redactions",
        batch.succeeded,
        batch.failed,
        redacted_count,
    )

    return 0


def cmd_ingest_gnaf(args: argparse.Namespace) -> int:
    """Ingest G-NAF PSV files into Parquet."""

    from womblex.ingest.gnaf import ingest_gnaf_directory

    root = Path(args.input)

    output_dir = Path(args.output)

    if not root.exists():
        logger.error("Input directory does not exist: %s", root)

        return 1

    written = ingest_gnaf_directory(
        root,
        output_dir,
        compute_md5=not args.no_md5,
    )

    if not written:
        logger.error("No files were written. Check logs for details.")

        return 1

    logger.info("Wrote %d Parquet files to %s", len(written), output_dir)

    return 0


def cmd_ingest_geo(args: argparse.Namespace) -> int:
    """Ingest geospatial Shapefiles into GeoParquet."""

    from womblex.ingest.geospatial import ingest_geospatial_directory

    root = Path(args.input)

    output_dir = Path(args.output)

    if not root.exists():
        logger.error("Input directory does not exist: %s", root)

        return 1

    results = ingest_geospatial_directory(
        root,
        output_dir,
        compute_md5=not args.no_md5,
    )

    succeeded = sum(1 for r in results if r.output is not None)

    if not succeeded:
        logger.error("No files were written. Check logs for details.")

        return 1

    logger.info("Wrote %d GeoParquet files to %s", succeeded, output_dir)

    return 0


def cmd_score(args: argparse.Namespace) -> int:
    """Score human-reviewed labels against the parquet extraction output."""
    from womblex.score import format_report_markdown, score_labels

    labels_dir = Path(args.labels)
    shards_dir = Path(args.shards)
    if not labels_dir.is_dir():
        logger.error("labels dir not found: %s", labels_dir)
        return 1
    if not shards_dir.is_dir():
        logger.error("shards dir not found: %s", shards_dir)
        return 1

    try:
        rows = score_labels(labels_dir, shards_dir, group_by=args.group_by)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    if not rows:
        logger.error("no labels scored — check --labels content and shard manifests")
        return 1

    group_label = args.group_by or "group"
    report = format_report_markdown(rows, group_label=group_label)

    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        logger.info("wrote: %s", out)
    else:
        print(report)
    logger.info("scored: %d pages", len(rows))
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    """Sample a tabular file and print per-column inferred schema."""
    import json
    from dataclasses import asdict

    from womblex.profile import profile_file

    path = Path(args.file)
    if not path.exists():
        logger.error("File does not exist: %s", path)
        return 1

    try:
        profiles = profile_file(path, sample_rows=args.sample_rows)
    except ValueError as e:
        logger.error("%s", e)
        return 1

    if args.json:
        print(json.dumps([asdict(p) for p in profiles], indent=2))
        return 0

    for tp in profiles:
        header = f"{tp.source}"
        if tp.sheet_name is not None:
            header += f" [{tp.sheet_name}]"
        header += f"  {tp.row_count} rows, {tp.column_count} cols"
        if tp.sampled_rows < tp.row_count:
            header += f" (sampled {tp.sampled_rows})"
        print(header)
        for col in tp.columns:
            flags = []
            if col.is_unique:
                flags.append("unique")
            if col.is_constant:
                flags.append("constant")
            if col.null_fraction > 0:
                flags.append(f"{col.null_fraction:.1%} null")
            if col.inferred_type in ("integer", "float", "date", "datetime"):
                if col.min_value is not None:
                    flags.append(f"[{col.min_value}..{col.max_value}]")
            elif col.inferred_type == "string" and col.max_length is not None:
                flags.append(f"max-len={col.max_length}")
            print(f"  {col.name:<30} {col.inferred_type:<10} {'  '.join(flags)}")
        print()
    return 0


def main(argv: list[str] | None = None) -> int:

    parser = argparse.ArgumentParser(prog="womblex", description="Document extraction pipeline")

    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")

    sub = parser.add_subparsers(dest="command")

    # womblex run

    run_p = sub.add_parser("run", help="Run all pipeline stages from a config file")

    run_p.add_argument("--config", type=Path, required=True, help="Path to config YAML")

    run_p.add_argument("--resume", action="store_true", help="Resume from last checkpoint")

    run_p.add_argument("--limit", type=int, default=None, help="Max documents to process")

    run_p.add_argument("--skip", type=int, default=0, help="Skip first N documents")

    run_p.add_argument("--batch-size", type=int, default=None, help="Override config batch size")

    # womblex extract

    ext_p = sub.add_parser("extract", help="Extract text from a single document")

    ext_p.add_argument("file", help="Path to document")

    ext_p.add_argument("-o", "--output", default="output/", help="Output directory")

    ext_p.add_argument(
        "--format",
        choices=["txt", "parquet"],
        default="txt",
        help="Output format: txt (one .txt per unit) or parquet (single .parquet file)",
    )

    # womblex chunk

    chunk_p = sub.add_parser("chunk", help="Extract and chunk documents")

    chunk_p.add_argument("--config", type=Path, required=True, help="Path to config YAML")

    chunk_p.add_argument("--limit", type=int, default=None, help="Max documents to process")

    # womblex redact

    redact_p = sub.add_parser("redact", help="Extract and apply redaction handling")

    redact_p.add_argument("--config", type=Path, required=True, help="Path to config YAML")

    redact_p.add_argument("--limit", type=int, default=None, help="Max documents to process")

    # womblex ingest-gnaf

    gnaf_p = sub.add_parser("ingest-gnaf", help="Ingest G-NAF PSV files into Parquet")

    gnaf_p.add_argument("input", help="Root directory of G-NAF PSV distribution")

    gnaf_p.add_argument("-o", "--output", default="output/gnaf", help="Output directory for Parquet files")

    gnaf_p.add_argument("--no-md5", action="store_true", help="Skip MD5 checksum computation")

    # womblex ingest-geo

    geo_p = sub.add_parser("ingest-geo", help="Ingest Shapefiles into GeoParquet")

    geo_p.add_argument("input", help="Root directory containing .shp files")

    geo_p.add_argument("-o", "--output", default="output/geo", help="Output directory for GeoParquet files")

    geo_p.add_argument("--no-md5", action="store_true", help="Skip MD5 checksum computation")

    # womblex score
    score_p = sub.add_parser(
        "score",
        help="Score labels packet (*.gt.md + *.meta.json) against elements parquet",
    )
    score_p.add_argument(
        "--labels", type=Path, required=True,
        help="Directory containing *.gt.md + *.meta.json label files",
    )
    score_p.add_argument(
        "--shards", type=Path, required=True,
        help="Directory containing *.elements.parquet + *._manifest.parquet shards",
    )
    score_p.add_argument(
        "--report", type=Path, default=None,
        help="Output markdown path. Prints to stdout if omitted.",
    )
    score_p.add_argument(
        "--group-by", default=None,
        help="Meta field used to bucket the per-page summary (e.g. 'strategy')",
    )

    # womblex profile
    prof_p = sub.add_parser(
        "profile",
        help="Sample a tabular file and print inferred per-column schema",
    )
    prof_p.add_argument("file", help="Path to CSV / XLSX / XLS / Parquet / NDJSON")
    prof_p.add_argument(
        "--sample-rows",
        type=int,
        default=10_000,
        help="Maximum rows to load for inference (0 = read all). Default: 10000.",
    )
    prof_p.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable text")

    args = parser.parse_args(argv)

    _setup_logging(args.verbose)

    if args.command == "run":
        return cmd_run(args)

    elif args.command == "extract":
        return cmd_extract(args)

    elif args.command == "chunk":
        return cmd_chunk(args)

    elif args.command == "redact":
        return cmd_redact(args)

    elif args.command == "ingest-gnaf":
        return cmd_ingest_gnaf(args)

    elif args.command == "ingest-geo":
        return cmd_ingest_geo(args)

    elif args.command == "score":
        return cmd_score(args)

    elif args.command == "profile":
        return cmd_profile(args)

    else:
        parser.print_help()

        return 0


if __name__ == "__main__":
    sys.exit(main())
