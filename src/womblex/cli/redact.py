"""Redaction CLI subcommands: ``redact`` (extract + redact in-memory),
``annotate-redactions`` (batch detection over extracted shards),
``validate-redactions`` (detector sanity check vs a labels packet)."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from womblex.cli._shared import Command, discover_files

logger = logging.getLogger("womblex")


# --- redact (extract + redact in-memory; legacy single-batch CLI) ----------


def _register_redact(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=Path, required=True, help="Path to config YAML")
    p.add_argument("--limit", type=int, default=None, help="Max documents to process")


def cmd_redact(args: argparse.Namespace) -> int:
    """Run extraction + redaction on a document directory."""
    from womblex.config import load_config
    from womblex.operations import BatchResult, run_extraction, run_redaction, write_batch_parquet

    config = load_config(args.config)
    if not config.redaction.enabled:
        logger.warning("Redaction is disabled in config. Set redaction.enabled: true to enable.")
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
        batch.succeeded, batch.failed, redacted_count,
    )
    return 0


# --- annotate-redactions (batch over extracted shards) ----------------------


def _register_annotate_redactions(p: argparse.ArgumentParser) -> None:
    p.add_argument("shards", type=Path, help="Directory containing extracted *.elements.parquet + *._manifest.parquet")
    p.add_argument("pdfs", type=Path, help="Directory containing the source PDFs (resolved via manifest filename)")
    p.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output directory for *.redactions.parquet sidecars (default: same as shards)",
    )
    p.add_argument(
        "--checkpoint", type=Path, default=None,
        help="JSON checkpoint path for resumable runs (skips already-processed batches)",
    )
    p.add_argument("--dpi", type=int, default=150, help="Page render DPI for raster fallback")
    p.add_argument(
        "--max-area-ratio", type=float, default=0.05,
        help="Reject candidate regions larger than this fraction of the page",
    )


def cmd_annotate_redactions(args: argparse.Namespace) -> int:
    """Detect redactions across extracted shards; persist sparse sidecar parquets."""
    from womblex.config import RedactionConfig
    from womblex.redact.batch import annotate_redactions_for_shards

    if not args.shards.is_dir():
        logger.error("shards dir not found: %s", args.shards)
        return 1
    if not args.pdfs.is_dir():
        logger.error("pdfs dir not found: %s", args.pdfs)
        return 1

    config = RedactionConfig(dpi=args.dpi, max_area_ratio=args.max_area_ratio)
    summary = annotate_redactions_for_shards(
        shard_dir=args.shards,
        pdf_dir=args.pdfs,
        config=config,
        output_dir=args.output,
        checkpoint_path=args.checkpoint,
    )
    n_with = sum(1 for v in summary.values() if v > 0)
    total = sum(summary.values())
    logger.info(
        "annotated %d docs, %d with redactions, %d total regions",
        len(summary), n_with, total,
    )
    return 0


# --- validate-redactions (detector sanity check vs labels packet) -----------


def _register_validate_redactions(p: argparse.ArgumentParser) -> None:
    p.add_argument("--labels", type=Path, required=True, help="Labels packet directory (*.meta.json files)")
    p.add_argument("--pdfs", type=Path, required=True, help="Source PDF directory")
    p.add_argument(
        "--report", type=Path, default=None,
        help="JSON output path (else markdown table printed to stdout)",
    )
    p.add_argument("--dpi", type=int, default=150, help="Page render DPI for raster fallback")
    p.add_argument(
        "--max-area-ratio", type=float, default=0.05,
        help="Reject candidate regions larger than this fraction of the page",
    )


def cmd_validate_redactions(args: argparse.Namespace) -> int:
    """Run detection over PDFs in a labels packet; print or save per-doc summaries."""
    from womblex.config import RedactionConfig
    from womblex.redact.batch import validate_redactions_against_labels

    if not args.labels.is_dir():
        logger.error("labels dir not found: %s", args.labels)
        return 1
    if not args.pdfs.is_dir():
        logger.error("pdfs dir not found: %s", args.pdfs)
        return 1

    config = RedactionConfig(dpi=args.dpi, max_area_ratio=args.max_area_ratio)
    summaries = validate_redactions_against_labels(args.labels, args.pdfs, config)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(
            [
                {
                    "source_pdf": s.source_pdf,
                    "n_pages": s.n_pages,
                    "total_regions": s.total_regions,
                    "affected_pages": s.affected_pages,
                    "labelled_pages": s.labelled_pages,
                    "per_page_bboxes": s.per_page_bboxes,
                }
                for s in summaries
            ],
            indent=2,
            default=str,
        ))
        logger.info("wrote: %s", args.report)
    else:
        print("| source pdf | pages | regions | affected pages | labelled pages |")
        print("|---|---:|---:|---|---|")
        for s in summaries:
            short = s.source_pdf[:60]
            print(
                f"| {short} | {s.n_pages} | {s.total_regions} | "
                f"{s.affected_pages} | {s.labelled_pages} |"
            )

    logger.info("validated %d docs", len(summaries))
    return 0


COMMANDS = [
    Command("redact", "Extract and apply redaction handling", _register_redact, cmd_redact),
    Command(
        "annotate-redactions",
        "Detect redactions across extracted shards; write sidecar parquets",
        _register_annotate_redactions,
        cmd_annotate_redactions,
    ),
    Command(
        "validate-redactions",
        "Validate the redaction detector against a labels packet",
        _register_validate_redactions,
        cmd_validate_redactions,
    ),
]
