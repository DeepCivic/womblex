"""Standalone-ingest CLI subcommands: ``ingest-gnaf`` (PSV → Parquet) and
``ingest-geo`` (Shapefile → GeoParquet). Both bypass the NLP pipeline."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from womblex.cli._shared import Command

logger = logging.getLogger("womblex")


# --- ingest-gnaf -------------------------------------------------------------


def _register_ingest_gnaf(p: argparse.ArgumentParser) -> None:
    p.add_argument("input", help="Root directory of G-NAF PSV distribution")
    p.add_argument("-o", "--output", default="output/gnaf", help="Output directory for Parquet files")
    p.add_argument("--no-md5", action="store_true", help="Skip MD5 checksum computation")


def cmd_ingest_gnaf(args: argparse.Namespace) -> int:
    """Ingest G-NAF PSV files into Parquet."""
    from womblex.ingest.gnaf import ingest_gnaf_directory

    root = Path(args.input)
    output_dir = Path(args.output)
    if not root.exists():
        logger.error("Input directory does not exist: %s", root)
        return 1

    written = ingest_gnaf_directory(root, output_dir, compute_md5=not args.no_md5)
    if not written:
        logger.error("No files were written. Check logs for details.")
        return 1

    logger.info("Wrote %d Parquet files to %s", len(written), output_dir)
    return 0


# --- ingest-geo --------------------------------------------------------------


def _register_ingest_geo(p: argparse.ArgumentParser) -> None:
    p.add_argument("input", help="Root directory containing .shp files")
    p.add_argument("-o", "--output", default="output/geo", help="Output directory for GeoParquet files")
    p.add_argument("--no-md5", action="store_true", help="Skip MD5 checksum computation")


def cmd_ingest_geo(args: argparse.Namespace) -> int:
    """Ingest geospatial Shapefiles into GeoParquet."""
    from womblex.ingest.geospatial import ingest_geospatial_directory

    root = Path(args.input)
    output_dir = Path(args.output)
    if not root.exists():
        logger.error("Input directory does not exist: %s", root)
        return 1

    results = ingest_geospatial_directory(root, output_dir, compute_md5=not args.no_md5)
    succeeded = sum(1 for r in results if r.output is not None)
    if not succeeded:
        logger.error("No files were written. Check logs for details.")
        return 1

    logger.info("Wrote %d GeoParquet files to %s", succeeded, output_dir)
    return 0


COMMANDS = [
    Command("ingest-gnaf", "Ingest G-NAF PSV files into Parquet", _register_ingest_gnaf, cmd_ingest_gnaf),
    Command("ingest-geo", "Ingest Shapefiles into GeoParquet", _register_ingest_geo, cmd_ingest_geo),
]
