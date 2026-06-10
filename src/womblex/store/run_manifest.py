"""Run-level document manifest consolidation.

The per-batch ``batch-NNNN._manifest.parquet`` sidecars are the only
mapping from ``source_hash`` (the join key on every other sidecar) back
to the source document (``doc_id`` / ``filename``). Consumers of a
shipped run need that mapping in one place, so the end of ``womblex run``
— and the standalone ``womblex manifest`` command — consolidate them
into a single ``manifest.parquet`` at the run root, one row per document
(``MANIFEST_SCHEMA``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pyarrow.parquet as pq

from womblex.store.output import read_manifest

logger = logging.getLogger(__name__)

RUN_MANIFEST_FILENAME = "manifest.parquet"


def run_manifest_path_for(shard_dir: Path) -> Path:
    """Default consolidated-manifest path: ``<run_root>/manifest.parquet``."""
    return shard_dir.parent / RUN_MANIFEST_FILENAME


def write_run_manifest(shard_dir: Path, output_path: Path | None = None) -> Path:
    """Consolidate all ``*._manifest.parquet`` in ``shard_dir`` into one parquet.

    Writes to ``output_path`` (default ``<run_root>/manifest.parquet``) and
    returns the path written. An empty shard directory still produces an
    empty-but-schema-correct file so downstream reads are safe.
    """
    table = read_manifest(shard_dir)
    target = output_path or run_manifest_path_for(shard_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(target), compression="zstd", compression_level=3)
    logger.info("Wrote run manifest %s: docs=%d", target, table.num_rows)
    return target


__all__ = ["RUN_MANIFEST_FILENAME", "run_manifest_path_for", "write_run_manifest"]
