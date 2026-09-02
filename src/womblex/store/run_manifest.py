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

import pyarrow as pa
import pyarrow.parquet as pq

from womblex.store.output import read_manifest
from womblex.store.run_stamp import stamp_from_footers
from womblex.store.source_provenance import IngestProvenance

logger = logging.getLogger(__name__)

RUN_MANIFEST_FILENAME = "manifest.parquet"


def _footer_from_columns(table: pa.Table) -> dict[bytes, bytes] | None:
    """Re-stamp the consolidated manifest's footer from the rows it consolidates.

    The per-batch shards were stamped at write time; consolidation reads them
    as columns, so the run manifest restates the same pair rather than needing
    the run's configuration handed to it a second time. A run whose shards
    predate the columns, or that mixes roots, leaves the footer off — a footer
    that named one of several roots would be worse than none.
    """
    if table.num_rows == 0:
        return None
    roots = set(table.column("ingest_root").to_pylist())
    collections = set(table.column("collection_id").to_pylist())
    if len(roots) != 1 or len(collections) != 1:
        return None
    root, collection = roots.pop(), collections.pop()
    if not root or not collection:
        return None
    prov = IngestProvenance(ingest_root=root, collection_id=collection)
    return prov.footer_metadata(
        r for r in table.column("source_relpath").to_pylist() if r
    )


def _merged_footer(shard_dir: Path, table: pa.Table) -> dict[bytes, bytes] | None:
    """The consolidated manifest's footer: where the corpus came from, and which run.

    The run is read back from the per-batch manifests' own stamps rather than
    recomputed, for the reason the batch sidecars inherit theirs — consolidation
    is handed a shard directory, not the run's configuration, and `womblex
    manifest` may be re-run long after. Batches naming more than one run leave
    the run keys off, the rule the ingest-root block above already follows.
    """
    stamp = stamp_from_footers(sorted(shard_dir.glob("*._manifest.parquet")), "manifest")
    merged: dict[bytes, bytes] = {}
    for part in (_footer_from_columns(table), stamp.footer_metadata() if stamp else None):
        if part:
            merged.update(part)
    return merged or None


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
    footer = _merged_footer(shard_dir, table)
    if footer:
        table = table.replace_schema_metadata(footer)
    target = output_path or run_manifest_path_for(shard_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(target), compression="zstd", compression_level=3)
    logger.info("Wrote run manifest %s: docs=%d", target, table.num_rows)
    return target


__all__ = ["RUN_MANIFEST_FILENAME", "run_manifest_path_for", "write_run_manifest"]
