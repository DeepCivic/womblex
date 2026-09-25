"""``source_index.parquet`` schema and IO for the egress bundle.

Self-contained, in the manner of the other ``store/*_output.py`` modules.
One row per document in the exported run (not deduplicated by
``source_hash`` — two documents sharing one hash each get their own row,
both naming the same ``raw_key``). ``status`` is drawn from
:mod:`womblex.store.source_resolver`'s vocabulary (``resolved`` /
``hash_mismatch`` / ``not_found`` / ``unsupported_basis``); ``raw_key`` is
null wherever no file was produced under ``sources/`` — an unresolved
document, or one hashed over record id + text rather than file bytes.

This module owns the schema and IO only. Building the bundle — resolving
sources, copying files, writing this index alongside ``egress_manifest.json``
— is ``store/egress.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

SOURCE_INDEX_FILENAME = "source_index.parquet"

SOURCE_INDEX_SCHEMA = pa.schema([
    ("source_hash", pa.string()),
    ("doc_id", pa.string()),
    ("filename", pa.string()),
    ("ext", pa.string()),
    ("raw_key", pa.string()),
    ("status", pa.string()),
])


def write_source_index(rows: list[dict], bundle_dir: Path) -> Path:
    """Write a bundle's ``source_index.parquet``.

    ``rows`` must match :data:`SOURCE_INDEX_SCHEMA`. Empty input still
    produces a schema-correct file so downstream readers can rely on it being
    present and typed. Returns the path written.
    """
    target = Path(bundle_dir) / SOURCE_INDEX_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=SOURCE_INDEX_SCHEMA)
    pq.write_table(table, str(target), compression="zstd", compression_level=3)
    logger.info("Wrote source index %s: rows=%d", target, table.num_rows)
    return target


def read_source_index(path: Path) -> pa.Table:
    """Read a bundle's source index, from the bundle directory or the file itself."""
    p = Path(path)
    target = p / SOURCE_INDEX_FILENAME if p.is_dir() else p
    raw = pq.read_table(str(target))
    missing = [f.name for f in SOURCE_INDEX_SCHEMA if f.name not in raw.schema.names]
    if missing:
        raise ValueError(
            f"source_index {target} missing columns {missing}; schema bump without compat shim?"
        )
    return raw.select([f.name for f in SOURCE_INDEX_SCHEMA]).cast(SOURCE_INDEX_SCHEMA)


__all__ = [
    "SOURCE_INDEX_FILENAME",
    "SOURCE_INDEX_SCHEMA",
    "read_source_index",
    "write_source_index",
]
