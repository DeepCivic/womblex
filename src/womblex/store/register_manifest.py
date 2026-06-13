"""Register-ingest manifest: a glob-free index of standalone-ingest output.

The standalone register ingests (``ingest-abn``, ``ingest-gnaf``,
``ingest-geo``) write one or more Parquet files per source into a flat
output directory, with no index. Consumers then have to glob fragile
patterns (``*_names.parquet`` vs the records files) to tell the outputs
apart. This module consolidates a directory of register outputs into a
single ``manifest.parquet`` — one row per output file — so discovery is
deterministic and glob-free.

It is deliberately separate from ``store/run_manifest.py``: that manifest
is the NLP pipeline's ``source_hash → doc_id`` document table and its
schema is extraction-specific. Register outputs have no extraction
metadata, so they get their own generic schema here. Self-contained and
SDK-free, mirroring ``store/run_manifest.py``'s shape.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

RUN_MANIFEST_FILENAME = "manifest.parquet"

# One row per output Parquet file. ``role`` distinguishes the records vs
# names sidecars (ABN) or the table type (G-NAF) without globbing.
REGISTER_MANIFEST_SCHEMA = pa.schema([
    ("source_file", pa.string()),
    ("output_file", pa.string()),
    ("role", pa.string()),
    ("row_count", pa.int64()),
    ("schema_version", pa.string()),
    ("source_md5", pa.string()),
    ("ingested_at_iso", pa.string()),
])

# Key-value metadata namespaces the register ingests attach to their
# Parquet footers (``abn.*`` / ``gnaf.*`` / ``geo.*``).
_KNOWN_NAMESPACES = ("abn", "gnaf", "geo")


def _decode_meta(raw: dict[bytes, bytes] | None) -> dict[str, str]:
    if not raw:
        return {}
    return {k.decode(): v.decode() for k, v in raw.items()}


def _namespace(meta: dict[str, str]) -> str | None:
    for key in meta:
        ns = key.split(".", 1)[0]
        if ns in _KNOWN_NAMESPACES:
            return ns
    return None


def _role_for(output_path: Path, ns: str | None, meta: dict[str, str]) -> str:
    """Derive a role label without globbing.

    G-NAF tags its table type in metadata; ABN names sidecars end in
    ``_names``; everything else is the primary records output.
    """
    if ns is not None:
        table = meta.get(f"{ns}.table_name")
        if table:
            return table
    if output_path.stem.endswith("_names"):
        return "names"
    return "records"


def _manifest_row(output_path: Path) -> dict[str, object]:
    md = pq.read_metadata(str(output_path))
    meta = _decode_meta(md.metadata)
    ns = _namespace(meta)
    source_file = meta.get(f"{ns}.source_file", "") if ns else ""
    schema_version = meta.get(f"{ns}.schema_version", "") if ns else ""
    source_md5 = meta.get(f"{ns}.source_md5") if ns else None
    return {
        "source_file": source_file,
        "output_file": output_path.name,
        "role": _role_for(output_path, ns, meta),
        "row_count": md.num_rows,
        "schema_version": schema_version,
        "source_md5": source_md5,
        "ingested_at_iso": datetime.now(timezone.utc).isoformat(),
    }


def write_register_manifest(output_dir: Path) -> Path:
    """Index all register-ingest Parquet outputs in *output_dir*.

    Scans ``*.parquet`` (excluding the manifest itself), reads each file's
    row count and footer metadata, and writes ``manifest.parquet`` into
    *output_dir*. Returns the path written. An empty directory still
    produces an empty, schema-correct file so downstream reads are safe.
    """
    files = sorted(
        p for p in output_dir.glob("*.parquet") if p.name != RUN_MANIFEST_FILENAME
    )
    rows = [_manifest_row(p) for p in files]
    table = pa.Table.from_pylist(rows, schema=REGISTER_MANIFEST_SCHEMA)
    target = output_dir / RUN_MANIFEST_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(target), compression="zstd", compression_level=3)
    logger.info("Wrote register manifest %s: outputs=%d", target, table.num_rows)
    return target


__all__ = [
    "REGISTER_MANIFEST_SCHEMA",
    "RUN_MANIFEST_FILENAME",
    "write_register_manifest",
]
