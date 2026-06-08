"""Parquet IO for the OCR-repair stage (``womblex spellfix``).

Two siblings per batch, both self-contained like :mod:`womblex.store.pii_output`:

- ``*.chunks_repaired.parquet`` — the repaired chunk layer. Reuses the chunk
  schema (:data:`womblex.store.output.CHUNKS_SCHEMA`); a complete passthrough
  layer (verbatim where nothing fired), parallel to ``*.chunks.parquet`` which
  is left untouched. Downstream consumers opt in to the repaired layer.
- ``*.spellfix_corrections.parquet`` — the audit trail: one row per applied
  rewrite (original → corrected), so every change is reviewable and the raw
  chunk text remains recoverable.

OCR repair runs *before* the Isaacus-facing chunk consumers (so the model sees
real words, not ``chi1d``) — the opposite ordering to PII masking, which is
terminal. See ``docs/decisions.md`` "Dictionary-gated OCR repair".
"""

from __future__ import annotations

import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from womblex.store.output import CHUNKS_SCHEMA, _write_rows

logger = logging.getLogger(__name__)

CHUNKS_REPAIRED_SUFFIX = ".chunks_repaired.parquet"
SPELLFIX_CORRECTIONS_SUFFIX = ".spellfix_corrections.parquet"

# Repaired chunks share the chunk schema exactly — it is a drop-in layer.
CHUNKS_REPAIRED_SCHEMA = CHUNKS_SCHEMA

SPELLFIX_CORRECTIONS_SCHEMA = pa.schema([
    ("source_hash", pa.string()),
    ("chunk_index", pa.int32()),
    ("content_type", pa.string()),
    ("offset", pa.int32()),       # char offset of the token within the chunk text
    ("original", pa.string()),
    ("corrected", pa.string()),
    ("method", pa.string()),      # homoglyph | edit1
])


def chunks_repaired_path_for(base_path: Path) -> Path:
    return base_path.parent / f"{base_path.stem}{CHUNKS_REPAIRED_SUFFIX}"


def spellfix_corrections_path_for(base_path: Path) -> Path:
    return base_path.parent / f"{base_path.stem}{SPELLFIX_CORRECTIONS_SUFFIX}"


def write_repaired_chunks(rows: list[dict], output_path: Path) -> Path:
    """Write a batch's repaired chunk rows (match :data:`CHUNKS_REPAIRED_SCHEMA`)."""
    target = chunks_repaired_path_for(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(rows, target, CHUNKS_REPAIRED_SCHEMA)
    logger.info("Wrote repaired chunks shard %s: rows=%d", target.name, len(rows))
    return target


def write_spellfix_corrections(rows: list[dict], output_path: Path) -> Path:
    """Write a batch's correction-audit rows (match :data:`SPELLFIX_CORRECTIONS_SCHEMA`)."""
    target = spellfix_corrections_path_for(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(rows, target, SPELLFIX_CORRECTIONS_SCHEMA)
    logger.info("Wrote spellfix corrections shard %s: rows=%d", target.name, len(rows))
    return target


def read_repaired_chunks(path: Path) -> pa.Table:
    """Read repaired chunks from a single shard file or a shard-directory glob."""
    p = Path(path)
    if p.is_dir():
        shards = sorted(p.glob(f"*{CHUNKS_REPAIRED_SUFFIX}"))
        if not shards:
            return _empty(CHUNKS_REPAIRED_SCHEMA)
        return pa.concat_tables([_read_shard(s, CHUNKS_REPAIRED_SCHEMA) for s in shards])
    rp = p if p.name.endswith(CHUNKS_REPAIRED_SUFFIX) else chunks_repaired_path_for(p)
    return _read_shard(rp, CHUNKS_REPAIRED_SCHEMA)


def read_spellfix_corrections(path: Path) -> pa.Table:
    """Read correction-audit rows from a single shard file or a directory glob."""
    p = Path(path)
    if p.is_dir():
        shards = sorted(p.glob(f"*{SPELLFIX_CORRECTIONS_SUFFIX}"))
        if not shards:
            return _empty(SPELLFIX_CORRECTIONS_SCHEMA)
        return pa.concat_tables([_read_shard(s, SPELLFIX_CORRECTIONS_SCHEMA) for s in shards])
    cp = p if p.name.endswith(SPELLFIX_CORRECTIONS_SUFFIX) else spellfix_corrections_path_for(p)
    return _read_shard(cp, SPELLFIX_CORRECTIONS_SCHEMA)


def _empty(schema: pa.Schema) -> pa.Table:
    return pa.table({f.name: pa.array([], type=f.type) for f in schema}, schema=schema)


def _read_shard(path: Path, schema: pa.Schema) -> pa.Table:
    raw = pq.read_table(str(path))
    missing = [f.name for f in schema if f.name not in raw.schema.names]
    if missing:
        raise ValueError(f"spellfix shard {path} missing columns {missing}; schema bump?")
    return raw.select([f.name for f in schema]).cast(schema)


__all__ = [
    "CHUNKS_REPAIRED_SCHEMA",
    "CHUNKS_REPAIRED_SUFFIX",
    "SPELLFIX_CORRECTIONS_SCHEMA",
    "SPELLFIX_CORRECTIONS_SUFFIX",
    "chunks_repaired_path_for",
    "read_repaired_chunks",
    "read_spellfix_corrections",
    "spellfix_corrections_path_for",
    "write_repaired_chunks",
    "write_spellfix_corrections",
]
