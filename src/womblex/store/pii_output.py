"""Parquet IO for the PII stage (``*.pii_spans.parquet`` sidecar).

Self-contained like :mod:`womblex.store.enrichment_output`. One row per
detected PII span, located within a chunk: join to ``*.chunks.parquet`` on
``(source_hash, chunk_index)`` and slice ``text[start:end]``. **Sidecar-only** —
spans are NOT applied to the chunk text here (clean_text rewrite is a later
iteration). ``detector`` records provenance (``enrichment`` from the Kanon-2
graph | ``regex_high`` | ``regex_context``) so consumers can filter by source.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from womblex.store.output import _write_rows
from womblex.store.run_stamp import sidecar_footer

logger = logging.getLogger(__name__)

PII_SPANS_SUFFIX = ".pii_spans.parquet"
CLEAN_TEXT_SUFFIX = ".clean_text.parquet"

PII_SPANS_SCHEMA = pa.schema([
    ("source_hash", pa.string()),
    ("chunk_index", pa.int32()),
    ("content_type", pa.string()),
    ("start", pa.int32()),
    ("end", pa.int32()),
    ("text", pa.string()),
    ("entity_type", pa.string()),
    ("entity_id", pa.string()),     # graph entity id ("" for regex spans)
    ("detector", pa.string()),
    ("score", pa.float32()),
    ("replacement", pa.string()),   # the tag the text becomes, e.g. <PERSON_1>
])

# Masked publishable text layer (pii stage, terminal — applied AFTER Isaacus).
# One row per chunk (masked where PII spans were found, verbatim passthrough
# otherwise) so it is a drop-in replacement for `*.chunks.parquet`. Join back on
# (source_hash, chunk_index). `n_masked` = spans replaced in this chunk.
CLEAN_TEXT_SCHEMA = pa.schema([
    ("source_hash", pa.string()),
    ("chunk_index", pa.int32()),
    ("content_type", pa.string()),
    ("text", pa.string()),
    ("n_masked", pa.int32()),
])


def pii_spans_path_for(base_path: Path) -> Path:
    """Return ``<base>.pii_spans.parquet`` sibling for a shard base path."""
    return base_path.parent / f"{base_path.stem}{PII_SPANS_SUFFIX}"


def write_pii_spans(rows: list[dict], output_path: Path) -> Path:
    """Write a batch's PII-span rows to ``batch-NNNN.pii_spans.parquet``.

    ``rows`` must match :data:`PII_SPANS_SCHEMA`. Empty input produces an
    empty-but-schema-correct file so downstream readers can glob safely.
    """
    target = pii_spans_path_for(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(rows, target, PII_SPANS_SCHEMA, metadata=sidecar_footer(output_path, "pii"))
    logger.info("Wrote pii_spans shard %s: rows=%d", target.name, len(rows))
    return target


def read_pii_spans(path: Path) -> pa.Table:
    """Read PII spans from a single shard file or a shard-directory glob."""
    p = Path(path)
    if p.is_dir():
        shards = sorted(p.glob(f"*{PII_SPANS_SUFFIX}"))
        if not shards:
            return pa.table(
                {f.name: pa.array([], type=f.type) for f in PII_SPANS_SCHEMA},
                schema=PII_SPANS_SCHEMA,
            )
        return pa.concat_tables([_read_pii_spans_shard(s) for s in shards])
    pii_p = p if p.name.endswith(PII_SPANS_SUFFIX) else pii_spans_path_for(p)
    return _read_pii_spans_shard(pii_p)


def _read_pii_spans_shard(path: Path) -> pa.Table:
    raw = pq.read_table(str(path))
    missing = [f.name for f in PII_SPANS_SCHEMA if f.name not in raw.schema.names]
    if missing:
        raise ValueError(
            f"pii_spans shard {path} missing columns {missing}; schema bump without compat shim?"
        )
    return raw.select([f.name for f in PII_SPANS_SCHEMA]).cast(PII_SPANS_SCHEMA)


def clean_text_path_for(base_path: Path) -> Path:
    """Return ``<base>.clean_text.parquet`` sibling for a shard base path."""
    return base_path.parent / f"{base_path.stem}{CLEAN_TEXT_SUFFIX}"


def write_clean_text(rows: list[dict], output_path: Path) -> Path:
    """Write a batch's masked-chunk rows to ``batch-NNNN.clean_text.parquet``.

    ``rows`` must match :data:`CLEAN_TEXT_SCHEMA`. Empty input produces an
    empty-but-schema-correct file so downstream readers can glob safely.
    """
    target = clean_text_path_for(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(rows, target, CLEAN_TEXT_SCHEMA, metadata=sidecar_footer(output_path, "pii"))
    logger.info("Wrote clean_text shard %s: rows=%d", target.name, len(rows))
    return target


def read_clean_text(path: Path) -> pa.Table:
    """Read masked clean_text from a single shard file or a shard-directory glob."""
    p = Path(path)
    if p.is_dir():
        shards = sorted(p.glob(f"*{CLEAN_TEXT_SUFFIX}"))
        if not shards:
            return pa.table(
                {f.name: pa.array([], type=f.type) for f in CLEAN_TEXT_SCHEMA},
                schema=CLEAN_TEXT_SCHEMA,
            )
        return pa.concat_tables([_read_clean_text_shard(s) for s in shards])
    ct_p = p if p.name.endswith(CLEAN_TEXT_SUFFIX) else clean_text_path_for(p)
    return _read_clean_text_shard(ct_p)


def _read_clean_text_shard(path: Path) -> pa.Table:
    raw = pq.read_table(str(path))
    missing = [f.name for f in CLEAN_TEXT_SCHEMA if f.name not in raw.schema.names]
    if missing:
        raise ValueError(
            f"clean_text shard {path} missing columns {missing}; schema bump without compat shim?"
        )
    return raw.select([f.name for f in CLEAN_TEXT_SCHEMA]).cast(CLEAN_TEXT_SCHEMA)


__all__ = [
    "CLEAN_TEXT_SCHEMA",
    "CLEAN_TEXT_SUFFIX",
    "PII_SPANS_SCHEMA",
    "PII_SPANS_SUFFIX",
    "clean_text_path_for",
    "pii_spans_path_for",
    "read_clean_text",
    "read_pii_spans",
    "write_clean_text",
    "write_pii_spans",
]
