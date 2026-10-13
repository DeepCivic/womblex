"""Parquet IO for a workflow's condition labels (``*.conditions.parquet`` sidecar).

Self-contained like :mod:`womblex.store.pii_output`. One row per document and
condition, joined to the batch's ``*._manifest.parquet`` on ``source_hash``:
``outcome`` is ``holds``, ``fails`` or ``unevaluable`` (a null manifest cell).
A label never stops anything; the footer counts each outcome so a reader need
not scan the rows.

One file per batch, like every sidecar: the last workflow with conditions to
run over the batch wrote it, and every row names that workflow
(``workflow_id``). A workflow without conditions leaves an earlier file as it
was.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from womblex.store.output import _write_rows
from womblex.store.run_stamp import sidecar_footer
from womblex.store.source_provenance import NAMESPACE

logger = logging.getLogger(__name__)

CONDITIONS_SUFFIX = ".conditions.parquet"
OUTCOMES_KEY = f"{NAMESPACE}.condition_outcomes"

HOLDS, FAILS, UNEVALUABLE = "holds", "fails", "unevaluable"

CONDITIONS_SCHEMA = pa.schema([
    ("source_hash", pa.string()),
    ("doc_id", pa.string()),
    ("workflow_id", pa.string()),     # the stored workflow whose conditions these are
    ("condition_index", pa.int32()),  # position in the workflow's `conditions`
    ("column", pa.string()),
    ("op", pa.string()),
    ("value", pa.string()),           # the condition's value, JSON
    ("cell", pa.string()),            # the manifest cell it was held against, JSON; null if null
    ("outcome", pa.string()),
])


def conditions_path_for(base_path: Path) -> Path:
    """Return ``<base>.conditions.parquet`` sibling for a shard base path."""
    return base_path.parent / f"{base_path.stem}{CONDITIONS_SUFFIX}"


def write_conditions(rows: list[dict], base_path: Path, workflow_id: str) -> Path:
    """Write *workflow_id*'s labels for a batch beside *base_path*, counting each outcome in the footer."""
    rows = [{**r, "workflow_id": workflow_id} for r in rows]
    target = conditions_path_for(base_path)
    counts = Counter(r["outcome"] for r in rows)
    meta = {
        **(sidecar_footer(base_path, "conditions") or {}),
        OUTCOMES_KEY.encode(): json.dumps({o: counts[o] for o in (HOLDS, FAILS, UNEVALUABLE)}).encode(),
    }
    _write_rows(rows, target, CONDITIONS_SCHEMA, metadata=meta)
    logger.info("Wrote conditions shard %s: rows=%d", target.name, len(rows))
    return target


def read_conditions(path: Path) -> pa.Table:
    """Read condition labels from one shard file or every one in a directory."""
    p = Path(path)
    if p.is_dir():
        shards = sorted(p.glob(f"*{CONDITIONS_SUFFIX}"))
        if not shards:
            return CONDITIONS_SCHEMA.empty_table()
        return pa.concat_tables([pq.read_table(str(s), schema=CONDITIONS_SCHEMA) for s in shards])
    target = p if p.name.endswith(CONDITIONS_SUFFIX) else conditions_path_for(p)
    return pq.read_table(str(target), schema=CONDITIONS_SCHEMA)


__all__ = [
    "CONDITIONS_SCHEMA",
    "CONDITIONS_SUFFIX",
    "FAILS",
    "HOLDS",
    "OUTCOMES_KEY",
    "UNEVALUABLE",
    "conditions_path_for",
    "read_conditions",
    "write_conditions",
]
