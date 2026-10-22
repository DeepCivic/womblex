"""One document's chunks with what the stages annotated on them: entities, PII spans, money, quality.

Each sidecar is read with the ``source_hash`` predicate pushed into the
parquet reader, in place, from a local run root or a store; staging a
whole-corpus sidecar to show one document would move every document's rows.
Rows are conformed to the current schema (a shard older than a column gets
nulls), a PII span located to the span gains its offset in its chunk's text,
and a money ``value`` travels as a string, since JSON would make it a float.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from womblex.store.enrichment_output import ENRICHMENT_ENTITIES_SUFFIX, ENTITY_SCHEMA
from womblex.store.money_output import MONEY_SPANS_SCHEMA, MONEY_SPANS_SUFFIX, money_spans_from_legacy
from womblex.store.output import CHUNKS_SCHEMA, CHUNKS_SUFFIX
from womblex.store.pii_output import PII_SPANS_SCHEMA, PII_SPANS_SUFFIX
from womblex.store.quality_output import CHUNK_QUALITY_SCHEMA, CHUNK_QUALITY_SUFFIX
from womblex.store.retention import is_safe_run_id
from womblex.store.run_reads import RUNS_PREFIX, hash_column, store_with_run

if TYPE_CHECKING:
    from womblex.cloud.deployment import DeploymentSettings

logger = logging.getLogger(__name__)

#: (response key, sidecar suffix, schema the rows are conformed to).
DETAIL_SIDECARS: tuple[tuple[str, str, pa.Schema], ...] = (
    ("chunks", CHUNKS_SUFFIX, CHUNKS_SCHEMA),
    ("entities", ENRICHMENT_ENTITIES_SUFFIX, ENTITY_SCHEMA),
    ("pii_spans", PII_SPANS_SUFFIX, PII_SPANS_SCHEMA),
    ("money_spans", MONEY_SPANS_SUFFIX, MONEY_SPANS_SCHEMA),
    ("quality", CHUNK_QUALITY_SUFFIX, CHUNK_QUALITY_SCHEMA),
)


def document_detail(settings: DeploymentSettings, run_id: str, source_hash: str) -> dict[str, Any] | None:
    """``{"chunks", "entities", "pii_spans", "money_spans", "quality"}`` for one document.

    ``None`` when the run does not exist or *run_id* is unsafe (`is_safe_run_id`);
    empty lists when a stage has not run or the document is not in the run.
    """
    if not is_safe_run_id(run_id):
        return None
    filesystem: object | None = None
    if settings.is_remote:
        store = store_with_run(settings, run_id)
        if store is None:
            return None
        prefix = f"{RUNS_PREFIX}/{run_id}/documents"
        paths = {s: [f"{store.root}/{k}" for k in store.list_files(prefix, f"*{s}")] for _, s, _ in DETAIL_SIDECARS}
        filesystem = store.fs
    else:
        run_dir = cast(Path, settings.output_root) / run_id
        if not run_dir.is_dir():
            return None
        shard_dir = run_dir / "documents"
        paths = {s: [str(p) for p in sorted(shard_dir.glob(f"*{s}"))] for _, s, _ in DETAIL_SIDECARS}
    detail = {key: _read_filtered(paths[s], schema, source_hash, filesystem) for key, s, schema in DETAIL_SIDECARS}
    # chunk_index is sequenced per document across narrative and table chunks.
    detail["chunks"].sort(key=lambda r: r["chunk_index"])
    # Only narrative money has an offset into chunk text; cell money anchors to its cell.
    detail["money_spans"] = [r for r in detail["money_spans"] if r["locus"] == "narrative"]
    for span in detail["money_spans"]:
        span["value"] = None if span["value"] is None else str(span["value"])
    _chunk_relative_pii(detail)
    return detail


def _conform(table: pa.Table, schema: pa.Schema) -> pa.Table:
    """*table* reindexed to *schema*, a column an older shard predates filled with nulls."""
    for field in schema:
        if field.name not in table.schema.names:
            table = table.append_column(field, pa.nulls(table.num_rows, type=field.type))
    return table.select([f.name for f in schema]).cast(schema)


def _read_filtered(paths: list[str], schema: pa.Schema, value: str, filesystem: object | None) -> list[dict]:
    """Rows for *value* across *paths*; an unreadable sidecar is warned about and skipped."""
    rows: list[dict] = []
    for p in paths:
        try:
            column = hash_column(p, filesystem=filesystem)
            table = pq.read_table(p, filesystem=filesystem, filters=[(column, "=", value)])
            if column != "source_hash":
                table = table.rename_columns(["source_hash" if n == column else n for n in table.schema.names])
            if schema is MONEY_SPANS_SCHEMA:
                table = money_spans_from_legacy(table)
            rows.extend(_conform(table, schema).to_pylist())
        except Exception as e:
            logger.warning("document detail: skipping unreadable sidecar %s: %s", p, e)
    return rows


def _chunk_relative_pii(detail: dict[str, Any]) -> None:
    """``start`` / ``end`` in the span's chunk text, or ``-1`` for a span not located to the span."""
    starts = {c["chunk_index"]: c["start_char"] for c in detail["chunks"]}
    for span in detail["pii_spans"]:
        origin = starts.get(span["chunk_index"])
        cs, ce = span["char_start"], span["char_end"]
        if span["anchor_level"] != "span" or origin is None or cs is None or ce is None:
            span["start"] = span["end"] = -1
        else:
            span["start"], span["end"] = cs - origin, ce - origin
