"""A workflow's conditions: labels over the run's manifest, never a gate.

A condition is `column op value` over a **manifest column** (`MANIFEST_SCHEMA`: the
document-level extraction profile Womblex persists — method, counts, status — and the
identity columns). Each document is labelled against each condition — `holds`, `fails`, or
`unevaluable` where its manifest cell is null — in a `*.conditions.parquet` beside its
batch, and the counts are logged. That is the whole semantics:

- the local runner labels before its first step, the fleet's coordinator once the run's
  extraction settles and before its first stage;
- a condition never refuses, halts or narrows a run: a document that fails one is labelled
  and the run goes on over every document (one document never decides a run);
- it never selects documents or skips a step, so no condition can disable `pii` or any other
  step (a condition has no step field to name one);
- a condition that could never be evaluated — a column outside the manifest, a value of the
  wrong type, an order operator on text — is refused when the workflow is saved and again
  when it is loaded, before launch, which is the only time anything refuses.

The per-page profile (`PageProfile`) is not persisted, so it is not a condition column.
"""

from __future__ import annotations

import json
import logging
import operator
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from womblex.store.conditions_output import CONDITIONS_SUFFIX, FAILS, HOLDS, UNEVALUABLE, write_conditions
from womblex.store.output import MANIFEST_SCHEMA, read_manifest

if TYPE_CHECKING:  # pragma: no cover - typing only
    from womblex.cloud.workflow_store import WorkflowCondition
    from womblex.store.remote import RemoteStore

logger = logging.getLogger(__name__)

#: column -> "number" | "text", from the manifest schema itself.
CONDITION_COLUMNS: dict[str, str] = {
    f.name: "number" if pa.types.is_integer(f.type) or pa.types.is_floating(f.type) else "text"
    for f in MANIFEST_SCHEMA
}

_ORDER: dict[str, Callable[[Any, Any], bool]] = {
    "gt": operator.gt, "gte": operator.ge, "lt": operator.lt, "lte": operator.le,
}
_MANIFEST_SUFFIX = "._manifest.parquet"


def check_conditions_static(conditions: Sequence[WorkflowCondition]) -> None:
    """Refuse a condition that could never be evaluated, at save time. Raises `ValueError`."""
    for c in conditions:
        kind = CONDITION_COLUMNS.get(c.column)
        if kind is None:
            raise ValueError(
                f"condition column {c.column!r} is not a manifest column "
                f"({', '.join(sorted(CONDITION_COLUMNS))})"
            )
        values = c.value if c.op in ("in", "not_in") else [c.value]
        if c.op in ("in", "not_in") and (not isinstance(c.value, list) or not c.value):
            raise ValueError(f"condition {c.column} {c.op} needs a non-empty list")
        want = (int, float) if kind == "number" else (str,)
        if any(isinstance(v, bool) or not isinstance(v, want) for v in values):
            raise ValueError(f"condition {c.column} {c.op}: value must be {kind}, got {c.value!r}")
        if c.op in _ORDER and kind != "number":
            raise ValueError(f"condition {c.column} {c.op}: order operators need a number column")


def _outcome(c: WorkflowCondition, cell: Any) -> str:
    if cell is None:
        return UNEVALUABLE
    if c.op in ("in", "not_in"):
        held = (cell in c.value) == (c.op == "in")
    elif c.op in ("eq", "ne"):
        held = (cell == c.value) == (c.op == "eq")
    else:
        held = _ORDER[c.op](cell, c.value)
    return HOLDS if held else FAILS


def label_manifest(conditions: Sequence[WorkflowCondition], manifest: pa.Table) -> list[dict]:
    """One `*.conditions.parquet` row per document and condition."""
    hashes = manifest.column("source_hash").to_pylist()
    docs = manifest.column("doc_id").to_pylist()
    rows: list[dict] = []
    for i, c in enumerate(conditions):
        for h, d, cell in zip(hashes, docs, manifest.column(c.column).to_pylist(), strict=True):
            rows.append({
                "source_hash": h, "doc_id": d, "condition_index": i, "column": c.column, "op": c.op,
                "value": json.dumps(c.value), "cell": None if cell is None else json.dumps(cell),
                "outcome": _outcome(c, cell),
            })
    return rows


def log_outcomes(workflow_id: str, conditions: Sequence[WorkflowCondition], rows: list[dict]) -> None:
    """Warn, per condition, how many documents fail it or cannot be evaluated (a few named)."""
    if not rows:
        logger.warning("workflow %s: no manifest rows, so no document was labelled", workflow_id)
        return
    for i, c in enumerate(conditions):
        mine = [r for r in rows if r["condition_index"] == i]
        bad = {o: [r["doc_id"] for r in mine if r["outcome"] == o] for o in (FAILS, UNEVALUABLE)}
        if not any(bad.values()):
            logger.info("workflow %s: %s %s %r holds for all %d document(s)",
                        workflow_id, c.column, c.op, c.value, len(mine))
            continue
        logger.warning(
            "workflow %s: %s %s %r — %d of %d document(s) fail it, %d unevaluable (e.g. %s); "
            "labelled, the run goes on", workflow_id, c.column, c.op, c.value, len(bad[FAILS]),
            len(mine), len(bad[UNEVALUABLE]), ", ".join(str(d) for d in (bad[FAILS] + bad[UNEVALUABLE])[:3]),
        )


def label_shard_dir(workflow_id: str, conditions: Sequence[WorkflowCondition], shard_dir: Path) -> list[dict]:
    """Label every batch's documents, writing `batch-NNNN.conditions.parquet` beside each manifest."""
    rows: list[dict] = []
    for manifest in sorted(shard_dir.glob(f"*{_MANIFEST_SUFFIX}")):
        base = shard_dir / f"{manifest.name.removesuffix(_MANIFEST_SUFFIX)}.parquet"
        batch = label_manifest(conditions, read_manifest(base))
        write_conditions(batch, base, workflow_id)
        rows.extend(batch)
    log_outcomes(workflow_id, conditions, rows)
    return rows


def label_store(
    workflow_id: str, conditions: Sequence[WorkflowCondition], store: RemoteStore, shard_prefix: str,
) -> dict[str, str]:
    """`label_shard_dir` over a run in an object store: stage in its manifests, label, publish the labels.

    Returns each published key with its SHA-256.
    """
    with tempfile.TemporaryDirectory(prefix="womblex-cond-") as tmp:
        scratch = Path(tmp)
        for key in store.list_files(shard_prefix, f"*{_MANIFEST_SUFFIX}"):
            store.download_file(key, scratch / key.rsplit("/", 1)[-1])
        label_shard_dir(workflow_id, conditions, scratch)
        return store.upload_glob(scratch, f"*{CONDITIONS_SUFFIX}", shard_prefix)


__all__ = [
    "CONDITION_COLUMNS",
    "check_conditions_static",
    "label_manifest",
    "label_shard_dir",
    "label_store",
    "log_outcomes",
]
