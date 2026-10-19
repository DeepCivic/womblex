"""A workflow's conditions: checked when saved and loaded, and not acted on.

A condition is `column op value` over a manifest column. Conditions no longer label anything
(Phase R, D3): the manifest holds every column they labelled, so a reader filters it instead.
They are still checked, so a stored workflow that names one loads or refuses as before, and
the runner and dispatch warn that they are not acted on. Phase R item R.1c removes them.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from womblex.store.output import MANIFEST_SCHEMA

if TYPE_CHECKING:  # pragma: no cover - typing only
    from womblex.cloud.workflow_store import WorkflowCondition

#: column -> "number" | "text", from the manifest schema itself.
CONDITION_COLUMNS: dict[str, str] = {
    f.name: "number" if pa.types.is_integer(f.type) or pa.types.is_floating(f.type) else "text"
    for f in MANIFEST_SCHEMA
}

_ORDER: dict[str, Callable[[Any, Any], bool]] = {
    "gt": operator.gt, "gte": operator.ge, "lt": operator.lt, "lte": operator.le,
}


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


__all__ = ["CONDITION_COLUMNS", "check_conditions_static"]
