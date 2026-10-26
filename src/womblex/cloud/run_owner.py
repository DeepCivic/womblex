"""Whose a run is, read from the board's rows: the owner its first row names.

`RunBoard._claim_run_owner` refuses a write by that rule, and reads use the same one, so a row
that names some other owner (a refused firing's own `<run>:trigger:<time>` row, written before the
refusal) never makes that owner see or read the run. Self-contained: rows in, rows out.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _attr(status: Any, key: str) -> Any:
    return (status.attributes or {}).get(key)


def run_owners(rows: Iterable[Any]) -> dict[str, str | None]:
    """Each run in *rows* with the owner its earliest row names (`None`: admin-only)."""
    owners: dict[str, str | None] = {}
    for s in sorted(rows, key=lambda s: s.created_at or 0):
        if (run := _attr(s, "run_id")) is not None:
            owners.setdefault(run, _attr(s, "owner"))
    return owners


def owned_rows(rows: list[Any], owners: dict[str, str | None], owner: str) -> list[Any]:
    """The rows of *rows* whose run *owners* says *owner* owns."""
    return [s for s in rows if (run := _attr(s, "run_id")) is not None and owners.get(run) == owner]


__all__ = ["owned_rows", "run_owners"]
