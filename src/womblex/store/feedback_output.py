"""One-file-per-report feedback writer (docs/ui-plan.md §4).

Any record the console's inspectors show can carry a report action. Each
report writes exactly one JSON file — never an append, so there is no
read-modify-write and no lost update when two reviewers click at once. The
record embeds the reported row rather than referencing it, so the feedback
file stays meaningful even after the run it reported on is purged.

Self-contained like :mod:`womblex.store.pii_output` — no pyarrow, no
schema, just JSON. ``womblex.ui.readers`` decides *where* the feedback
root sits for local vs. store-backed runs; this module only knows how to
name and write one report once it has that root.
"""
from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FEEDBACK_DIRNAME = "feedback"


def feedback_filename() -> str:
    """A sortable, collision-resistant filename for one report.

    ``uuid4`` suffix (not just the timestamp) is what makes two reports
    filed in the same second by different reviewers land as two files
    instead of one clobbering the other.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}.json"


def build_feedback_record(
    *,
    run_id: str,
    record_type: str,
    source_hash: str,
    chunk_index: int | None,
    row: dict[str, Any],
    note: str,
    reported_by: str | None,
) -> dict[str, Any]:
    """Assemble the feedback record's on-disk shape (docs/ui-plan.md §4)."""
    return {
        "reported_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reported_by": reported_by,
        "run_id": run_id,
        "record_type": record_type,
        "source_hash": source_hash,
        "chunk_index": chunk_index,
        "row": row,
        "note": note,
    }


def write_feedback_record(feedback_root: Path, run_id: str, record: dict[str, Any]) -> Path:
    """Write *record* to its own file under ``<feedback_root>/<run_id>/``.

    ``feedback_root`` is never a run directory itself (docs/ui-plan.md §4)
    — callers resolve it as a sibling location, not a child of any run, so
    retention purges and re-runs cannot disturb accumulated feedback.
    """
    target_dir = feedback_root / run_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / feedback_filename()
    target.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return target


__all__ = [
    "FEEDBACK_DIRNAME",
    "build_feedback_record",
    "feedback_filename",
    "write_feedback_record",
]
