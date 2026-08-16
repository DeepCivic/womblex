"""``/api/runs/{run_id}/feedback`` — the report action both inspectors share.

The console's only write path (docs/ui-plan.md §4, §6). Even this one never
touches a run: it writes to a ``feedback/`` location that is always a
sibling of the run it reports on, never a child of it, so nothing here can
disturb a run's own artefacts. Same governing rule as the read routes —
``womblex.ui.readers`` does the IO, this module stays a thin DI wrapper.
"""
from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from womblex.ui import readers
from womblex.ui.deps import UISettings, get_settings

router = APIRouter(prefix="/api/runs", tags=["feedback"])

# A trusted reverse proxy sets this; there is no auth layer to verify it
# against (docs/ui-plan.md §6), so it is advisory the same way the env
# fallback below is.
_REPORTED_BY_HEADER = "X-Womblex-Reported-By"


class FeedbackReport(BaseModel):
    """One reviewer's report on one record — the ``ReportIssue`` control's payload."""

    record_type: str
    source_hash: str
    chunk_index: int | None = None
    row: dict
    note: str = ""


@router.post("/{run_id}/feedback", status_code=201)
def post_feedback(
    run_id: str,
    body: FeedbackReport,
    request: Request,
    settings: UISettings = Depends(get_settings),  # noqa: B008
) -> dict:
    """Write one feedback file and return it. ``reported_by`` is advisory, not verified."""
    reported_by = request.headers.get(_REPORTED_BY_HEADER) or os.environ.get(
        "WOMBLEX_UI_REPORTED_BY"
    )
    record = readers.write_feedback(
        settings, run_id,
        record_type=body.record_type, source_hash=body.source_hash,
        chunk_index=body.chunk_index, row=body.row, note=body.note,
        reported_by=reported_by,
    )
    if record is None:
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
    return record
