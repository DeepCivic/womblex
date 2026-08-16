"""``/api/runs`` — the run selector's data source (docs/ui-plan.md §3).

Read-only: calls nothing but ``womblex.ui.readers``, which calls nothing but
existing ``store/`` readers. No pipeline logic lives here.
"""
from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException

from womblex.ui import readers
from womblex.ui.deps import UISettings, get_settings

router = APIRouter(prefix="/api/runs", tags=["runs"])


@router.get("")
def list_runs(settings: UISettings = Depends(get_settings)) -> dict:  # noqa: B008 - FastAPI DI idiom
    """All runs visible under the console's configured run source."""
    runs = readers.list_run_summaries(settings)
    return {"runs": [asdict(r) for r in runs]}


@router.get("/{run_id}/manifest")
def get_manifest(run_id: str, settings: UISettings = Depends(get_settings)) -> dict:  # noqa: B008
    """The run's documents table (``MANIFEST_SCHEMA``), one row per document."""
    rows = readers.get_manifest_rows(settings, run_id)
    if rows is None:
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
    return {"run_id": run_id, "documents": rows}
