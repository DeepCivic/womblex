"""``/api/runs`` — the run selector's data source (docs/ui-plan.md §3).

Read-only: calls nothing but ``womblex.ui.readers``, which calls nothing but
existing ``store/`` readers. No pipeline logic lives here.
"""
from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException

from womblex.store.retention import STAGE_SUFFIXES
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


@router.get("/{run_id}/stage-presence/{stage}")
def get_stage_presence(
    run_id: str, stage: str, settings: UISettings = Depends(get_settings),  # noqa: B008
) -> dict:
    """``source_hash`` values with a `stage` sidecar — the checkpoint switcher's data."""
    if stage not in STAGE_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"unknown stage: {stage}")
    hashes = readers.get_stage_presence(settings, run_id, stage)
    if hashes is None:
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
    return {"run_id": run_id, "stage": stage, "source_hashes": hashes}


@router.get("/{run_id}/audit")
def get_audit(run_id: str, settings: UISettings = Depends(get_settings)) -> dict:  # noqa: B008
    """Shard-directory integrity audit — the Corpus Inspector's verify-shards action."""
    report = readers.get_shard_audit(settings, run_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
    return report
