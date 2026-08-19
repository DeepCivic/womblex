"""``/api/resources`` — connection cards for the Resources Console
(docs/ui-plan.md merge 10).

``GET`` returns the three cards cheaply — configuration and masked
credentials only, no network call. Each card's live reachability check is a
separate ``POST /test/*`` action, matching the plan's "test actions" and
letting a slow or dead connection block only the card the operator clicked,
not the page load.

``PUT /locations`` is the one write here: save, update or clear the operator
override for the ingest/output cards. Guarded like dispatch is — 403 when
``--audit-only``, 409 with no ``--settings-dir`` — since editing where
documents come from and shards land is dispatch-adjacent, not a pure read.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from womblex.ui import dashboard, resources
from womblex.ui.deps import UISettings, get_base_settings, get_settings

router = APIRouter(prefix="/api/resources", tags=["resources"])


@router.get("")
def get_resources(settings: UISettings = Depends(get_settings)) -> dict:  # noqa: B008
    """The store, ingest, queue and Isaacus cards, cheap enough to load with the screen."""
    return resources.get_resources(settings)


@router.post("/test/store")
def post_test_store(settings: UISettings = Depends(get_settings)) -> dict:  # noqa: B008
    """Live reachability: a local dir check, or `RemoteStore.list_dirs("runs")`."""
    return resources.test_store(settings)


@router.post("/test/ingest")
def post_test_ingest(settings: UISettings = Depends(get_settings)) -> dict:  # noqa: B008
    """Live reachability of the configured ingest location, if any."""
    return resources.test_ingest(settings)


@router.post("/test/queue")
def post_test_queue(
    stale_after: float = Query(dashboard.DEFAULT_STALE_AFTER, gt=0),
    window_seconds: float = Query(dashboard.DEFAULT_THROUGHPUT_WINDOW, gt=0),
    job_limit: int = Query(200, gt=0, le=1000),
    settings: UISettings = Depends(get_settings),  # noqa: B008
) -> dict:
    """Connect to the queue and read fleet + queue-depth state in one go."""
    return resources.test_queue(
        settings, stale_after=stale_after, window_seconds=window_seconds, job_limit=job_limit,
    )


class SaveLocationsRequest(BaseModel):
    """The location-edit form — a full replace of the saved override.

    Each location field is either a new location or ``null`` ("reset to the
    flag/env default"). A ``PUT`` replaces the *whole* location override, so a
    caller keeping a field must resubmit it.

    The S3 credential pair is the exception, because the console masks the
    saved secret in every response and the frontend cannot resubmit a secret
    it can no longer read. Both fields omitted (the default) *keeps* whatever
    credential override was saved; passing both sets a new pair;
    ``clear_credentials`` removes it and reverts to the env keys. A half-set
    pair is refused (400).
    """

    ingest_uri: str | None = None
    store_uri: str | None = None
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    clear_credentials: bool = False


@router.put("/locations")
def put_locations(
    body: SaveLocationsRequest,
    base: UISettings = Depends(get_base_settings),  # noqa: B008
) -> dict:
    """Save / update / clear the ingest and output location override.

    403 when the console is audit-only, 409 when no ``--settings-dir`` (or
    ``$WOMBLEX_UI_SETTINGS_DIR``) is configured, 400 on an overlapping
    ingest/output pair or a location ``RemoteStore`` cannot open.
    """
    if base.audit_only:
        raise HTTPException(
            status_code=403,
            detail="This console is audit-only. Restart it without --audit-only to edit locations.",
        )
    if not base.settings_writable:
        raise HTTPException(
            status_code=409,
            detail="This console has no settings dir configured; location edits are "
                   "disabled. Set --settings-dir (or $WOMBLEX_UI_SETTINGS_DIR) to edit them.",
        )
    try:
        return resources.save_locations(
            base,
            ingest_uri=body.ingest_uri,
            store_uri=body.store_uri,
            s3_access_key_id=body.s3_access_key_id,
            s3_secret_access_key=body.s3_secret_access_key,
            clear_credentials=body.clear_credentials,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
