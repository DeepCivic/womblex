"""Collections as a service sees them: the stored workflows that feed one, and acting on it.

A collection is a long-lived run named for a workflow's `collection` (`StoredWorkflow.run_id`,
Requirement 34). Several stored workflows may feed one, so acting on a collection names the
workflow when it is not the only one. Adding fires the stored copy (`workflow_schedules.add_to_collection`)
and rerunning dispatches its current steps over the run (`workflow_dispatch.dispatch_workflow`),
each after the checks a firing or dispatch makes, so a refusal writes no row.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import sqlalchemy.exc

from womblex.cloud import dbos_app
from womblex.cloud.jobs import schema_missing
from womblex.cloud.workflow_dispatch import WorkflowDispatch, check_firing, dispatch_workflow
from womblex.cloud.workflow_schedules import add_to_collection
from womblex.cloud.workflow_store import StoredWorkflow, WorkflowStore
from womblex.store.remote_manifest import open_store, remote_manifest_table
from womblex.store.run_manifest import RUN_RECORD_KEY

if TYPE_CHECKING:
    from womblex.cloud.deployment import DeploymentSettings
    from womblex.presets import Preset


class AmbiguousWorkflow(Exception):
    """A collection is fed by several stored workflows and the caller named none."""

    def __init__(self, names: list[str]) -> None:
        super().__init__(f"fed by several workflows: {', '.join(names)}")
        self.names = names


def stored_collections(dsn: str | None, owner: str | None) -> dict[str, list[StoredWorkflow]]:
    """Collection name -> the stored workflows that feed it; *owner* narrows to its own (`None`: all).

    A database no `schedule-workflow` has written to holds none.
    """
    try:
        with WorkflowStore(dsn) as wf_store:
            workflows = wf_store.list()
    except sqlalchemy.exc.DBAPIError as e:
        if not schema_missing(e):
            raise
        return {}
    found: dict[str, list[StoredWorkflow]] = {}
    for w in workflows:
        if w.spec.collection and (owner is None or w.owner == owner):
            found.setdefault(w.spec.collection, []).append(w)
    return found


def pick_workflow(candidates: list[StoredWorkflow], name: str | None) -> StoredWorkflow | None:
    """The workflow to act through: the one *name* names, else the only one; `None` if there is none such.

    Raises `AmbiguousWorkflow` when several feed the collection and none is named.
    """
    if name is not None:
        return next((w for w in candidates if w.id == name), None)
    if len(candidates) > 1:
        raise AmbiguousWorkflow(sorted(w.id for w in candidates))
    return candidates[0] if candidates else None


def collection_documents(store_uri: str, name: str) -> tuple[int, int | None, list[dict[str, Any]]]:
    """The collection's document count, how many are the current version, and the workflows its files record.

    The count of current documents is `None` until the run has a consolidated manifest that
    marks versions (`store/document_versions.py`). Raises `StoreUnreachable`.
    """
    store = open_store(store_uri)
    table = remote_manifest_table(store, f"runs/{name}")
    current: int | None = None
    if "is_current" in table.schema.names and table.column("is_current").null_count == 0:
        current = sum(1 for v in table.column("is_current").to_pylist() if v)
    footer = store.read_parquet_footer(f"runs/{name}/manifest.parquet")
    if footer is None:
        return table.num_rows, current, []
    meta, _ = footer
    raw = (meta or {}).get(RUN_RECORD_KEY.encode())
    try:
        record = json.loads(raw.decode()) if raw else {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        record = {}
    written = record.get("workflows", []) if isinstance(record, dict) else []
    return table.num_rows, current, [{"name": w.get("name"), "digest": w.get("digest"), "stages": w.get("stages", [])}
                  for w in written if isinstance(w, dict)]


def add_documents(
    settings: DeploymentSettings, stored: StoredWorkflow, resolve_preset: Callable[[str], Preset | None],
) -> str:
    """Check the firing, then enqueue it now; return the board id of the firing.

    Raises `WorkflowRefusedError` (before any row) for what a scheduled firing would refuse.
    """
    from womblex.store.remote import RemoteStore

    store_uri = settings.store_uri or ""
    check_firing(stored, run_id=stored.run_id, resolve_preset=resolve_preset,
                 store=RemoteStore.from_uri(store_uri), dsn=settings.db_dsn)
    return add_to_collection(dbos_app.open_client(settings.db_dsn), stored)


def rerun_documents(
    settings: DeploymentSettings, stored: StoredWorkflow, resolve_preset: Callable[[str], Preset | None],
) -> WorkflowDispatch:
    """Dispatch *stored*'s current steps over its whole collection (`rerun-collection`, from the stored copy)."""
    return dispatch_workflow(stored, resolve_preset=resolve_preset, run_id=stored.run_id,
                             store_uri=settings.store_uri, dsn=settings.db_dsn)


__all__ = [
    "AmbiguousWorkflow", "add_documents", "collection_documents", "pick_workflow", "rerun_documents",
    "stored_collections",
]
