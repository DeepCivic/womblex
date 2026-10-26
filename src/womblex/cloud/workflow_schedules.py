"""Standing triggers: a stored workflow's cron triggers as DBOS schedules.

One DBOS schedule per trigger, named `womblex-wf:<workflow id>:<n>`, all firing
`womblex.run_scheduled` on the firing queue of the workflow's input store, so only a worker on
that store runs it, with a context naming the workflow, the run it appends to and its store. `sync_schedules` makes the board's schedules for a workflow equal its
stored triggers (atomic apply, then delete the stale ones), so removing a trigger or the
workflow's last trigger unschedules it. The firing itself is `workflows.run_scheduled`.

`schedule_workflow` is the one place a workflow is stored (Phase R, D10): it saves the
copy the firings read and syncs its schedules in one action, so the two never disagree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sqlalchemy.exc

from womblex.cloud import dbos_app
from womblex.cloud.jobs import schema_missing
from womblex.cloud.workflow_store import StoredWorkflow, WorkflowStore

if TYPE_CHECKING:  # pragma: no cover - typing only
    from dbos import DBOSClient


def schedule_prefix(workflow_id: str) -> str:
    """Names are `<prefix><n>`; the trailing colon keeps `a` from matching `a-b`'s schedules."""
    return f"womblex-wf:{workflow_id}:"


def schedule_inputs(stored: StoredWorkflow) -> list[dict[str, Any]]:
    return [
        {
            "schedule_name": f"{schedule_prefix(stored.id)}{i}",
            "workflow_name": dbos_app.SCHEDULED_WORKFLOW,
            "schedule": t.cron,
            "cron_timezone": t.timezone,
            "queue_name": dbos_app.firing_queue(stored.spec.input_binding.store or ""),
            "context": {"workflow_id": stored.id, "run_id": stored.run_id,
                        "store": stored.spec.input_binding.store, "trigger": i},
            "automatic_backfill": False,
        }
        for i, t in enumerate(stored.spec.triggers)
    ]


def list_schedules(client: DBOSClient, workflow_id: str) -> list[dict[str, Any]]:
    """A workflow's schedules; none in a database no writer has migrated (readers do not migrate)."""
    try:
        return [dict(s) for s in client.list_schedules(schedule_name_prefix=schedule_prefix(workflow_id))]
    except sqlalchemy.exc.DBAPIError as e:
        if not schema_missing(e):
            raise
        return []


def sync_schedules(client: DBOSClient, stored: StoredWorkflow) -> list[str]:
    """Make the schedules for *stored* equal its triggers; return the names now scheduled."""
    wanted = schedule_inputs(stored)
    if wanted:
        client.apply_schedules(wanted)  # type: ignore[arg-type]
    keep = {w["schedule_name"] for w in wanted}
    for existing in list_schedules(client, stored.id):
        if existing["schedule_name"] not in keep:
            client.delete_schedule(existing["schedule_name"])
    return sorted(keep)


def remove_schedules(client: DBOSClient, workflow_id: str) -> int:
    """Unschedule every trigger of *workflow_id*; return how many were removed."""
    names = [s["schedule_name"] for s in list_schedules(client, workflow_id)]
    for name in names:
        client.delete_schedule(name)
    return len(names)


def schedule_workflow(client: DBOSClient, wf_store: WorkflowStore, stored: StoredWorkflow) -> list[str]:
    """Store *stored* as the copy its firings read and make its schedules equal its triggers.

    A workflow with no triggers is unscheduled and its copy deleted. If the sync fails, the
    previous copy is put back (or the new one deleted), so a firing never reads a copy its
    schedules do not match; the error propagates.
    """
    previous = wf_store.get(stored.id)
    if stored.spec.triggers:
        wf_store.save(stored.id, stored.spec)
    try:
        names = sync_schedules(client, stored)
    except Exception:
        if previous is not None:
            wf_store.save(previous.id, previous.spec)
        elif stored.spec.triggers:
            wf_store.delete(stored.id)
        raise
    if not stored.spec.triggers:
        wf_store.delete(stored.id)
    return names


def unschedule_workflow(client: DBOSClient, wf_store: WorkflowStore, workflow_id: str) -> int:
    """Remove *workflow_id*'s schedules and its stored copy; return how many schedules went."""
    removed = remove_schedules(client, workflow_id)
    wf_store.delete(workflow_id)
    return removed


__all__ = [
    "list_schedules", "remove_schedules", "schedule_inputs", "schedule_prefix", "schedule_workflow",
    "sync_schedules", "unschedule_workflow",
]
