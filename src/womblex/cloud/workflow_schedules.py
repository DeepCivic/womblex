"""Standing triggers: a stored workflow's cron triggers as DBOS schedules.

One DBOS schedule per trigger, named `womblex-wf:<workflow id>:<n>`, all firing
`womblex.run_scheduled` on the firing queue of the workflow's input store, so only a worker on
that store runs it, with a context naming the workflow, the run it appends to and its store. `sync_schedules` makes the board's schedules for a workflow equal its
stored triggers (atomic apply, then delete the stale ones), so removing a trigger or the
workflow's last trigger unschedules it. The firing itself is `workflows.run_scheduled`.

`schedule_workflow` is the one place a workflow is stored (Phase R, D10): it saves the
copy the firings read and syncs its schedules in one action, so the two never disagree.
The copy is kept while the workflow has triggers, a collection or inbound hooks (something that fires it
or that other systems find it by); `add_to_collection` fires it once, on demand, as a schedule would.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import sqlalchemy.exc

from womblex.cloud import dbos_app
from womblex.cloud.jobs import schema_missing
from womblex.cloud.workflow_store import StoredWorkflow, StoredWorkflowSpec, WorkflowStore

if TYPE_CHECKING:  # pragma: no cover - typing only
    from dbos import DBOSClient


def schedule_prefix(workflow_id: str) -> str:
    """Names are `<prefix><n>`; the trailing colon keeps `a` from matching `a-b`'s schedules."""
    return f"womblex-wf:{workflow_id}:"


def firing_context(stored: StoredWorkflow) -> dict[str, Any]:
    """What a firing is told of its workflow: the same for a schedule's and an on-demand one."""
    context = {"workflow_id": stored.id, "run_id": stored.run_id, "store": stored.spec.input_binding.store}
    if stored.owner is not None:  # the firing's own row joins the run it opens, so it carries the run's owner
        context["owner"] = stored.owner
    return context


def trigger_attributes(context: dict[str, Any], run_id: str, firing: str) -> dict[str, Any]:
    """The attributes of a firing's board row, from the context `firing_context` built."""
    attrs: dict[str, Any] = {"run_id": run_id, "kind": dbos_app.KIND_TRIGGER,
                             "workflow_id": context["workflow_id"], "firing": firing}
    if context.get("owner") is not None:
        attrs["owner"] = context["owner"]
    return attrs


def keeps_copy(spec: StoredWorkflowSpec) -> bool:
    """Whether *spec* needs its copy stored: triggers to fire it, a collection to add to or inbound hooks."""
    return bool(spec.triggers or spec.collection or spec.inbound)


def schedule_inputs(stored: StoredWorkflow) -> list[dict[str, Any]]:
    return [
        {
            "schedule_name": f"{schedule_prefix(stored.id)}{i}",
            "workflow_name": dbos_app.SCHEDULED_WORKFLOW,
            "schedule": t.cron,
            "cron_timezone": t.timezone,
            "queue_name": dbos_app.firing_queue(stored.spec.input_binding.store or ""),
            "context": {**firing_context(stored), "trigger": i},
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


def schedule_workflow(
    client: DBOSClient, wf_store: WorkflowStore, stored: StoredWorkflow, *, owner: str | None = None,
) -> list[str]:
    """Store *stored* as the copy its firings read, owned by *owner* (`None`: admin), and make its schedules equal its triggers.

    A workflow with no triggers is unscheduled, and its copy deleted unless it names a collection.
    If the sync fails, the previous copy is put back (or the new one deleted), so a firing never
    reads a copy its schedules do not match; the error propagates.
    """
    previous = wf_store.get(stored.id)
    stored = replace(stored, owner=owner)  # its schedules' context names the owner its firings run under
    if keeps_copy(stored.spec):
        wf_store.save(stored.id, stored.spec, owner=owner)
    try:
        names = sync_schedules(client, stored)
    except Exception:
        if previous is not None:
            wf_store.save(previous.id, previous.spec, owner=previous.owner)
        elif keeps_copy(stored.spec):
            wf_store.delete(stored.id)
        raise
    if not keeps_copy(stored.spec):
        wf_store.delete(stored.id)
    return names


def add_to_collection(client: DBOSClient, stored: StoredWorkflow, *, now: datetime | None = None) -> str:
    """Enqueue one firing of *stored* now, on its input store's firing queue; return its board id.

    The path a schedule takes (`workflows.fire_trigger`, which makes the same checks), labelled
    `manual-<UTC time>` so a repeat within the second attaches rather than firing twice. The
    caller has checked the firing (`workflow_dispatch.check_firing`) before this writes a row.
    """
    firing = f"manual-{(now or datetime.now(UTC)):%Y%m%dT%H%M%SZ}"
    run_id, context = stored.run_id, firing_context(stored)
    workflow_id = dbos_app.trigger_id(run_id, firing)
    client.enqueue(
        {"workflow_name": dbos_app.TRIGGER_WORKFLOW,
         "queue_name": dbos_app.firing_queue(stored.spec.input_binding.store or ""),
         "workflow_id": workflow_id,
         "attributes": trigger_attributes(context, run_id, firing)},
        run_id, context, firing,
    )
    return workflow_id


def unschedule_workflow(client: DBOSClient, wf_store: WorkflowStore, workflow_id: str) -> int:
    """Remove *workflow_id*'s schedules and its stored copy; return how many schedules went.

    A collection's run and its shards are untouched.
    """
    removed = remove_schedules(client, workflow_id)
    wf_store.delete(workflow_id)
    return removed


__all__ = [
    "add_to_collection", "firing_context", "keeps_copy", "list_schedules", "remove_schedules", "schedule_inputs", "schedule_prefix", "schedule_workflow",
    "sync_schedules", "unschedule_workflow",
]
