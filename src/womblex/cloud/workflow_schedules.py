"""Standing triggers: a stored workflow's cron triggers as DBOS schedules.

One DBOS schedule per trigger, named `womblex-wf:<workflow id>:<n>`, all firing
`womblex.run_scheduled` on the always-served coordinator queue with a context naming the
workflow and its owner. `sync_schedules` makes the board's schedules for a workflow equal its
stored triggers (atomic apply, then delete the stale ones), so removing a trigger or the
workflow's last trigger unschedules it. The firing itself is `workflows.run_scheduled`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sqlalchemy.exc

from womblex.cloud import dbos_app
from womblex.cloud.jobs import schema_missing
from womblex.cloud.workflow_store import StoredWorkflow

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
            "queue_name": dbos_app.DOWNSTREAM_QUEUE,
            "context": {"workflow_id": stored.id, "owner": stored.owner, "trigger": i},
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


__all__ = ["list_schedules", "remove_schedules", "schedule_inputs", "schedule_prefix", "sync_schedules"]
