"""The run events a fleet worker sends a workflow's webhooks, each a recorded DBOS step.

A step's result is recorded, so a recovered workflow neither resends an event nor renames
it: an event's id is the id of the workflow that sends it plus a key naming the event
(`cloud/webhook.py` holds the events, the signing and the retries). `batch_progress` is sent
from the coordinator as extraction batches settle and from `run_stage` as each unit finishes.
"""

from __future__ import annotations

from typing import Any

from dbos import DBOS

#: A workflow's statuses that mean a job has finished well; the others it can settle in are failures.
_SUCCEEDED = "SUCCESS"
_UNSETTLED = ("ENQUEUED", "DELAYED", "PENDING")


def has_webhook(workflow: dict[str, Any] | None) -> bool:
    """Whether the dispatched *workflow* names a webhook destination."""
    destinations = (workflow or {}).get("spec", {}).get("destinations") or []
    return any(d.get("kind") == "webhook" for d in destinations)


def send(run_id: str, workflow: dict[str, Any], event: str, event_id: str, status: str,
         fields: dict[str, Any]) -> list[dict[str, Any]]:
    """Send one run event to the workflow's webhooks (`cloud/webhook.py`); never raises."""
    from womblex.cloud.delivery import dispatched_workflow
    from womblex.cloud.webhook import emit

    return emit(dispatched_workflow(workflow), run_id, event, event_id, status, **fields)


def record(run_id: str, workflow: dict[str, Any] | None, event: str, key: str, status: str = "running",
           **fields: Any) -> list[dict[str, Any]]:
    """Send *event* as a recorded step of the running workflow, under the id `<workflow id>:<key>`.

    Nothing is sent, and no step recorded, for a workflow without a webhook.
    """
    if workflow is None or not has_webhook(workflow):
        return []
    return DBOS.run_step({"name": f"event:{key}"}, send, run_id, workflow, event,
                         f"{DBOS.workflow_id}:{key}", status, fields)


def batch_counts(run_id: str, bases: list[str] | None = None) -> dict[str, int]:
    """Where the run's extraction batches stand: `done`, `failed`, `total`. *bases* (a firing's
    `batch-NNNN` names) narrow it to a firing's own batches."""
    wanted = {f"{run_id}:batch:{int(b.rsplit('-', 1)[1]):04d}" for b in bases} if bases else None
    rows = [r for r in DBOS.list_workflows(workflow_id_prefix=f"{run_id}:batch:", load_input=False, load_output=False)
            if wanted is None or r.workflow_id in wanted]
    done = sum(r.status == _SUCCEEDED for r in rows)
    failed = sum(r.status != _SUCCEEDED and r.status not in _UNSETTLED for r in rows)
    return {"done": done, "failed": failed, "total": len(rows)}


def extraction_progress(run_id: str, workflow: dict[str, Any] | None, bases: list[str] | None,
                        seen: int) -> tuple[int, list[dict[str, Any]]]:
    """Send `batch_progress` for extraction if more batches have settled than *seen*; return the
    settled count now and what was sent."""
    if workflow is None or not has_webhook(workflow):
        return seen, []
    counts = DBOS.run_step({"name": "extraction-counts"}, batch_counts, run_id, bases)
    settled = counts["done"] + counts["failed"]
    if settled <= seen:
        return seen, []
    return settled, record(run_id, workflow, "batch_progress", f"batch_progress:extract:{settled}",
                           stage="extract", done=counts["done"], failed=counts["failed"], total=counts["total"])
