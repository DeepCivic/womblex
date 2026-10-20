"""Deliver a finished run to the destinations its workflow names (Requirement 27).

Delivery runs after a workflow's last step, failed steps included, and never stops or fails
the run: each destination is tried a bounded number of times, and what happened is written as
the destination's delivery record (`store/delivery_record.py`), which the run record lists. A
delivery after a failed step names the steps it follows, in its record and in what it delivers.
Redelivery replaces what was delivered.

Every destination is checked before launch (`check_destinations`): its secrets resolve and the
dependencies its URI needs are installed. A check that fails refuses the run, like a failed
model check.

An object-store destination receives the run's egress bundle (`store/egress.py`) under the run
id, with or without raw sources as the destination chooses. A Postgres destination receives one
row per embedded chunk with its embedding (`cloud/destination_postgres.py`).
"""

from __future__ import annotations

import io
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from womblex.store.delivery_record import delivery_key, dump_delivery
from womblex.store.remote import RemoteStore

if TYPE_CHECKING:  # pragma: no cover - typing only
    from womblex.cloud.workflow_store import (
        ObjectStoreDestination,
        PostgresDestination,
        StoredWorkflow,
        WorkflowDestination,
    )

logger = logging.getLogger(__name__)

#: How many times one destination is tried before its failure is recorded.
DELIVERY_ATTEMPTS = 3
#: Seconds before the second attempt; each later wait doubles it.
DELIVERY_BACKOFF_SECONDS = 2.0


def check_destinations(destinations: tuple[WorkflowDestination, ...]) -> list[tuple[str, str]]:
    """``(destination, problem)`` for each destination that could not be delivered to; empty when
    every one can. Opens nothing remote and sends nothing."""
    import fsspec

    from womblex.cloud.workflow_store import PostgresDestination, WebhookDestination

    problems: list[tuple[str, str]] = []
    for d in destinations:
        if isinstance(d, WebhookDestination):
            if not os.environ.get(d.secret_env):
                problems.append((d.name, f"signing key variable {d.secret_env} not set"))
            continue
        if isinstance(d, PostgresDestination):
            if not os.environ.get(d.dsn_env):
                problems.append((d.name, f"environment variable {d.dsn_env} not set"))
            try:
                import pgvector.psycopg  # noqa: F401
            except ImportError:
                problems.append((d.name, "pgvector is not installed (install womblex[pgvector])"))
            continue
        if d.credentials is not None:
            unset = [v for v in (d.credentials.key_env, d.credentials.secret_env) if not os.environ.get(v)]
            if unset:
                problems.append((d.name, f"environment variable(s) {', '.join(unset)} not set"))
        protocol = d.uri.split("://", 1)[0] if "://" in d.uri else "file"
        try:
            fsspec.get_filesystem_class(protocol)
        except (ImportError, ValueError) as e:
            problems.append((d.name, f"cannot write to {protocol}:// here: {e}"))
    return problems


@dataclass
class DeliveryOutcome:
    """One destination's delivery, as its record holds it."""

    destination: str
    kind: str
    uri: str  # an object store's URI, or a Postgres destination's table
    workflow: str
    status: str  # "delivered" | "failed"
    attempts: int
    after_failed: list[str] = field(default_factory=list)
    include_sources: bool = True
    error: str | None = None
    documents: int | None = None
    sources_by_status: dict[str, int] = field(default_factory=dict)
    rows: int | None = None  # a Postgres destination's chunk rows, and those left out for want of text
    rows_skipped: int | None = None
    events: list[dict[str, Any]] = field(default_factory=list)  # a webhook's events, each sent or failed
    delivered_at: str = ""


def _open(d: ObjectStoreDestination) -> RemoteStore:
    creds = (os.environ[d.credentials.key_env], os.environ[d.credentials.secret_env]) if d.credentials else None
    return RemoteStore.from_uri(d.uri, credentials=creds)


def deliver(
    stored: StoredWorkflow, run_root: Path | str | RemoteStore, run_id: str, *,
    after_failed: tuple[str, ...] = (), attempts: int = DELIVERY_ATTEMPTS,
    backoff: float = DELIVERY_BACKOFF_SECONDS,
) -> list[DeliveryOutcome]:
    """Deliver the run at *run_root* to each of *stored*'s destinations; never raises.

    *after_failed* names the workflow's failed steps; a delivery made after them says so. A webhook
    is sent the run's events instead (`cloud/webhook.py`), so it is not delivered to here.
    """
    from womblex.cloud.workflow_store import ObjectStoreDestination, WebhookDestination

    outcomes: list[DeliveryOutcome] = []
    for d in stored.spec.destinations:
        if isinstance(d, WebhookDestination):
            continue
        where = d.uri if isinstance(d, ObjectStoreDestination) else d.table
        outcome = DeliveryOutcome(d.name, d.kind, where, stored.id, "failed", 0, list(after_failed),
                                  d.include_sources if isinstance(d, ObjectStoreDestination) else False)
        marker = {"workflow": stored.id, "destination": d.name, "after_failed": list(after_failed)}
        for attempt in range(1, attempts + 1):
            outcome.attempts = attempt
            try:
                _deliver_one(d, run_root, run_id, marker, outcome)
            except Exception as e:  # recorded, never raised: a delivery does not stop the run
                outcome.error = f"{type(e).__name__}: {e}"
                logger.warning("delivery to %s, attempt %d of %d, failed: %s", d.name, attempt, attempts,
                               outcome.error)
                if attempt < attempts:
                    time.sleep(backoff * 2 ** (attempt - 1))
                continue
            outcome.status, outcome.error = "delivered", None
            break
        outcome.delivered_at = datetime.now(UTC).isoformat(timespec="seconds")
        if after_failed:
            logger.warning("delivered to %s after failed step(s) %s", d.name, ", ".join(after_failed))
        logger.info("delivery to %s (%s): %s after %d attempt(s)", d.name, where, outcome.status, outcome.attempts)
        outcomes.append(outcome)
    return outcomes


def _deliver_one(
    d: ObjectStoreDestination | PostgresDestination, run_root: Path | str | RemoteStore, run_id: str,
    marker: dict[str, Any],
    outcome: DeliveryOutcome,
) -> None:
    """One attempt at *d*, its counts written onto *outcome*; raises on failure."""
    from womblex.cloud.workflow_store import PostgresDestination as Postgres
    from womblex.store.egress import build_bundle, open_run

    if isinstance(d, Postgres):
        from womblex.cloud.destination_postgres import deliver_postgres

        written = deliver_postgres(d, open_run(run_root), run_id)
        outcome.rows, outcome.rows_skipped = written.rows, written.skipped
        return
    bundle = build_bundle(run_root, _open(d), run_id=run_id, include_sources=d.include_sources, delivery=marker)
    outcome.documents, outcome.sources_by_status = bundle.documents, bundle.sources_by_status


def record_deliveries(run: RemoteStore, outcomes: list[DeliveryOutcome]) -> None:
    """Write each outcome as its destination's delivery record in the *run*'s directory."""
    for o in outcomes:
        run.write_stream(io.BytesIO(dump_delivery(asdict(o)).encode()), delivery_key(o.destination))


def dispatched_workflow(workflow: dict[str, Any]) -> StoredWorkflow:
    """The workflow a fleet run was dispatched with, from the dict its coordinator carries."""
    from womblex.cloud.workflow_store import StoredWorkflow, StoredWorkflowSpec

    return StoredWorkflow(workflow["id"], StoredWorkflowSpec(**workflow["spec"]), workflow["digest"])



__all__ = [
    "DELIVERY_ATTEMPTS",
    "DeliveryOutcome",
    "check_destinations",
    "deliver",
    "dispatched_workflow",
    "record_deliveries",
]
