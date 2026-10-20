"""A workflow's webhook destination: run events, signed, sent with bounded retries (Requirement 27).

Six events and no others (no per-document or per-batch events): `extraction_settled`,
`stage_finished`, `stage_failed`, `run_finished`, `delivered` and `delivery_failed`. Each body
carries an `event_id`, kept when the event is resent so a receiver can drop a duplicate, the
`run_id` and the run's status in the API's words (`pending` / `running` / `done` / `failed` /
`cancelled`). The body is signed with HMAC-SHA256 under the key in the environment variable the
destination names (`X-Womblex-Signature: sha256=<hex>`), held where the run executes.

A send that still fails after its attempts is recorded on the run, in the webhook's delivery
record (`events`), and never stops it. Polling the API still works whatever a webhook does.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from womblex.cloud.delivery import DeliveryOutcome
    from womblex.cloud.workflow_store import StoredWorkflow, WebhookDestination

logger = logging.getLogger(__name__)

EVENTS = ("extraction_settled", "stage_finished", "stage_failed", "run_finished", "delivered", "delivery_failed")
SIGNATURE_HEADER = "X-Womblex-Signature"
EVENT_ATTEMPTS = 3
EVENT_BACKOFF_SECONDS = 2.0
EVENT_TIMEOUT_SECONDS = 10.0


def sign(secret: str, body: bytes) -> str:
    """The signature header's value for *body*: what a receiver recomputes to trust it."""
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def event_body(event: str, event_id: str, run_id: str, status: str, workflow: str, **fields: Any) -> bytes:
    if event not in EVENTS:
        raise ValueError(f"unknown event {event!r}")
    return json.dumps({"event": event, "event_id": event_id, "run_id": run_id, "status": status,
                       "workflow": workflow, "at": datetime.now(UTC).isoformat(timespec="seconds"), **fields},
                      sort_keys=True).encode()


@dataclass
class EventOutcome:
    event: str
    event_id: str
    status: str  # "sent" | "failed"
    attempts: int
    error: str | None = None


def send_event(
    d: WebhookDestination, event: str, event_id: str, body: bytes, *,
    attempts: int = EVENT_ATTEMPTS, backoff: float = EVENT_BACKOFF_SECONDS,
) -> EventOutcome:
    """POST *body* to *d*, signed, up to *attempts* times; a 2xx answer is sent. Never raises."""
    import httpx

    headers = {"Content-Type": "application/json", "X-Womblex-Event": event, "X-Womblex-Event-Id": event_id}
    outcome = EventOutcome(event, event_id, "failed", 0)
    for attempt in range(1, attempts + 1):
        outcome.attempts = attempt
        try:
            headers[SIGNATURE_HEADER] = sign(os.environ[d.secret_env], body)
            answer = httpx.post(d.url, content=body, headers=headers, timeout=EVENT_TIMEOUT_SECONDS)
            if answer.is_success:
                outcome.status, outcome.error = "sent", None
                return outcome
            outcome.error = f"HTTP {answer.status_code}"
        except Exception as e:  # recorded, never raised: an event does not stop the run
            outcome.error = f"{type(e).__name__}: {e}"
        logger.warning("webhook %s: %s attempt %d of %d failed: %s", d.name, event, attempt, attempts, outcome.error)
        if attempt < attempts:
            time.sleep(backoff * 2 ** (attempt - 1))
    return outcome


def emit(
    stored: StoredWorkflow, run_id: str, event: str, event_id: str, status: str, **fields: Any,
) -> list[dict[str, Any]]:
    """Send *event* to each of *stored*'s webhooks; one outcome per webhook, as plain dicts."""
    from womblex.cloud.workflow_store import WebhookDestination

    sent: list[dict[str, Any]] = []
    for d in stored.spec.destinations:
        if isinstance(d, WebhookDestination):
            body = event_body(event, event_id, run_id, status, stored.id, **fields)
            sent.append({"destination": d.name, **asdict(send_event(d, event, event_id, body))})
    return sent


def webhook_records(stored: StoredWorkflow, events: list[dict[str, Any]]) -> list[DeliveryOutcome]:
    """Each webhook's delivery record: the events it was sent, `delivered` only if every one was."""
    from womblex.cloud.delivery import DeliveryOutcome
    from womblex.cloud.workflow_store import WebhookDestination

    records = []
    for d in stored.spec.destinations:
        if isinstance(d, WebhookDestination):
            mine = [{k: v for k, v in e.items() if k != "destination"} for e in events if e["destination"] == d.name]
            failed = [e for e in mine if e["status"] != "sent"]
            records.append(DeliveryOutcome(
                d.name, d.kind, d.url, stored.id, "failed" if failed else "delivered", len(mine),
                error=f"{len(failed)} of {len(mine)} event(s) not sent" if failed else None, events=mine,
                delivered_at=datetime.now(UTC).isoformat(timespec="seconds")))
    return records


class RunEvents:
    """One run's events to its workflow's webhooks, ids prefixed *prefix*; a no-op without webhooks."""

    def __init__(self, stored: StoredWorkflow, run_id: str, prefix: str) -> None:
        from womblex.cloud.workflow_store import WebhookDestination

        self.stored, self.run_id, self.prefix = stored, run_id, prefix
        self.active = any(isinstance(d, WebhookDestination) for d in stored.spec.destinations)
        self.sent: list[dict[str, Any]] = []

    def emit(self, event: str, status: str, subject: str = "", **fields: Any) -> None:
        if self.active:
            event_id = f"{self.prefix}:{event}" + (f":{subject}" if subject else "")
            self.sent += emit(self.stored, self.run_id, event, event_id, status, **fields)

    def records(self) -> list[DeliveryOutcome]:
        return webhook_records(self.stored, self.sent)


__all__ = [
    "EVENTS",
    "SIGNATURE_HEADER",
    "EventOutcome",
    "RunEvents",
    "emit",
    "event_body",
    "send_event",
    "sign",
    "webhook_records",
]
