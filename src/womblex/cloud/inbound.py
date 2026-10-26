"""What an inbound webhook event does (Requirement 35): authenticate it, record it once, act on it.

`handle_event` is the body of `POST /v1/hooks/<workflow>/<hook>`. The hook is authenticated by
an HMAC over the raw body (`webhook.sign`, the outbound recipe), not by a service token. Checks,
in order: the workflow and hook are stored, the signature, the event is one the hook declares,
then the event id is inserted (`inbound_store`): an id seen before returns the stored outcome,
acceptance or refusal, and does nothing else. Only then are the firing's own checks made, before
any queue row, and their outcome is stored. A refusal before the insert writes nothing, so an
unauthenticated or malformed request never fills the table.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import sqlalchemy.exc

from womblex.cloud import collections as coll
from womblex.cloud.inbound_store import InboundEvents
from womblex.cloud.jobs import RunOwnedError, schema_missing
from womblex.cloud.webhook import sign
from womblex.cloud.workflow_runner import WorkflowRefusedError
from womblex.cloud.workflow_store import (
    INSECURE_INBOUND_ENV,
    InboundHook,
    StoredWorkflow,
    WorkflowStore,
    insecure_inbound,
    is_safe_workflow_id,
)
from womblex.store.delivery_record import delivery_key, dump_delivery, earlier_key, seq_of

if TYPE_CHECKING:
    from collections.abc import Callable

    from womblex.cloud.deployment import DeploymentSettings
    from womblex.presets import Preset

logger = logging.getLogger(__name__)

#: Longest event id taken; the id is a primary key.
MAX_EVENT_ID = 200


@dataclass(frozen=True)
class Answer:
    """The HTTP answer: a status and the JSON body (`{status, ...}`, or `{code, message}` for a refusal)."""

    status: int
    body: dict[str, Any]


def refuse(status: int, code: str, message: str) -> Answer:
    return Answer(status, {"status": "refused", "code": code, "message": message})


def _stored(dsn: str | None, workflow_id: str) -> StoredWorkflow | None:
    try:
        with WorkflowStore(dsn) as wf_store:
            return wf_store.get(workflow_id)
    except sqlalchemy.exc.DBAPIError as e:
        if not schema_missing(e):
            raise
        return None


def _authenticated(hook: InboundHook, raw: bytes, signature: str | None) -> bool:
    if insecure_inbound():
        logger.warning("inbound %s: accepted without checking its signature (%s=1)", hook.name, INSECURE_INBOUND_ENV)
        return True
    secret = os.environ.get(hook.secret_env)
    return bool(secret and signature and hmac.compare_digest(sign(secret, raw), signature))


def _event(raw: bytes) -> tuple[dict[str, Any], str, str] | None:
    """The payload, its event name and its id, or `None` for a body that is not a signed event."""
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    event, event_id = payload.get("event"), payload.get("event_id")
    if not (isinstance(event, str) and isinstance(event_id, str) and 0 < len(event_id) <= MAX_EVENT_ID):
        return None
    return payload, event, event_id


def acknowledge_delivery(
    settings: DeploymentSettings, stored: StoredWorkflow, payload: dict[str, Any], event_id: str,
) -> Answer:
    """Append a consumer's acknowledgement to the delivery record it names (`destination`, `seq`).

    The run's status is not touched. A destination or `seq` with no record is refused.
    """
    from womblex.store.remote import RemoteStore

    dest, seq = payload.get("destination"), payload.get("seq")
    if not (isinstance(dest, str) and is_safe_workflow_id(dest) and isinstance(seq, int) and seq >= 1):
        return refuse(400, "bad_event", "delivery_received names a `destination` and an integer `seq`")
    run = f"runs/{stored.run_id}"
    store = RemoteStore.from_uri(settings.store_uri or "")
    latest = f"{run}/{delivery_key(dest)}"
    record: dict[str, Any] | None = None
    key = latest
    if store.exists(latest):
        record = json.loads(store.read_text(latest))
    if record is None or seq_of(record) != seq:
        key = f"{run}/{earlier_key(dest, seq)}"
        record = json.loads(store.read_text(key)) if store.exists(key) else None
    if record is None:
        return refuse(409, "unknown_delivery", f"no delivery record for {dest!r} with seq {seq}")
    record.setdefault("acknowledgements", []).append({
        "event_id": event_id, "at": datetime.now(UTC).isoformat(timespec="seconds"), "detail": payload.get("detail")})
    store.write_stream(io.BytesIO(dump_delivery(record).encode()), key)
    return Answer(202, {"status": "accepted", "destination": dest, "seq": seq})


def _act(
    settings: DeploymentSettings, stored: StoredWorkflow, event: str, payload: dict[str, Any], event_id: str,
    resolve_preset: Callable[[str], Preset | None],
) -> Answer:
    firing = f"event-{hashlib.sha256(event_id.encode()).hexdigest()[:8]}"
    try:
        if event in ("documents_added", "run_requested"):
            row = coll.add_documents(settings, stored, resolve_preset, firing=firing)
            return Answer(202, {"status": "accepted", "firing": row})
        if event == "rerun_requested":
            sent = coll.rerun_documents(settings, stored, resolve_preset)
            return Answer(202, {"status": "accepted", "stages": list(sent.stages), "newly_dispatched": sent.newly_enqueued})
        return acknowledge_delivery(settings, stored, payload, event_id)
    except WorkflowRefusedError as e:
        return refuse(409, e.code.value, str(e))
    except RunOwnedError as e:
        return refuse(409, "run_owned", str(e))


def handle_event(
    settings: DeploymentSettings, workflow_id: str, hook_name: str, raw: bytes, signature: str | None,
    resolve_preset: Callable[[str], Preset | None],
) -> Answer:
    """Authenticate, record and act on one inbound event; see the module docstring for the order."""
    stored = _stored(settings.db_dsn, workflow_id)
    hook = next((h for h in stored.spec.inbound if h.name == hook_name), None) if stored else None
    if stored is None or hook is None:
        return refuse(404, "unknown_hook", f"no hook {hook_name!r} on workflow {workflow_id!r}")
    if not _authenticated(hook, raw, signature):
        return refuse(401, "bad_signature", "the signature does not match the body")
    parsed = _event(raw)
    if parsed is None:
        return refuse(400, "bad_event", "the body is a JSON object with a string `event` and `event_id`")
    payload, event, event_id = parsed
    if event not in hook.events:
        return refuse(403, "event_not_declared", f"hook {hook_name!r} does not accept {event!r}")
    with InboundEvents(settings.db_dsn) as events:
        key = f"{workflow_id}:{hook_name}:{event_id}"  # an id is the sender's, so it is kept to its own hook
        if (seen := events.begin(key, workflow_id, hook_name, event)) is not None:
            if seen.outcome is None:
                return refuse(409, "in_progress", "this event is still being handled; resend it shortly")
            return Answer(seen.outcome["http_status"], seen.outcome["body"])
        try:
            answer = _act(settings, stored, event, payload, event_id, resolve_preset)
        except Exception:  # not a refusal: the sender may resend the same id once the fault is fixed
            events.forget(key)
            raise
        events.settle(key, {"http_status": answer.status, "body": answer.body})
    return answer


__all__ = ["Answer", "acknowledge_delivery", "handle_event", "refuse"]
