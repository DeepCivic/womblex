"""`POST /v1/hooks/{workflow}/{hook}`: signed events act on a stored workflow, once."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("dbos")
from fastapi.testclient import TestClient

from tests.test_workflow_runner import _PRESET
from tests.test_workflow_schedules import _inbox, saved_preset  # noqa: F401
from tests.test_workflows import fleet, work  # noqa: F401
from womblex.api.app import create_api_app
from womblex.cloud.inbound_store import InboundEvents, ensure_inbound_schema
from womblex.cloud.jobs import RunBoard
from womblex.cloud.webhook import SIGNATURE_HEADER, sign
from womblex.cloud.workflow_store import WorkflowStore, ensure_workflow_schema, validate_workflow_spec

KEY = "crm-key"
EVENTS = ["documents_added", "run_requested", "rerun_requested", "delivery_received"]


def _spec(ingest: str, steps: list[str] | None = None):
    return validate_workflow_spec({
        "preset": _PRESET.name, "steps": [{"stage": s} for s in steps or ["normalise"]], "collection": "c1",
        "input_binding": {"store": ingest, "ingest_prefix": "inbox"},
        "inbound": [{"name": "crm", "secret_env": "CRM_KEY", "events": EVENTS}],
    })


@pytest.fixture
def hooks(fleet, saved_preset, monkeypatch):  # noqa: F811
    import os
    import shutil

    store_dir, ingest, tmp = fleet
    shutil.copytree(os.environ["WOMBLEX_UI_PRESETS_DIR"], Path(str(store_dir)) / "presets")
    monkeypatch.setenv("CRM_KEY", KEY)
    monkeypatch.delenv("WOMBLEX_INSECURE_INBOUND", raising=False)
    _inbox(fleet)
    other = tmp / "other"  # an input store no worker serves
    (other / "inbox").mkdir(parents=True)
    (other / "inbox" / "a.csv").write_text("name,role\nA,B\n")
    ensure_workflow_schema(None)
    ensure_inbound_schema(None)
    with WorkflowStore(None) as store:
        store.save("feed", _spec(ingest), owner="alice")
        store.save("lonely", validate_workflow_spec({
            **_spec(str(other)).model_dump(mode="json"), "collection": "c2",
            "inbound": [{"name": "crm", "secret_env": "CRM_KEY_2", "events": EVENTS}]}))
    monkeypatch.setenv("CRM_KEY_2", "k2")
    return TestClient(create_api_app(store_uri=str(store_dir), db_dsn="", registry=None, ingest_uri=ingest))


def send(client: TestClient, event: dict, *, workflow: str = "feed", hook: str = "crm", key: str | None = KEY,
         signature: str | None = None):
    raw = json.dumps(event).encode()
    headers = {"content-type": "application/json"}
    if signature or key:
        headers[SIGNATURE_HEADER] = signature or sign(key or "", raw)
    return client.post(f"/v1/hooks/{workflow}/{hook}", content=raw, headers=headers)


def stored_events(workflow: str = "feed") -> list[str]:
    with InboundEvents(None) as events:
        return [e.event_id for e in events.list(workflow)]


def trigger_rows() -> int:
    with RunBoard(None) as board:
        return len([r for r in board._list(None) if (r.attributes or {}).get("kind") == "trigger"])


def test_a_signed_documents_added_adds_one_file_and_a_resend_adds_nothing(hooks, fleet) -> None:  # noqa: F811
    first = send(hooks, {"event": "documents_added", "event_id": "e1"})
    assert first.status_code == 202 and first.json()["firing"].startswith("c1:trigger:event-")
    again = send(hooks, {"event": "documents_added", "event_id": "e1"})
    assert (again.status_code, again.json()) == (202, first.json())
    work(fleet, idle_timeout=4)
    with RunBoard(None) as board:
        assert board.next_batch_num("c1") == 2  # one document, one batch
        assert [r.run_id for r in board.runs(owner="alice")] == ["c1"]  # under the workflow's owner
    assert trigger_rows() == 1 and stored_events() == ["feed:crm:e1"]


def test_a_bad_signature_a_wrong_hook_and_an_undeclared_event_write_nothing(hooks) -> None:
    body = {"event": "documents_added", "event_id": "e1"}
    assert send(hooks, body, signature="sha256=00").status_code == 401
    assert send(hooks, body, key=None).json()["code"] == "bad_signature"  # unsigned
    assert send(hooks, body, hook="nope").json()["code"] == "unknown_hook"
    assert send(hooks, body, workflow="nope").status_code == 404
    refused = send(hooks, {"event": "run_started", "event_id": "e2"})
    assert (refused.status_code, refused.json()["code"]) == (403, "event_not_declared")
    assert send(hooks, [1, 2]).json()["code"] == "bad_event"
    assert stored_events() == [] and trigger_rows() == 0


def test_a_refusal_with_no_live_worker_is_stored_and_a_resend_returns_it(hooks) -> None:
    first = send(hooks, {"event": "run_requested", "event_id": "e1"}, workflow="lonely", key="k2")
    assert (first.status_code, first.json()["code"]) == (409, "no_worker")
    assert stored_events("lonely") == ["lonely:crm:e1"] and trigger_rows() == 0
    assert send(hooks, {"event": "run_requested", "event_id": "e1"}, workflow="lonely", key="k2").json() == first.json()


def test_an_unsigned_event_is_accepted_only_with_the_insecure_switch(hooks, monkeypatch, caplog) -> None:
    body = {"event": "run_requested", "event_id": "e1"}
    assert send(hooks, body, key=None).status_code == 401
    monkeypatch.setenv("WOMBLEX_INSECURE_INBOUND", "1")
    assert send(hooks, body, key=None).status_code == 202
    assert "without checking its signature" in caplog.text


def test_rerun_requested_dispatches_the_stored_steps_over_the_collection(hooks, fleet) -> None:  # noqa: F811
    assert send(hooks, {"event": "documents_added", "event_id": "e1"}).status_code == 202
    work(fleet, idle_timeout=4)
    with WorkflowStore(None) as store:
        store.save("feed", _spec(fleet[1], ["normalise", "spellfix"]), owner="alice")
    answer = send(hooks, {"event": "rerun_requested", "event_id": "e2"})
    assert answer.status_code == 202 and answer.json()["stages"] == ["normalise", "spellfix"]
    work(fleet, idle_timeout=4)
    assert (Path(str(fleet[0])) / "runs/c1/documents/batch-0001.spellfix_text.parquet").exists()


def test_delivery_received_lands_on_the_named_seq_and_an_unknown_seq_is_refused(hooks, fleet) -> None:  # noqa: F811
    deliveries = Path(str(fleet[0])) / "runs/c1/deliveries"
    deliveries.mkdir(parents=True)
    (deliveries / "archive.1.json").write_text(json.dumps({"destination": "archive", "seq": 1}))
    (deliveries / "archive.json").write_text(json.dumps({"destination": "archive", "seq": 2, "replaces": 1}))
    ack = {"event": "delivery_received", "destination": "archive", "detail": {"rows": 7}}
    assert send(hooks, {**ack, "event_id": "d1", "seq": 1}).status_code == 202
    assert send(hooks, {**ack, "event_id": "d2", "seq": 2}).status_code == 202
    first = json.loads((deliveries / "archive.1.json").read_text())
    latest = json.loads((deliveries / "archive.json").read_text())
    assert [a["event_id"] for a in first["acknowledgements"]] == ["d1"]
    assert [a["event_id"] for a in latest["acknowledgements"]] == ["d2"]
    assert first["acknowledgements"][0]["detail"] == {"rows": 7} and latest["seq"] == 2
    unknown = send(hooks, {**ack, "event_id": "d3", "seq": 9})
    assert (unknown.status_code, unknown.json()["code"]) == (409, "unknown_delivery")
    with RunBoard(None) as board:
        assert board.runs() == []  # the run's status is never touched


def test_the_events_listing_is_owner_scoped_and_shows_each_outcome(hooks, fleet) -> None:  # noqa: F811
    from womblex.api.auth import hash_token, parse_registry

    send(hooks, {"event": "documents_added", "event_id": "e1"})
    send(hooks, {"event": "run_requested", "event_id": "e2"}, workflow="lonely", key="k2")
    registry = parse_registry({"clients": [
        {"client_id": "alice", "token_sha256": hash_token("ta"), "scopes": ["read"]},
        {"client_id": "bob", "token_sha256": hash_token("tb"), "scopes": ["read"]},
        {"client_id": "ops", "token_sha256": hash_token("to"), "scopes": ["admin"]},
    ]})
    client = TestClient(create_api_app(store_uri=str(fleet[0]), db_dsn="", registry=registry, ingest_uri=fleet[1]))

    def listing(token: str, workflow: str):
        return client.get(f"/v1/workflows/{workflow}/inbound-events", headers={"Authorization": f"Bearer {token}"})

    mine = listing("ta", "feed").json()["events"]
    assert [(e["event_id"], e["event"], e["outcome"]["status"], e["outcome"]["http_status"]) for e in mine] == [
        ("e1", "documents_added", "accepted", 202)]
    assert listing("tb", "feed").status_code == 404  # another owner's workflow
    assert listing("ta", "nope").status_code == 404
    refused = listing("to", "lonely").json()["events"]  # admin sees every workflow's
    assert [(e["event_id"], e["outcome"]["code"]) for e in refused] == [("e2", "no_worker")]
    assert client.get("/v1/workflows/feed/inbound-events").status_code == 401
