"""Webhook events: six events, signed, resent under the same id, recorded on the run, never stopping it."""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

import httpx
import pytest

from tests.test_delivery import _to, run_root  # noqa: F401
from tests.test_workflow_runner import _resolve, _stored
from womblex.cloud import delivery, webhook
from womblex.cloud.workflow_runner import run_workflow_local
from womblex.cloud.workflow_store import validate_workflow_spec
from womblex.store.run_manifest import read_run_record

_HOOK = {"kind": "webhook", "name": "hook", "url": "https://consumer.invalid/womblex", "secret_env": "WOMBLEX_TEST_HOOK_KEY"}


class _Receiver:
    """Stands in for `httpx.post`: answers each call with the next status, keeps what it was sent."""

    def __init__(self, *statuses: int) -> None:
        self.statuses, self.calls = list(statuses), []

    def __call__(self, url: str, *, content: bytes, headers: dict, timeout: float) -> httpx.Response:
        self.calls.append((json.loads(content), dict(headers), content))
        return httpx.Response(self.statuses.pop(0) if self.statuses else 200)


@pytest.fixture
def receiver(monkeypatch: pytest.MonkeyPatch) -> _Receiver:
    got = _Receiver()
    monkeypatch.setenv(_HOOK["secret_env"], "s3cret")
    monkeypatch.setattr(httpx, "post", got)
    monkeypatch.setattr(webhook.time, "sleep", lambda _s: None)
    return got


@pytest.mark.parametrize("url", ["ftp://consumer/x", "https://user:pw@consumer/x", "consumer/x"])
def test_a_webhook_url_is_http_and_holds_no_secret(url: str):
    with pytest.raises(ValueError, match="webhook URL"):
        validate_workflow_spec({"preset": "p", "steps": [{"stage": "chunk"}], "destinations": [{**_HOOK, "url": url}]})


def test_its_signing_key_is_checked_before_launch(monkeypatch):
    monkeypatch.delenv(_HOOK["secret_env"], raising=False)
    (d,) = validate_workflow_spec({"preset": "p", "steps": [{"stage": "chunk"}], "destinations": [_HOOK]}).destinations
    assert delivery.check_destinations((d,)) == [("hook", "signing key variable WOMBLEX_TEST_HOOK_KEY not set")]


def test_an_event_is_resent_under_the_same_id_until_it_lands(receiver: _Receiver):
    receiver.statuses = [503, 500, 200]
    (d,) = validate_workflow_spec({"preset": "p", "steps": [{"stage": "chunk"}], "destinations": [_HOOK]}).destinations
    outcome = webhook.send_event(d, "run_finished", "r1:x:run_finished", b'{"a": 1}')
    assert (outcome.status, outcome.attempts) == ("sent", 3)
    assert {h["X-Womblex-Event-Id"] for _, h, _ in receiver.calls} == {"r1:x:run_finished"}


def test_an_event_that_never_lands_is_recorded_failed(receiver: _Receiver):
    receiver.statuses = [500] * webhook.EVENT_ATTEMPTS
    (d,) = validate_workflow_spec({"preset": "p", "steps": [{"stage": "chunk"}], "destinations": [_HOOK]}).destinations
    outcome = webhook.send_event(d, "stage_failed", "id", b"{}")
    assert (outcome.status, outcome.attempts, outcome.error) == ("failed", webhook.EVENT_ATTEMPTS, "HTTP 500")


def test_a_local_run_sends_its_events_signed_and_records_them(run_root: Path, tmp_path: Path, receiver):  # noqa: F811
    stored = _stored(["normalise"], destinations=[_to(tmp_path, include_sources=False), _HOOK])

    result = run_workflow_local(stored, run_root / "documents", resolve_preset=_resolve)

    assert result.exit_code == 0
    bodies = [b for b, _, _ in receiver.calls]
    assert [(b["event"], b["status"]) for b in bodies] == [
        ("extraction_settled", "running"), ("stage_finished", "running"), ("delivered", "running"),
        ("run_finished", "done")]
    assert {b["run_id"] for b in bodies} == {"run-7"} and {b["workflow"] for b in bodies} == {"wf"}
    assert (bodies[1]["stage"], bodies[2]["destination"]) == ("normalise", "archive")
    assert len({b["event_id"] for b in bodies}) == 4
    for _, headers, raw in receiver.calls:
        expected = "sha256=" + hmac.new(b"s3cret", raw, hashlib.sha256).hexdigest()
        assert headers[webhook.SIGNATURE_HEADER] == expected
    saved = json.loads((run_root / "deliveries" / "hook.json").read_text())
    assert saved["status"] == "delivered" and [e["event"] for e in saved["events"]][-1] == "run_finished"
    record = read_run_record(run_root / "manifest.parquet")
    assert sorted(d["destination"] for d in record["deliveries"]) == ["archive", "hook"]


def test_a_webhook_that_is_down_never_stops_the_run(run_root: Path, tmp_path: Path, receiver):  # noqa: F811
    receiver.statuses = [500] * 100
    result = run_workflow_local(_stored(["normalise"], destinations=[_HOOK]), run_root / "documents",
                                resolve_preset=_resolve)
    assert [s.status for s in result.steps] == ["ok"]
    (hook,) = result.deliveries
    assert (hook.status, hook.error) == ("failed", "3 of 3 event(s) not sent")
    assert result.exit_code == 1


def test_a_failed_delivery_is_its_own_event_and_the_run_still_reads_done(
    run_root: Path, tmp_path: Path, receiver, monkeypatch,  # noqa: F811
):
    def refused(*_a, **_k):
        raise OSError("destination refused the write")

    monkeypatch.setattr("womblex.store.egress.build_bundle", refused)
    monkeypatch.setattr(delivery.time, "sleep", lambda _s: None)
    result = run_workflow_local(_stored(["normalise"], destinations=[_to(tmp_path), _HOOK]),
                                run_root / "documents", resolve_preset=_resolve)
    assert result.exit_code == 1
    assert [(b["event"], b["status"]) for b, _, _ in receiver.calls][-2:] == [
        ("delivery_failed", "running"), ("run_finished", "done")]


def test_the_fleet_refuses_a_webhook_it_would_not_send_to(tmp_path: Path, receiver):
    from womblex.cloud.workflow_dispatch import dispatch_workflow
    from womblex.cloud.workflow_runner import RefusalCode, WorkflowRefusedError

    with pytest.raises(WorkflowRefusedError, match="webhook") as refused:
        dispatch_workflow(_stored(["normalise"], destinations=[_HOOK]), resolve_preset=_resolve,
                          run_id="r1", store_uri=None, dsn=None, dry_run=True)
    assert refused.value.code is RefusalCode.DESTINATION_UNAVAILABLE
