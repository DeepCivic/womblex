"""Webhook events: eight events, signed, resent under the same id, recorded on the run, never stopping it."""

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
        ("batch_progress", "running"), ("extraction_settled", "running"), ("stage_started", "running"),
        ("batch_progress", "running"), ("stage_finished", "running"), ("delivered", "running"),
        ("run_finished", "done")]
    assert {b["run_id"] for b in bodies} == {"run-7"} and {b["workflow"] for b in bodies} == {"wf"}
    assert (bodies[4]["stage"], bodies[5]["destination"]) == ("normalise", "archive")
    assert len({b["event_id"] for b in bodies}) == 7
    for _, headers, raw in receiver.calls:
        expected = "sha256=" + hmac.new(b"s3cret", raw, hashlib.sha256).hexdigest()
        assert headers[webhook.SIGNATURE_HEADER] == expected
    saved = json.loads((run_root / "deliveries" / "hook.json").read_text())
    assert saved["status"] == "delivered" and [e["event"] for e in saved["events"]][-1] == "run_finished"
    record = read_run_record(run_root / "manifest.parquet")
    assert sorted(d["destination"] for d in record["deliveries"]) == ["archive", "hook"]


def test_every_event_names_the_workflows_collection_or_none(run_root: Path, tmp_path: Path, receiver):  # noqa: F811
    run_workflow_local(_stored(["normalise"], destinations=[_HOOK], collection="c1"),
                       run_root / "documents", resolve_preset=_resolve)
    assert {b["collection"] for b, _, _ in receiver.calls} == {"c1"}
    receiver.calls.clear()
    run_workflow_local(_stored(["normalise"], destinations=[_HOOK]), run_root / "documents", resolve_preset=_resolve)
    assert {b["collection"] for b, _, _ in receiver.calls} == {None}


def test_a_webhook_that_is_down_never_stops_the_run(run_root: Path, tmp_path: Path, receiver):  # noqa: F811
    receiver.statuses = [500] * 100
    result = run_workflow_local(_stored(["normalise"], destinations=[_HOOK]), run_root / "documents",
                                resolve_preset=_resolve)
    assert [s.status for s in result.steps] == ["ok"]
    (hook,) = result.deliveries
    assert (hook.status, hook.error) == ("failed", "6 of 6 event(s) not sent")
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


def _counts(body: dict) -> tuple[int, int, int]:
    return body["done"], body["failed"], body["total"]


def test_progress_events_count_up_to_the_total(run_root: Path, receiver):  # noqa: F811
    run_workflow_local(_stored(["normalise"], destinations=[_HOOK]), run_root / "documents", resolve_preset=_resolve)

    progress = [b for b, _, _ in receiver.calls if b["event"] == "batch_progress"]
    extraction = [b for b in progress if b["stage"] == "extract"]
    assert [_counts(b)[0] for b in extraction] == list(range(1, extraction[0]["total"] + 1))
    assert all(sum(_counts(b)[:2]) <= b["total"] for b in progress)
    assert _counts(extraction[-1])[0] + _counts(extraction[-1])[1] == extraction[-1]["total"]
    (unit,) = [b for b in progress if b["stage"] == "normalise"]
    assert _counts(unit) == (1, 0, 1)
    events = [b["event"] for b, _, _ in receiver.calls]
    assert events.index("stage_started") < events.index("stage_finished")


def test_a_stage_that_fails_counts_its_unit_as_failed(run_root: Path, receiver, monkeypatch):  # noqa: F811
    monkeypatch.setattr("womblex.cloud.workflow_runner.run_stage_local",
                        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))
    run_workflow_local(_stored(["normalise"], destinations=[_HOOK]), run_root / "documents", resolve_preset=_resolve)

    (unit,) = [b for b, _, _ in receiver.calls if b["event"] == "batch_progress" and b["stage"] == "normalise"]
    assert _counts(unit) == (0, 1, 1)
    assert "stage_failed" in [b["event"] for b, _, _ in receiver.calls]


def test_a_webhook_naming_events_is_sent_only_those(run_root: Path, receiver):  # noqa: F811
    coarse = ["extraction_settled", "stage_finished", "stage_failed", "run_finished"]
    result = run_workflow_local(_stored(["normalise"], destinations=[{**_HOOK, "events": coarse}]),
                                run_root / "documents", resolve_preset=_resolve)

    assert [b["event"] for b, _, _ in receiver.calls] == ["extraction_settled", "stage_finished", "run_finished"]
    (hook,) = result.deliveries
    assert (hook.status, hook.attempts) == ("delivered", 3)  # the record lists what was sent, not what was dropped


@pytest.mark.parametrize(("events", "message"), [
    (["stage_started", "per_document"], "unknown event"),
    ([], "leave it out"),
])
def test_a_webhook_naming_a_bad_event_list_is_refused(events: list[str], message: str):
    with pytest.raises(ValueError, match=message):
        validate_workflow_spec({"preset": "p", "steps": [{"stage": "chunk"}],
                                "destinations": [{**_HOOK, "events": events}]})


def test_naming_events_changes_the_digest_and_leaving_them_out_does_not():
    from womblex.cloud.workflow_store import workflow_digest

    base = {"preset": "p", "steps": [{"stage": "chunk"}]}
    plain = validate_workflow_spec({**base, "destinations": [_HOOK]})
    narrowed = validate_workflow_spec({**base, "destinations": [{**_HOOK, "events": ["run_finished"]}]})
    assert workflow_digest(plain) != workflow_digest(narrowed)
    assert validate_workflow_spec({**base, "destinations": [{**_HOOK, "events": None}]}) == plain
