"""`womblex_inbound_events`: insert first, a conflict means seen, and the stored outcome is what a resend gets."""
from __future__ import annotations

from pathlib import Path

import pytest

from womblex.cloud.inbound_store import InboundEvents, ensure_inbound_schema


@pytest.fixture
def events(tmp_path: Path):
    dsn = f"sqlite:///{tmp_path}/events.db"
    ensure_inbound_schema(dsn)
    ensure_inbound_schema(dsn)  # idempotent
    with InboundEvents(dsn) as e:
        yield e


def test_a_new_event_is_recorded_and_a_resend_gets_the_first_outcome(events) -> None:
    assert events.begin("e1", "wf", "crm", "documents_added") is None
    seen = events.begin("e1", "wf", "crm", "documents_added")
    assert seen is not None and seen.outcome is None  # checks not finished yet
    events.settle("e1", {"status": "refused", "code": "no_worker"})
    again = events.begin("e1", "other", "other", "run_requested")  # same id, whatever else it says
    assert again is not None and again.outcome == {"code": "no_worker", "status": "refused"}
    assert (again.workflow_id, again.hook, again.event) == ("wf", "crm", "documents_added")


def test_events_list_newest_first_per_workflow(events) -> None:
    for i in range(3):
        events.begin(f"e{i}", "wf", "crm", "run_requested")
    events.begin("x", "other", "crm", "run_requested")
    assert [e.event_id for e in events.list("wf")] == ["e2", "e1", "e0"]
    assert [e.event_id for e in events.list("wf", limit=1)] == ["e2"]
    assert events.get("missing") is None


def test_a_failed_handling_can_be_forgotten_but_a_settled_event_cannot(events) -> None:
    events.begin("e1", "wf", "crm", "run_requested")
    events.forget("e1")
    assert events.get("e1") is None and events.begin("e1", "wf", "crm", "run_requested") is None
    events.settle("e1", {"http_status": 202, "body": {}})
    events.forget("e1")
    assert events.get("e1") is not None
