"""Standing triggers: stored cron triggers as DBOS schedules, fired by a worker."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import pytest

pytest.importorskip("dbos")

from tests.test_workflow_runner import _PRESET
from tests.test_workflows import fleet, work  # noqa: F401
from womblex.cloud import dbos_app
from womblex.cloud.jobs import RunBoard
from womblex.cloud.workflow_schedules import list_schedules, remove_schedules, sync_schedules
from womblex.cloud.workflow_store import (
    StoredWorkflow,
    WorkflowStore,
    ensure_workflow_schema,
    validate_workflow_spec,
)
from womblex.ui.presets import build_preset_record, preset_filename, serialise_preset_record

if TYPE_CHECKING:
    from dbos import DBOSClient

_TRIGGER = {"cron": "0 0 1 1 *", "timezone": "Australia/Sydney"}  # never fires within a test


def _spec(steps: list[str], triggers: list[dict] | None = None, prefix: str | None = None, store: str | None = None):
    return validate_workflow_spec({
        "preset": _PRESET.name, "steps": [{"stage": s} for s in steps],
        "triggers": [_TRIGGER] if triggers is None else triggers,
        "input_binding": {"store": store, "ingest_prefix": prefix},
    })


def _inbox(fleet) -> None:  # noqa: F811
    """A document under `inbox/`, the prefix a firing extracts."""
    (Path(fleet[1]) / "inbox").mkdir()
    (Path(fleet[1]) / "inbox" / "people.csv").write_text("name,role\nAlice,Director\n")


def _fire_all_at_once(client: DBOSClient, stored: StoredWorkflow) -> None:
    """Enqueue one firing per schedule, all at one instant, each with the workflow name, queue
    and context its stored schedule holds (not DBOS's `sched-…` workflow id)."""
    from datetime import UTC, datetime

    sync_schedules(client, stored)
    now = datetime.now(UTC)
    for sched in list_schedules(client, stored.id):
        client.enqueue({"workflow_name": sched["workflow_name"], "queue_name": sched["queue_name"]},
                       now, sched["context"])


@pytest.fixture
def saved_preset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record, _ = build_preset_record(name=_PRESET.name, description="", formats=(), config=_PRESET.config)
    (tmp_path / preset_filename(_PRESET.name)).write_text(serialise_preset_record(record))
    monkeypatch.setenv("WOMBLEX_UI_PRESETS_DIR", str(tmp_path))


def test_sync_makes_schedules_equal_triggers_and_remove_unschedules(fleet) -> None:  # noqa: F811
    ensure_workflow_schema(None)
    with WorkflowStore(None) as store:
        client = dbos_app.open_client(None)
        stored = store.save("nightly", _spec(["normalise"], [_TRIGGER, {"cron": "0 5 * * *"}], store=fleet[1]))
        assert len(sync_schedules(client, stored)) == 2
        assert sync_schedules(client, stored) == [s["schedule_name"] for s in list_schedules(client, "nightly")]
        assert {s["cron_timezone"] for s in list_schedules(client, "nightly")} == {"Australia/Sydney", None}
        assert list_schedules(client, "night") == []  # a prefix of the id is not the workflow

        one = store.save("nightly", _spec(["normalise"], [_TRIGGER], store=fleet[1]))
        assert len(sync_schedules(client, one)) == 1 and len(list_schedules(client, "nightly")) == 1
        assert remove_schedules(client, "nightly") == 1 and list_schedules(client, "nightly") == []
        assert sync_schedules(client, store.save("nightly", _spec(["normalise"], [], store=fleet[1]))) == []


def _where(fleet) -> dict:  # noqa: F811
    """The locations `schedule-workflow` checks a firing against."""
    return {"store": str(fleet[0]), "presets_dir": None}


def _file(fleet, name: str, steps: list[str], triggers: list[dict] | None = None,  # noqa: F811
          prefix: str | None = None) -> str:
    from tests.test_workflow_runner import write_workflow

    return str(write_workflow(fleet[2], name, steps, triggers=[_TRIGGER] if triggers is None else triggers,
                              input_binding={"store": fleet[1], "ingest_prefix": prefix}))


def test_scheduling_stores_the_copy_and_its_schedules_together(fleet, saved_preset, monkeypatch) -> None:  # noqa: F811
    import argparse

    from womblex.cli.workflow import cmd_schedule_workflow
    from womblex.cloud import workflow_schedules

    path = _file(fleet, "nightly", ["normalise"], [_TRIGGER, {"cron": "0 5 * * *"}])
    args = argparse.Namespace(workflow=path, dsn=None, remove=False, status=False, **_where(fleet))
    assert cmd_schedule_workflow(args) == 0
    with WorkflowStore(None) as store:
        first = store.get("nightly")
    assert first is not None and len(first.spec.triggers) == 2
    assert len(list_schedules(dbos_app.open_client(None), "nightly")) == 2

    _file(fleet, "nightly", ["normalise"], [_TRIGGER])  # the operator edits the file
    def boom(*_: object) -> None:
        raise OSError("db gone")

    with monkeypatch.context() as m, pytest.raises(OSError):
        m.setattr(workflow_schedules, "sync_schedules", boom)
        cmd_schedule_workflow(args)
    with WorkflowStore(None) as store:  # the copy the firings read still matches its schedules
        assert store.get("nightly").digest == first.digest
    assert cmd_schedule_workflow(args) == 0
    with WorkflowStore(None) as store:
        assert len(store.get("nightly").spec.triggers) == 1
    assert len(list_schedules(dbos_app.open_client(None), "nightly")) == 1

    _file(fleet, "nightly", ["normalise"], [])  # no triggers: unscheduled, and nothing stored
    assert cmd_schedule_workflow(args) == 0
    with WorkflowStore(None) as store:
        assert store.get("nightly") is None
    assert list_schedules(dbos_app.open_client(None), "nightly") == []


def test_remove_and_status_work_after_the_workflow_file_is_deleted(
    fleet, saved_preset, capsys, monkeypatch,  # noqa: F811
) -> None:
    import argparse

    from womblex.cli.workflow import cmd_schedule_workflow

    fresh = fleet[2] / "none" / "dbos.sqlite"
    with monkeypatch.context() as m:  # --status changes nothing, not even by creating a database
        m.setenv("WOMBLEX_DBOS_PATH", str(fresh))
        assert cmd_schedule_workflow(argparse.Namespace(workflow="x", dsn=None, remove=False, status=True)) == 0
    assert not fresh.parent.exists()
    path = _file(fleet, "gone", ["normalise"], [_TRIGGER, {"cron": "0 5 * * *"}])
    args = argparse.Namespace(workflow=path, dsn=None, remove=False, status=False, **_where(fleet))
    assert cmd_schedule_workflow(args) == 0
    Path(path).unlink()
    assert cmd_schedule_workflow(args) == 1  # syncing needs the file
    capsys.readouterr()
    by_name = argparse.Namespace(**{**vars(args), "workflow": "gone"})
    assert cmd_schedule_workflow(argparse.Namespace(**{**vars(by_name), "status": True})) == 0
    assert capsys.readouterr().out.count("womblex-wf:gone:") == 2
    assert cmd_schedule_workflow(argparse.Namespace(**{**vars(by_name), "remove": True})) == 0
    assert list_schedules(dbos_app.open_client(None), "gone") == []
    with WorkflowStore(None) as store:
        assert store.get("gone") is None


def test_a_firing_extracts_the_prefix_and_runs_the_workflow_unowned_and_stamped(
    fleet, saved_preset,  # noqa: F811
) -> None:
    store_dir, _, _ = fleet
    _inbox(fleet)
    ensure_workflow_schema(None)
    with WorkflowStore(None) as store:
        stored = store.save("nightly", _spec(["normalise"], [_TRIGGER, {"cron": "0 5 * * *"}], "inbox", store=fleet[1]))
    client = dbos_app.open_client(None)
    _fire_all_at_once(client, stored)  # two triggers, one instant: one run
    work(fleet, idle_timeout=4)

    with RunBoard(None) as board:
        runs = board.runs()
        assert board.runs(owner="alice") == []  # a workflow's run is admin-only
        rows = board._list(runs[0].run_id)
    assert len(runs) == 1 and runs[0].run_id.startswith("nightly-")
    kinds = sorted((s.attributes or {}).get("kind") for s in rows)
    assert kinds == ["batch", "downstream", "stage", "trigger"]  # one firing ran, the other attached
    assert all("owner" not in (s.attributes or {}) for s in rows)  # no row the firing made has an owner
    out = Path(str(store_dir)) / f"runs/{runs[0].run_id}/documents"
    meta = pq.read_metadata(str(out / "batch-0001.normalised_text.parquet")).metadata
    assert meta[b"womblex.workflow_digest"] == stored.digest.encode()


def test_a_firing_that_cannot_complete_queues_nothing(fleet, saved_preset) -> None:  # noqa: F811
    """The firing body, outside DBOS: `pii` is never dispatched, so it raises before any row."""
    from tests.test_cloud import _minimal_config
    from womblex.cloud import workflows
    from womblex.cloud.workflow_runner import WorkflowRefusedError
    from womblex.store.remote import RemoteStore

    store_dir, ingest, tmp = fleet
    ensure_workflow_schema(None)
    with WorkflowStore(None) as store:
        store.save("bad", _spec(["pii"], [], store=fleet[1]))
    ingest_store = RemoteStore.from_uri(ingest)
    workflows.set_context(workflows.WorkerContext(
        _minimal_config(tmp), RemoteStore.from_uri(str(store_dir)), ingest_store, ingest, None))
    database = tmp / "dbos.sqlite"
    before = database.read_bytes()
    try:
        with pytest.raises(WorkflowRefusedError):
            workflows._fire("bad-20261012T000000Z", {"workflow_id": "bad", "store": ingest})
    finally:
        workflows.set_context(None)
    with RunBoard(None) as board:
        assert board.runs() == []
    assert database.read_bytes() == before


def _firing_args(fleet) -> dict:  # noqa: F811
    import os

    from womblex.cloud.workflow_runner import preset_resolver
    from womblex.store.remote import RemoteStore

    store_dir = fleet[0]
    return {"resolve_preset": preset_resolver(Path(os.environ["WOMBLEX_UI_PRESETS_DIR"])),
            "store": RemoteStore.from_uri(str(store_dir)), "dsn": None}


def test_a_firing_whose_dispatch_fails_leaves_its_extraction_running_and_the_run_orphaned(
    fleet, saved_preset, monkeypatch, caplog,  # noqa: F811
) -> None:
    from womblex.cloud import workflow_dispatch

    ensure_workflow_schema(None)
    with WorkflowStore(None) as store:
        stored = store.save("wf", _spec(["normalise"], store=fleet[1]))
    real, down = workflow_dispatch.dispatch_workflow, [True]

    def dispatch(stored, **kw):
        if down[0] and not kw.get("dry_run"):
            raise OSError("system database dropped")
        return real(stored, **kw)

    monkeypatch.setattr(workflow_dispatch, "dispatch_workflow", dispatch)
    _fire_all_at_once(dbos_app.open_client(None), stored)
    work(fleet, idle_timeout=3)
    with RunBoard(None) as board:
        (run,) = board.runs()
    assert run.counts == {"done": 1, "failed": 1} and run.orphaned  # the batch ran; the firing failed
    assert f"dispatch-workflow <the wf workflow file> --run-id {run.run_id}" in caplog.text
    down[0] = False  # the operator dispatches the stages: the run is no longer orphaned
    real(stored, run_id=run.run_id, store_uri=None, **_firing_args(fleet))
    with RunBoard(None) as board:
        assert not board.runs()[0].orphaned


def test_a_firing_refuses_a_run_whose_batches_were_cancelled(fleet, saved_preset) -> None:  # noqa: F811
    from tests.test_workflows import batch
    from womblex.cloud.workflow_dispatch import fire_workflow
    from womblex.cloud.workflow_runner import RefusalCode, WorkflowRefusedError
    from womblex.store.remote import RemoteStore

    _, ingest, _ = fleet
    ensure_workflow_schema(None)
    with WorkflowStore(None) as store:
        stored = store.save("wf", _spec(["normalise"], [], store=fleet[1]))
    with RunBoard(None) as board:
        board.enqueue("r1", [batch(ingest)])
    dbos_app.open_client(None).cancel_workflow(dbos_app.batch_id("r1", 1))
    with pytest.raises(WorkflowRefusedError) as e:  # a replayed firing would attach to it and never run
        fire_workflow(stored, run_id="r1", ingest=RemoteStore.from_uri(ingest),
                      **_firing_args(fleet))
    assert e.value.code is RefusalCode.BATCHES_CANCELLED
    with RunBoard(None) as board:
        assert [(j.kind, j.status) for j in board.list_jobs("r1")] == [("batch", "cancelled")]


def test_a_failed_firing_is_on_the_board_under_its_run(fleet, saved_preset) -> None:  # noqa: F811
    ensure_workflow_schema(None)
    with WorkflowStore(None) as store:
        stored = store.save("bad", _spec(["pii"], store=fleet[1]))
    client = dbos_app.open_client(None)
    _fire_all_at_once(client, stored)
    work(fleet, idle_timeout=3)
    with RunBoard(None) as board:
        (run, *_) = board.runs()
        (row,) = board.list_jobs(run.run_id)
    assert run.run_id.startswith("bad-") and run.counts == {"failed": 1}
    assert row.kind == "trigger" and "[stage_not_dispatchable]" in (row.error or "")


def test_two_firings_never_conflict_and_each_dry_run_matches_its_dispatch(
    fleet, saved_preset, monkeypatch, caplog,  # noqa: F811
) -> None:
    from tests.test_cloud import _minimal_config
    from womblex.cloud import workflow_dispatch, workflows
    from womblex.store.remote import RemoteStore

    store_dir, ingest, tmp = fleet
    _inbox(fleet)
    ensure_workflow_schema(None)
    with WorkflowStore(None) as store:
        store.save("nightly", _spec(["normalise"], [], "inbox", store=fleet[1]))
    calls: list[dict] = []
    real = workflow_dispatch.dispatch_workflow

    def recording(stored: StoredWorkflow, **kw: object) -> object:
        calls.append(kw)
        return real(stored, **kw)

    monkeypatch.setattr(workflow_dispatch, "dispatch_workflow", recording)
    workflows.set_context(workflows.WorkerContext(
        _minimal_config(tmp), RemoteStore.from_uri(str(store_dir)), RemoteStore.from_uri(ingest), ingest, None))
    try:
        for run_id in ("nightly-1", "nightly-2", "nightly-2"):  # the repeat: two triggers, one instant
            workflows._fire(run_id, {"workflow_id": "nightly", "store": ingest})
    finally:
        workflows.set_context(None)
    for dry, real_call in zip(calls[::2], calls[1::2], strict=True):
        assert dry.pop("dry_run") is True and "dry_run" not in real_call and dry == real_call
    with RunBoard(None) as board:
        coordinators = [r for r in board.list_jobs() if r.kind == "downstream"]
    assert sorted(r.run_id for r in coordinators) == ["nightly-1", "nightly-2"]
    assert "not acted on" not in caplog.text  # the firing acts on the input prefix


@pytest.mark.parametrize("steps, prefix, code", [
    (["pii"], None, "stage_not_dispatchable"),  # the dry-run dispatch the firing makes
    (["normalise"], "empty", "no_documents"),
])
def test_a_workflow_its_firing_would_refuse_is_not_scheduled(
    fleet, saved_preset, caplog, monkeypatch, steps, prefix, code,  # noqa: F811
) -> None:
    import argparse

    from womblex.cli.workflow import cmd_schedule_workflow

    path = _file(fleet, "wf", steps, prefix=prefix)
    (Path(fleet[1]) / "empty").mkdir()
    args = argparse.Namespace(workflow=path, dsn=None, remove=False, status=False, **_where(fleet))
    assert cmd_schedule_workflow(args) == 1
    assert f"[{code}]" in caplog.text
    assert list_schedules(dbos_app.open_client(None), "wf") == []
    assert cmd_schedule_workflow(argparse.Namespace(**{**vars(args), "store": None})) == 1  # nothing to check against


def test_a_firing_runs_only_on_a_worker_on_the_workflows_input_store(fleet, saved_preset) -> None:  # noqa: F811
    ensure_workflow_schema(None)
    with WorkflowStore(None) as store:
        stored = store.save("elsewhere", _spec(["normalise"], store="s3://another-inbox"))
    client = dbos_app.open_client(None)
    _fire_all_at_once(client, stored)
    work(fleet, idle_timeout=2)  # this worker's ingest root is not the workflow's store
    queued = client.list_workflows(queue_name=dbos_app.firing_queue("s3://another-inbox"), load_input=False)
    assert [w.status for w in queued] == ["ENQUEUED"]
    with RunBoard(None) as board:
        assert board.runs() == []


def test_a_scheduled_workflow_must_name_its_input_store(fleet, saved_preset, caplog) -> None:  # noqa: F811
    import argparse

    from tests.test_workflow_runner import write_workflow
    from womblex.cli.workflow import cmd_schedule_workflow

    path = write_workflow(fleet[2], "nowhere", ["normalise"], triggers=[_TRIGGER])
    args = argparse.Namespace(workflow=str(path), dsn=None, remove=False, status=False, **_where(fleet))
    assert cmd_schedule_workflow(args) == 1
    assert "[spec_invalid]" in caplog.text and "input_binding.store" in caplog.text
    assert list_schedules(dbos_app.open_client(None), "nowhere") == []
