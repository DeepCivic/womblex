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
    assert len(runs) == 1 and runs[0].run_id == "nightly"  # the workflow's one run
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
            workflows._fire("bad", {"workflow_id": "bad", "store": ingest}, "20261012T000000Z")
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
    (firing,) = {(j.id.rsplit(":", 1)[-1]) for j in RunBoard(None).list_jobs(run.run_id) if j.kind == "trigger"}
    assert f"dispatch-workflow <the wf workflow file> --run-id wf --firing {firing}" in caplog.text
    down[0] = False  # the operator dispatches the firing's stages: the run is no longer orphaned
    real(stored, run_id=run.run_id, store_uri=None, firing=firing, **_firing_args(fleet))
    with RunBoard(None) as board:
        assert not board.runs()[0].orphaned


def test_a_replayed_firing_refuses_its_own_cancelled_batches(fleet, saved_preset) -> None:  # noqa: F811
    from tests.test_workflows import batch
    from womblex.cloud.workflow_dispatch import fire_workflow
    from womblex.cloud.workflow_runner import RefusalCode, WorkflowRefusedError
    from womblex.store.remote import RemoteStore

    _, ingest, _ = fleet
    ensure_workflow_schema(None)
    with WorkflowStore(None) as store:
        stored = store.save("wf", _spec(["normalise"], [], store=fleet[1]))
    with RunBoard(None) as board:
        board.enqueue("r1", [batch(ingest)], firing="f1")
    dbos_app.open_client(None).cancel_workflow(dbos_app.batch_id("r1", 1))
    with pytest.raises(WorkflowRefusedError) as e:  # a replayed firing would attach to it and never run
        fire_workflow(stored, run_id="r1", firing="f1", ingest=RemoteStore.from_uri(ingest),
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
    assert run.run_id == "bad" and run.counts == {"failed": 1}
    assert row.kind == "trigger" and "[stage_not_dispatchable]" in (row.error or "")


def test_firings_append_only_new_documents_to_one_run(fleet, saved_preset, monkeypatch, caplog) -> None:  # noqa: F811
    import logging

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

    def fire(*firings: str) -> None:
        workflows.set_context(workflows.WorkerContext(
            _minimal_config(tmp), RemoteStore.from_uri(str(store_dir)), RemoteStore.from_uri(ingest), ingest, None))
        try:
            for f in firings:
                workflows._fire("nightly", {"workflow_id": "nightly", "store": ingest}, f)
        finally:
            workflows.set_context(None)

    monkeypatch.setattr(workflow_dispatch, "dispatch_workflow", recording)
    with caplog.at_level(logging.INFO):
        fire("1", "1", "2")  # a replayed firing reuses its batches; one with nothing new runs nothing
        inbox = Path(ingest) / "inbox"
        (inbox / "more.csv").write_text("name,role\nBob,Clerk\n")
        (inbox / "people.csv").write_text("name,role\nAlice,Secretary\n")  # changed: a new document
        fire("3")
    real_calls = [c for c in calls if not c.get("dry_run")]
    assert [c["firing"] for c in real_calls] == ["1", "1", "3"]
    with RunBoard(None) as board:
        jobs = board.list_jobs("nightly")
        assert board.firing_batches("nightly", "1") == [1] and board.firing_batches("nightly", "3") == [2]
        assert len(board.seen_inputs("nightly")) == 3  # both versions of people.csv, and more.csv
    assert sorted(j.id for j in jobs if j.kind == "batch") == ["nightly:batch:0001", "nightly:batch:0002"]
    assert len([j for j in jobs if j.kind == "downstream"]) == 2
    assert "no new documents under its input prefix; firing 2 runs nothing" in caplog.text
    work(fleet, idle_timeout=4)
    (third,) = dbos_app.open_client(None).list_workflows(workflow_ids=["nightly:fire-3:stage:normalise"])
    assert third.output["units"] == 1  # the third firing's stage ran over its own batch alone
    assert "not acted on" not in caplog.text  # the firing acts on the input store and prefix


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


def test_a_schedule_synced_before_input_stores_is_refused_not_stranded(fleet, saved_preset) -> None:  # noqa: F811
    """Its context names no store: the firing is refused on the board, not left on a queue no worker serves."""
    from datetime import UTC, datetime

    client = dbos_app.open_client(None)
    client.enqueue({"workflow_name": dbos_app.SCHEDULED_WORKFLOW, "queue_name": dbos_app.DOWNSTREAM_QUEUE},
                   datetime.now(UTC), {"workflow_id": "old", "trigger": 0})  # the context schedules held then
    work(fleet, idle_timeout=3)
    with RunBoard(None) as board:
        (run,) = board.runs()
        (row,) = board.list_jobs(run.run_id)
    assert run.counts == {"failed": 1} and row.kind == "trigger"
    assert "[spec_invalid]" in (row.error or "") and "schedule-workflow" in (row.error or "")


def test_a_rewrite_of_the_same_bytes_is_not_queued_again(fleet, saved_preset, caplog) -> None:  # noqa: F811
    """Its fingerprint moves (new mtime) but its bytes hash to the run's recorded source_hash."""
    import logging
    import os

    from tests.test_cloud import _minimal_config
    from womblex.cloud import workflows
    from womblex.store.remote import RemoteStore

    store_dir, ingest, tmp = fleet
    _inbox(fleet)
    ensure_workflow_schema(None)
    with WorkflowStore(None) as store:
        store.save("nightly", _spec(["normalise"], [], "inbox", store=fleet[1]))

    def fire(firing: str) -> None:
        workflows.set_context(workflows.WorkerContext(
            _minimal_config(tmp), RemoteStore.from_uri(str(store_dir)), RemoteStore.from_uri(ingest), ingest, None))
        try:
            workflows._fire("nightly", {"workflow_id": "nightly", "store": ingest}, firing)
        finally:
            workflows.set_context(None)

    people = Path(ingest) / "inbox" / "people.csv"
    fire("1")
    work(fleet, idle_timeout=4)  # extracted: the run's manifest records the file's source_hash
    people.write_bytes(people.read_bytes())  # the same bytes, copied over again
    stat = people.stat()
    os.utime(people, (stat.st_atime + 60, stat.st_mtime + 60))
    with caplog.at_level(logging.INFO):
        fire("2")
    assert "rewritten with the same bytes" in caplog.text
    with RunBoard(None) as board:
        assert board.firing_batches("nightly", "2") == []
    people.write_text("name,role\nAlice,Secretary\n")  # different bytes: a new document
    fire("3")
    with RunBoard(None) as board:
        assert board.firing_batches("nightly", "3") == [2]


def test_a_file_found_unchanged_is_not_hashed_again(fleet, saved_preset, monkeypatch) -> None:  # noqa: F811
    """A touched file with the same bytes is hashed once: the firing records its new fingerprint
    (`unchanged`), so a later firing counts it seen (mistake 3: unattended work redoes nothing)."""
    import os
    from datetime import UTC, datetime

    from womblex.store.remote import RemoteStore

    _, ingest, _ = fleet
    _inbox(fleet)
    ensure_workflow_schema(None)
    with WorkflowStore(None) as store:
        store.save("nightly", _spec(["normalise"], [], "inbox", store=ingest))
    hashed: list[str] = []
    real = RemoteStore.sha256
    monkeypatch.setattr(RemoteStore, "sha256", lambda self, key: hashed.append(key) or real(self, key))
    client = dbos_app.open_client(None)

    def fire(minute: int) -> None:  # a recorded firing, as a schedule starts it
        client.enqueue({"workflow_name": dbos_app.SCHEDULED_WORKFLOW, "queue_name": dbos_app.DOWNSTREAM_QUEUE},
                       datetime(2026, 10, 20, 1, minute, tzinfo=UTC), {"workflow_id": "nightly", "store": ingest})
        work(fleet, idle_timeout=4)

    people = Path(ingest) / "inbox" / "people.csv"
    fire(0)
    people.write_bytes(people.read_bytes())  # the same bytes, copied over again
    stat = people.stat()
    os.utime(people, (stat.st_atime + 60, stat.st_mtime + 60))
    fire(1)
    fire(2)
    assert len(hashed) == 1  # the second firing hashed it; the third found its fingerprint seen
    with RunBoard(None) as board:
        assert board.next_batch_num("nightly") == 2  # one batch: the touched file was never queued again
