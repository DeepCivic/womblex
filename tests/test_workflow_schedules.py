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


def _spec(steps: list[str], triggers: list[dict] | None = None, prefix: str | None = None):
    return validate_workflow_spec({
        "preset": _PRESET.name, "steps": [{"stage": s} for s in steps],
        "triggers": [_TRIGGER] if triggers is None else triggers,
        "input_binding": {"ingest_prefix": prefix},
    })


def _alices_inbox(fleet) -> None:  # noqa: F811
    """An owned firing extracts only under `<owner>/`, so put the document there."""
    (Path(fleet[1]) / "alice").mkdir()
    (Path(fleet[1]) / "alice" / "people.csv").write_text("name,role\nAlice,Director\n")


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
        stored = store.save("nightly", _spec(["normalise"], [_TRIGGER, {"cron": "0 5 * * *"}]))
        assert len(sync_schedules(client, stored)) == 2
        assert sync_schedules(client, stored) == [s["schedule_name"] for s in list_schedules(client, "nightly")]
        assert {s["cron_timezone"] for s in list_schedules(client, "nightly")} == {"Australia/Sydney", None}
        assert list_schedules(client, "night") == []  # a prefix of the id is not the workflow

        one = store.save("nightly", _spec(["normalise"], [_TRIGGER]))
        assert len(sync_schedules(client, one)) == 1 and len(list_schedules(client, "nightly")) == 1
        assert remove_schedules(client, "nightly") == 1 and list_schedules(client, "nightly") == []
        assert sync_schedules(client, store.save("nightly", _spec(["normalise"], []))) == []


def _where(fleet) -> dict:  # noqa: F811
    """The locations `schedule-workflow` checks a firing against."""
    return {"store": str(fleet[0]), "ingest": fleet[1], "presets_dir": None}


def test_remove_and_status_work_after_the_workflow_is_deleted(
    fleet, saved_preset, capsys, monkeypatch,  # noqa: F811
) -> None:
    import argparse

    from womblex.cli.workflow import cmd_schedule_workflow

    for var in ("WOMBLEX_DB_DSN", "DATABASE_URL"):  # the CLI reads these; CI sets one for Postgres
        monkeypatch.delenv(var, raising=False)
    fresh = fleet[2] / "none" / "dbos.sqlite"
    with monkeypatch.context() as m:  # --status changes nothing, not even by creating a database
        m.setenv("WOMBLEX_DBOS_PATH", str(fresh))
        assert cmd_schedule_workflow(argparse.Namespace(workflow_id="x", dsn=None, remove=False, status=True)) == 0
    assert not fresh.parent.exists()
    ensure_workflow_schema(None)
    with WorkflowStore(None) as store:
        store.save("gone", _spec(["normalise"], [_TRIGGER, {"cron": "0 5 * * *"}]))
    args = argparse.Namespace(workflow_id="gone", dsn=None, remove=False, status=False, **_where(fleet))
    assert cmd_schedule_workflow(args) == 0
    with WorkflowStore(None) as store:
        store.delete("gone")
    assert cmd_schedule_workflow(args) == 1  # syncing needs the stored triggers
    capsys.readouterr()
    assert cmd_schedule_workflow(argparse.Namespace(**{**vars(args), "status": True})) == 0
    assert capsys.readouterr().out.count("womblex-wf:gone:") == 2
    assert cmd_schedule_workflow(argparse.Namespace(**{**vars(args), "remove": True})) == 0
    assert list_schedules(dbos_app.open_client(None), "gone") == []


def test_a_firing_extracts_the_prefix_and_runs_the_workflow_owned_and_stamped(
    fleet, saved_preset,  # noqa: F811
) -> None:
    store_dir, _, _ = fleet
    _alices_inbox(fleet)
    ensure_workflow_schema(None)
    with WorkflowStore(None) as store:
        stored = store.save("nightly", _spec(["normalise"], [_TRIGGER, {"cron": "0 5 * * *"}], "alice"),
                            owner="alice")
    client = dbos_app.open_client(None)
    _fire_all_at_once(client, stored)  # two triggers, one instant: one run
    work(fleet, idle_timeout=4)

    with RunBoard(None) as board:
        runs = board.runs(owner="alice")
        assert board.runs(owner="bob") == []
        rows = board._list(runs[0].run_id)
    assert len(runs) == 1 and runs[0].run_id.startswith("nightly-")
    kinds = sorted((s.attributes or {}).get("kind") for s in rows)
    assert kinds == ["batch", "downstream", "stage", "trigger"]  # one firing ran, the other attached
    assert all((s.attributes or {}).get("owner") == "alice" for s in rows)  # every row the firing made
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
        store.save("bad", _spec(["pii"], []))
    ingest_store = RemoteStore.from_uri(ingest)
    workflows.set_context(workflows.WorkerContext(
        _minimal_config(tmp), RemoteStore.from_uri(str(store_dir)), ingest_store, ingest, None))
    database = tmp / "dbos.sqlite"
    before = database.read_bytes()
    try:
        with pytest.raises(WorkflowRefusedError):
            workflows._fire("bad-20261012T000000Z", {"workflow_id": "bad", "owner": None})
    finally:
        workflows.set_context(None)
    with RunBoard(None) as board:
        assert board.runs() == []
    assert database.read_bytes() == before


def _fire_with_a_failing_dispatch(fleet, monkeypatch) -> None:  # noqa: F811
    """Fire `wf` with the dry run and the batches succeeding and the real dispatch failing."""
    import os

    from womblex.cloud import workflow_dispatch
    from womblex.cloud.workflow_runner import preset_resolver
    from womblex.store.remote import RemoteStore

    store_dir, ingest, _ = fleet
    ensure_workflow_schema(None)
    with WorkflowStore(None) as store:
        stored = store.save("wf", _spec(["normalise"], []))
    real = workflow_dispatch.dispatch_workflow

    def dispatch(stored, **kw):
        if kw.get("dry_run"):
            return real(stored, **kw)
        raise OSError("system database dropped")

    monkeypatch.setattr(workflow_dispatch, "dispatch_workflow", dispatch)
    with pytest.raises(OSError, match="dropped"):
        workflow_dispatch.fire_workflow(
            stored, run_id="wf-1", resolve_preset=preset_resolver(Path(os.environ["WOMBLEX_UI_PRESETS_DIR"])),
            store=RemoteStore.from_uri(str(store_dir)), ingest=RemoteStore.from_uri(ingest),
            ingest_root=ingest, dsn=None)


def test_a_firing_whose_dispatch_fails_cancels_the_batches_it_queued(fleet, saved_preset, monkeypatch) -> None:  # noqa: F811
    _fire_with_a_failing_dispatch(fleet, monkeypatch)
    with RunBoard(None) as board:
        jobs = board.list_jobs("wf-1")
    assert [(j.kind, j.status) for j in jobs] == [("batch", "failed")]  # cancelled: never started
    assert work(fleet, idle_timeout=1) == 0  # and no worker picks it up


def test_a_cancel_that_fails_too_names_the_orphaned_run(fleet, saved_preset, monkeypatch, caplog) -> None:  # noqa: F811
    def broken(self, run_id):
        raise OSError("still down")

    monkeypatch.setattr(RunBoard, "cancel_batches", broken)
    _fire_with_a_failing_dispatch(fleet, monkeypatch)  # the dispatch error, not the cancel's, surfaces
    assert "run wf-1 has extraction and no stages" in caplog.text


def test_a_failed_firing_is_on_the_board_under_its_run(fleet, saved_preset) -> None:  # noqa: F811
    ensure_workflow_schema(None)
    with WorkflowStore(None) as store:
        stored = store.save("bad", _spec(["pii"]), owner="alice")
    client = dbos_app.open_client(None)
    _fire_all_at_once(client, stored)
    work(fleet, idle_timeout=3)
    with RunBoard(None) as board:
        (run, *_) = board.runs(owner="alice")
        (row,) = board.list_jobs(run.run_id, owner="alice")
    assert run.run_id.startswith("bad-") and run.counts == {"failed": 1}
    assert row.kind == "trigger" and "[stage_not_dispatchable]" in (row.error or "")


def test_two_firings_never_conflict_and_each_dry_run_matches_its_dispatch(
    fleet, saved_preset, monkeypatch, caplog,  # noqa: F811
) -> None:
    from tests.test_cloud import _minimal_config
    from womblex.cloud import workflow_dispatch, workflows
    from womblex.store.remote import RemoteStore

    store_dir, ingest, tmp = fleet
    _alices_inbox(fleet)
    ensure_workflow_schema(None)
    with WorkflowStore(None) as store:
        store.save("nightly", _spec(["normalise"], [], "alice"), owner="alice")
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
            workflows._fire(run_id, {"workflow_id": "nightly", "owner": "alice"})
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

    for var in ("WOMBLEX_DB_DSN", "DATABASE_URL"):  # the CLI reads these; CI sets one for Postgres
        monkeypatch.delenv(var, raising=False)
    ensure_workflow_schema(None)
    with WorkflowStore(None) as store:
        store.save("wf", _spec(steps, prefix=prefix))
    (Path(fleet[1]) / "empty").mkdir()
    args = argparse.Namespace(workflow_id="wf", dsn=None, remove=False, status=False, **_where(fleet))
    assert cmd_schedule_workflow(args) == 1
    assert f"[{code}]" in caplog.text
    assert list_schedules(dbos_app.open_client(None), "wf") == []
    assert cmd_schedule_workflow(argparse.Namespace(**{**vars(args), "store": None})) == 1  # nothing to check against


def test_an_owned_workflow_never_extracts_outside_its_owners_tree(
    fleet, saved_preset, monkeypatch,  # noqa: F811
) -> None:
    """The API's rule for a submitted prefix holds for a firing, and nothing is scheduled or queued."""
    import argparse

    from tests.test_cloud import _minimal_config
    from womblex.cli.workflow import cmd_schedule_workflow
    from womblex.cloud import workflows
    from womblex.cloud.workflow_runner import RefusalCode, WorkflowRefusedError
    from womblex.store.remote import RemoteStore

    store_dir, ingest, tmp = fleet
    for var in ("WOMBLEX_DB_DSN", "DATABASE_URL"):  # the CLI reads these; CI sets one for Postgres
        monkeypatch.delenv(var, raising=False)
    ensure_workflow_schema(None)
    with WorkflowStore(None) as store:
        for wid, prefix in (("root", None), ("bobs", "bob")):
            store.save(wid, _spec(["normalise"], [_TRIGGER], prefix), owner="alice")
    for wid in ("root", "bobs"):
        args = argparse.Namespace(workflow_id=wid, dsn=None, remove=False, status=False, **_where(fleet))
        assert cmd_schedule_workflow(args) == 1
        assert list_schedules(dbos_app.open_client(None), wid) == []
    workflows.set_context(workflows.WorkerContext(
        _minimal_config(tmp), RemoteStore.from_uri(str(store_dir)), RemoteStore.from_uri(ingest), ingest, None))
    try:
        with pytest.raises(WorkflowRefusedError) as e:
            workflows._fire("bobs-1", {"workflow_id": "bobs", "owner": "alice"})
    finally:
        workflows.set_context(None)
    assert e.value.code is RefusalCode.PREFIX_NOT_OWNED and "[prefix_not_owned]" in str(e.value)
    with RunBoard(None) as board:
        assert board.runs() == []
