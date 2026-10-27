"""Collections from the CLI: add-to-collection, remove-workflow, rerun-collection and a stored workflow's owner."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("dbos")

from tests.test_workflow_runner import _PRESET
from tests.test_workflow_schedules import _TRIGGER, _fire_all_at_once, _inbox, _where, saved_preset  # noqa: F401
from tests.test_workflows import fleet, work  # noqa: F401
from womblex.cloud import dbos_app
from womblex.cloud.jobs import RunBoard
from womblex.cloud.workflow_schedules import list_schedules
from womblex.cloud.workflow_store import (
    StoredWorkflow,
    WorkflowStore,
    ensure_workflow_schema,
    validate_workflow_spec,
)


@pytest.fixture(autouse=True)
def _no_closed_dbos_stream() -> None:
    """DBOS binds its log handler to the stderr of the test that first launches it; under `capsys`
    that stream closes with the test, and a later launch then fails writing to it."""
    import logging

    dbos_logger = logging.getLogger("dbos")
    for h in list(dbos_logger.handlers):
        if isinstance(h, logging.StreamHandler) and getattr(h.stream, "closed", False):
            dbos_logger.removeHandler(h)


def _collection_file(fleet, name: str = "feed") -> str:  # noqa: F811
    from tests.test_workflow_runner import write_workflow

    return str(write_workflow(fleet[2], name, ["normalise"], collection="c1",
                              input_binding={"store": fleet[1], "ingest_prefix": "inbox"}))


def test_a_collection_keeps_its_copy_without_triggers_and_remove_workflow_deletes_it(
    fleet, saved_preset, capsys,  # noqa: F811
) -> None:
    import argparse

    from womblex.cli.workflow import cmd_remove_workflow, cmd_schedule_workflow

    path = _collection_file(fleet)
    args = argparse.Namespace(workflow=path, dsn=None, remove=False, status=False, **_where(fleet))
    assert cmd_schedule_workflow(argparse.Namespace(**{**vars(args), "owner": "a/b"})) == 1  # not a client_id
    assert cmd_schedule_workflow(argparse.Namespace(**{**vars(args), "owner": "alice"})) == 0
    with WorkflowStore(None) as store:
        assert store.get("feed").owner == "alice"  # stored on the client's behalf
    assert cmd_schedule_workflow(args) == 0
    with WorkflowStore(None) as store:
        assert store.get("feed").run_id == "c1"  # no triggers, but a collection to add to
        assert store.get("feed").owner is None  # stored again without --owner: admin's
    assert cmd_schedule_workflow(argparse.Namespace(**{**vars(args), "remove": True})) == 0
    with WorkflowStore(None) as store:
        assert store.get("feed") is not None  # --remove only unschedules
    capsys.readouterr()
    assert cmd_remove_workflow(argparse.Namespace(workflow="feed", dsn=None)) == 0
    with WorkflowStore(None) as store:
        assert store.get("feed") is None


def test_add_to_collection_fires_the_stored_copy_into_the_collection_and_leaves_the_run_on_removal(
    fleet, saved_preset, capsys,  # noqa: F811
) -> None:
    import argparse

    from womblex.cli.workflow import cmd_add_to_collection, cmd_remove_workflow, cmd_schedule_workflow

    _inbox(fleet)
    path = _collection_file(fleet)
    assert cmd_schedule_workflow(argparse.Namespace(
        workflow=path, dsn=None, remove=False, status=False, **_where(fleet))) == 0
    capsys.readouterr()
    assert cmd_add_to_collection(argparse.Namespace(workflow=path, dsn=None, **_where(fleet))) == 0
    (row_id,) = capsys.readouterr().out.split()
    assert row_id.startswith("c1:trigger:manual-")
    work(fleet, idle_timeout=4)
    with RunBoard(None) as board:
        assert [r.run_id for r in board.runs()] == ["c1"]
        assert board.next_batch_num("c1") == 2  # one batch extracted
    assert cmd_remove_workflow(argparse.Namespace(workflow="feed", dsn=None)) == 0
    with RunBoard(None) as board:
        assert [r.run_id for r in board.runs()] == ["c1"]  # the run and its shards stay


def test_add_to_collection_queues_nothing_for_a_workflow_that_is_not_stored_or_has_no_documents(
    fleet, saved_preset, caplog,  # noqa: F811
) -> None:
    import argparse

    from womblex.cli.workflow import cmd_add_to_collection, cmd_schedule_workflow

    path = _collection_file(fleet)  # nothing under inbox/
    where = _where(fleet)
    assert cmd_add_to_collection(argparse.Namespace(workflow=path, dsn=None, **where)) == 1
    assert "is not stored" in caplog.text
    assert cmd_schedule_workflow(argparse.Namespace(
        workflow=path, dsn=None, remove=False, status=False, **where)) == 0
    assert cmd_add_to_collection(argparse.Namespace(workflow=path, dsn=None, **where)) == 1
    assert "no_documents" in caplog.text
    with RunBoard(None) as board:
        assert board.runs() == []


def test_a_firing_runs_under_the_workflows_owner_and_refuses_a_run_another_owner_holds(
    fleet, saved_preset,  # noqa: F811
) -> None:
    from tests.test_cloud import _minimal_config
    from womblex.cloud import workflows
    from womblex.cloud.jobs import RunOwnedError
    from womblex.cloud.workflow_runner import RefusalCode, WorkflowRefusedError
    from womblex.store.remote import RemoteStore

    store_dir, ingest, tmp = fleet
    _inbox(fleet)
    ensure_workflow_schema(None)
    spec = validate_workflow_spec({
        "preset": _PRESET.name, "steps": [{"stage": "normalise"}], "triggers": [], "collection": "c1",
        "input_binding": {"store": ingest, "ingest_prefix": "inbox"},
    })
    with WorkflowStore(None) as store:
        store.save("feed", spec, owner="alice")
        store.save("rival", spec, owner="bob")

    def fire(name: str, firing: str) -> None:
        workflows.set_context(workflows.WorkerContext(
            _minimal_config(tmp), RemoteStore.from_uri(str(store_dir)), RemoteStore.from_uri(ingest), ingest, None))
        try:
            workflows._fire("c1", {"workflow_id": name, "run_id": "c1", "store": ingest}, firing)
        finally:
            workflows.set_context(None)

    fire("feed", "1")
    with RunBoard(None) as board:
        assert [r.run_id for r in board.runs(owner="alice")] == ["c1"]
        assert board.runs(owner="bob") == []
    (Path(ingest) / "inbox" / "more.csv").write_text("name,role\nBob,Clerk\n")  # new, so the firing would queue it
    with pytest.raises(WorkflowRefusedError) as refused:  # check_firing refuses before any row
        fire("rival", "2")
    assert refused.value.code is RefusalCode.RUN_OWNED
    with RunBoard(None) as board:
        assert board.firing_batches("c1", "2") == []  # nothing was queued
        with pytest.raises(RunOwnedError):  # and the board refuses the write itself
            board.enqueue("c1", [], owner="bob")


def test_a_refused_firings_own_row_does_not_give_its_owner_the_run(fleet, saved_preset) -> None:  # noqa: F811
    from datetime import UTC, datetime, timedelta

    from womblex.cloud.workflow_schedules import add_to_collection

    _inbox(fleet)
    ensure_workflow_schema(None)
    spec = validate_workflow_spec({
        "preset": _PRESET.name, "steps": [{"stage": "normalise"}], "collection": "c1",
        "input_binding": {"store": fleet[1], "ingest_prefix": "inbox"},
    })
    with WorkflowStore(None) as store:
        alice, bob = store.save("feed", spec, owner="alice"), store.save("rival", spec, owner="bob")
    client = dbos_app.open_client(None)
    add_to_collection(client, alice)
    work(fleet, idle_timeout=4)
    add_to_collection(client, bob, now=datetime.now(UTC) + timedelta(seconds=5))  # a schedule's firing: no CLI check
    work(fleet, idle_timeout=4)
    with RunBoard(None) as board:
        rows = board._list("c1")
        assert any((s.attributes or {}).get("owner") == "bob" for s in rows)  # the refused firing's own row
        assert board.runs(owner="bob") == []  # still not bob's run
        assert [r.run_id for r in board.runs(owner="alice")] == ["c1"]


def _rerun_args(path: str, fleet, **extra) -> object:  # noqa: F811
    import argparse

    return argparse.Namespace(workflow=Path(path), dsn=None, output_prefix=None, max_attempts=3,
                              dry_run=False, **_where(fleet), **extra)


def test_rerun_collection_runs_an_added_step_over_the_whole_collection_and_skips_what_is_unchanged(
    fleet, saved_preset, capsys,  # noqa: F811
) -> None:
    import argparse

    from tests.test_workflow_runner import write_workflow
    from womblex.cli.workflow import cmd_add_to_collection, cmd_rerun_collection, cmd_schedule_workflow

    _inbox(fleet)
    path = _collection_file(fleet)
    assert cmd_schedule_workflow(argparse.Namespace(
        workflow=path, dsn=None, remove=False, status=False, **_where(fleet))) == 0
    assert cmd_add_to_collection(argparse.Namespace(workflow=path, dsn=None, **_where(fleet))) == 0
    work(fleet, idle_timeout=4)
    out = Path(str(fleet[0])) / "runs/c1/documents"
    assert (out / "batch-0001.normalised_text.parquet").exists()
    assert not (out / "batch-0001.spellfix_text.parquet").exists()

    write_workflow(fleet[2], "feed", ["normalise", "spellfix"], collection="c1",
                   input_binding={"store": fleet[1], "ingest_prefix": "inbox"})
    assert cmd_rerun_collection(_rerun_args(path, fleet)) == 0
    work(fleet, idle_timeout=4)
    assert (out / "batch-0001.spellfix_text.parquet").exists()
    with RunBoard(None) as board:  # the changed workflow's stages carry its digest in their ids
        reran = {r.id.rsplit(":", 1)[1]: r.status for r in board.list_jobs("c1") if r.kind == "stage" and ":wf-" in r.id}
    assert reran == {"normalise": "skipped", "spellfix": "done"}  # normalise's published unit still matches


def test_rerun_collection_is_refused_before_any_row_for_a_step_the_fleet_cannot_run(
    fleet, saved_preset, caplog,  # noqa: F811
) -> None:
    from tests.test_workflow_runner import write_workflow
    from womblex.cli.workflow import cmd_rerun_collection

    path = write_workflow(fleet[2], "feed", ["quality"], collection="c1",
                          input_binding={"store": fleet[1], "ingest_prefix": "inbox"})
    assert cmd_rerun_collection(_rerun_args(str(path), fleet)) == 1
    assert "stage_not_dispatchable" in caplog.text
    with RunBoard(None) as board:
        assert board.runs() == []

def test_a_scheduled_firing_of_an_owned_workflow_opens_a_run_its_owner_sees(fleet, saved_preset) -> None:  # noqa: F811
    from womblex.cloud.workflow_schedules import schedule_workflow

    _inbox(fleet)
    ensure_workflow_schema(None)
    client = dbos_app.open_client(None)
    spec = validate_workflow_spec({
        "preset": _PRESET.name, "steps": [{"stage": "normalise"}], "triggers": [_TRIGGER], "collection": "c1",
        "input_binding": {"store": fleet[1], "ingest_prefix": "inbox"},
    })
    with WorkflowStore(None) as store:
        schedule_workflow(client, store, StoredWorkflow("feed", spec, ""), owner="alice")
        stored = store.get("feed")
    assert stored is not None and stored.owner == "alice"
    (sched,) = list_schedules(client, "feed")
    assert sched["context"]["owner"] == "alice"
    _fire_all_at_once(client, stored)
    work(fleet, idle_timeout=4)
    with RunBoard(None) as board:
        (run,) = board.runs(owner="alice")
        kinds = sorted((s.attributes or {}).get("kind") for s in board._list(run.run_id))
        assert board.runs(owner="bob") == []
    assert run.run_id == "c1" and run.counts.get("failed") is None
    assert kinds == ["batch", "downstream", "stage", "trigger"]  # every row of the firing is alice's
