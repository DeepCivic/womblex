"""Dispatching a stored workflow to the fleet: refused before any row, then run under its recipe."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

pytest.importorskip("dbos")

from tests.test_workflow_runner import _resolve, _stored
from tests.test_workflows import batch, fleet, work  # noqa: F401
from womblex.cloud.jobs import RunBoard
from womblex.cloud.workflow_dispatch import dispatch_workflow
from womblex.cloud.workflow_runner import RefusalCode, WorkflowRefusedError
from womblex.cloud.workflow_store import StoredWorkflow, StoredWorkflowSpec, workflow_digest

_NORMALISED = {"processing": {"text_source": "normalised"}}


def _dispatch(fleet, steps, run_id="r1", **kw):  # noqa: F811
    store, _, _ = fleet
    stored = kw.pop("stored", None) or _stored(steps, config=kw.pop("config", {}))
    return dispatch_workflow(
        stored, resolve_preset=kw.pop("resolve", _resolve), run_id=run_id,
        store_uri=str(store), dsn=None, **kw,
    )


def _duplicated_steps() -> StoredWorkflow:
    """A row that would no longer pass `check_spec` (built past the validator)."""
    good = _stored(["normalise"]).spec
    spec = StoredWorkflowSpec.model_construct(**{**dict(good), "steps": good.steps * 2})
    return StoredWorkflow("wf", spec, workflow_digest(spec), "", "")


@pytest.mark.parametrize(("steps", "kw", "code"), [
    (["quality"], {}, RefusalCode.STAGE_NOT_DISPATCHABLE),
    (["layout"], {}, RefusalCode.STAGE_NOT_DISPATCHABLE),
    (["chunk", "normalise"], {"config": _NORMALISED}, RefusalCode.ORDER_NOT_PIPELINE),
    (["chunk"], {"config": _NORMALISED}, RefusalCode.STEP_ORDER),
    (["normalise"], {"resolve": lambda _n: None}, RefusalCode.PRESET_UNKNOWN),
    (["spellfix"], {"config": {"spellfix": {"enabled": True, "dict_name": "no-such-dict"}}},
     RefusalCode.CONFIG_INVALID),
    (["normalise"], {"run_id": "../r1"}, RefusalCode.CONFIG_INVALID),
    ([], {"stored": _duplicated_steps()}, RefusalCode.SPEC_INVALID),
])
def test_refusals_come_before_any_row(fleet, steps, kw, code):  # noqa: F811
    database = fleet[2] / "dbos.sqlite"
    before = database.read_bytes()
    with pytest.raises(WorkflowRefusedError) as e:
        _dispatch(fleet, steps, **kw)
    assert e.value.code is code
    with RunBoard(None) as board:
        assert board.list_jobs("r1") == board.list_jobs("../r1") == []
    assert database.read_bytes() == before


def test_dry_run_checks_and_queues_nothing(fleet):  # noqa: F811
    database = fleet[2] / "dbos.sqlite"
    before = database.read_bytes()
    result = _dispatch(fleet, ["normalise", "chunk"], config=_NORMALISED, dry_run=True)
    assert result.stages == ("normalise", "chunk") and result.newly_enqueued == 0
    assert database.read_bytes() == before
    with RunBoard(None) as board:
        assert board.list_jobs("r1") == []


def test_a_dry_run_creates_no_database_and_is_refused_with_no_worker_recorded(fleet, monkeypatch):  # noqa: F811
    fresh = fleet[2] / "none" / "dbos.sqlite"
    monkeypatch.setenv("WOMBLEX_DBOS_PATH", str(fresh))
    with pytest.raises(WorkflowRefusedError) as e:  # no worker has ever recorded itself there
        _dispatch(fleet, ["normalise"], config=_NORMALISED, dry_run=True)
    assert e.value.code is RefusalCode.NO_WORKER and "normalise" in str(e.value)
    assert not fresh.parent.exists()


def test_a_workflow_no_live_worker_serves_is_refused_before_any_row(fleet):  # noqa: F811
    from womblex.cloud.worker_registry import WorkerRegistry

    with WorkerRegistry(None) as registry:
        registry.remove("standby")
    with pytest.raises(WorkflowRefusedError) as e:
        _dispatch(fleet, ["normalise"], config=_NORMALISED)
    assert e.value.code is RefusalCode.NO_WORKER
    with RunBoard(None) as board:
        assert board.list_jobs("r1") == []


def test_a_workflow_runs_on_the_fleet_and_a_different_one_runs_its_own_stages(fleet):  # noqa: F811
    store, ingest, _ = fleet
    config = {"processing": {"text_source": "normalised"}, "chunking": {"chunk_size": 8}}
    with RunBoard(None) as board:
        board.enqueue("r1", [batch(ingest)])
    first = _dispatch(fleet, ["normalise", "chunk"], config=config)
    assert first.newly_enqueued == 2
    work(fleet, idle_timeout=3)

    meta = pq.read_metadata(str(Path(str(store)) / "runs/r1/documents/batch-0001.chunks.parquet")).metadata
    assert meta[b"womblex.workflow_digest"] == first.digest.encode()

    assert _dispatch(fleet, ["normalise", "chunk"], config=config).newly_enqueued == 0  # same workflow
    second = _dispatch(fleet, ["normalise", "chunk"], config={**config, "chunking": {"chunk_size": 9}})
    assert second.newly_enqueued == 2  # same run, other recipe: its own stages, not the first's results
    work(fleet, idle_timeout=3)
    meta = pq.read_metadata(str(Path(str(store)) / "runs/r1/documents/batch-0001.chunks.parquet")).metadata
    assert meta[b"womblex.workflow_digest"] == second.digest.encode()


def test_a_workflow_that_names_pii_masks_on_the_fleet(fleet):  # noqa: F811
    """Phase 3, item 3.3: a fleet run performs `pii` when its workflow names the step."""
    store, ingest, _ = fleet
    with RunBoard(None) as board:
        board.enqueue("r1", [batch(ingest)])
    dispatched = _dispatch(fleet, ["chunk", "pii"])
    assert dispatched.stages == ("chunk", "pii")
    work(fleet, idle_timeout=3)
    with RunBoard(None) as board:
        assert {r.stage: r.status for r in board.list_jobs("r1") if r.kind == "stage"} == {
            "chunk": "done", "pii": "done"}
    clean = Path(str(store)) / "runs/r1/documents/batch-0001.clean_text.parquet"
    assert pq.read_metadata(str(clean)).metadata[b"womblex.workflow_digest"] == dispatched.digest.encode()


def test_cli_exit_codes_and_one_line_per_refusal(fleet, caplog, monkeypatch):  # noqa: F811
    import argparse
    import logging

    from tests.test_workflow_runner import _PRESET, write_workflow
    from womblex.cli.workflow import cmd_dispatch_workflow
    from womblex.presets import build_preset_record, preset_filename, serialise_preset_record

    store, _, tmp = fleet
    write_workflow(tmp, "good", ["normalise"], config=_NORMALISED)
    write_workflow(tmp, "misordered", ["chunk", "normalise"], config=_NORMALISED)
    record, _ = build_preset_record(name=_PRESET.name, description="", formats=(), config=_PRESET.config)
    (tmp / preset_filename(_PRESET.name)).write_text(serialise_preset_record(record))

    def run(wid: str, **kw) -> tuple[int, list[str]]:
        caplog.clear()
        args = argparse.Namespace(**{
            "workflow": tmp / f"{wid}.yaml", "run_id": "r1", "store": str(store), "output_prefix": None,
            "max_attempts": 3, "dry_run": False, "dsn": None, "presets_dir": tmp, **kw})
        with caplog.at_level(logging.INFO):
            rc = cmd_dispatch_workflow(args)
        return rc, [r.getMessage() for r in caplog.records if r.name == "womblex"]

    rc, lines = run("missing")
    assert rc == 1 and len(lines) == 1 and lines[0].startswith("could not read workflow file ")
    rc, lines = run("misordered")
    assert rc == 1 and len(lines) == 1 and lines[0].startswith("[order_not_pipeline] ")
    rc, lines = run("good", dry_run=True)
    assert rc == 0 and "dry run: nothing queued" in lines[-1]
    with RunBoard(None) as board:
        assert board.list_jobs("r1") == []
    rc, lines = run("good")
    assert rc == 0 and "1 newly dispatched" in lines[-1]
    with RunBoard(None) as board:
        assert [r.kind for r in board.list_jobs("r1")] == ["downstream"]
