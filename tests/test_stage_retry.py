"""Retrying a failed stage by fork (D-2.1): a real worker on a SQLite system database."""
# ruff: noqa: F811  (`fleet` is a fixture imported from test_workflows, then taken as an argument)

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

pytest.importorskip("dbos")

from tests.test_cloud import _minimal_config
from tests.test_workflows import batch, fleet, work  # noqa: F401
from womblex.cloud import dbos_app, workflows
from womblex.cloud.jobs import RunBoard, RunOwnedError
from womblex.cloud.retry import RetryCode, RetryRefusedError, retry_stage

PREFIX = "runs/r1/documents"


class Fault:
    """A stage unit that raises while `broken`, and the log of every unit `run_unit` is asked to run."""

    def __init__(self, monkeypatch, stage: str, unit: list[str]):
        self.broken, self.calls = True, []
        real = workflows.run_unit

        def run_unit(contract, config, ctx, store, prefix, u, ingest, by=None):
            self.calls.append((contract.name, u))
            if self.broken and contract.name == stage and u == unit:
                raise RuntimeError(f"{stage} broke on {u[0]}")
            return real(contract, config, ctx, store, prefix, u, ingest, by)

        monkeypatch.setattr(workflows, "run_unit", run_unit)

    def fix(self) -> list[tuple[str, list[str]]]:
        """The fault is fixed; the log so far is returned and restarted."""
        self.broken, seen, self.calls = False, self.calls, []
        return seen


def _failed_run(fleet, monkeypatch, stages=("normalise",), broken="normalise"):
    """Run r1 (two batches) to the point `broken` has failed on its second batch."""
    _, ingest, _ = fleet
    fault = Fault(monkeypatch, broken, ["batch-0002"])
    with RunBoard(None) as board:
        board.enqueue("r1", [batch(ingest, 1), batch(ingest, 2)])
        board.enqueue_stages("r1", list(stages), PREFIX, max_attempts=1)
        work(fleet, idle_timeout=2)
    return fault


def _attempts(stage):
    with RunBoard(None) as board:
        return board.stage_attempts("r1")[stage]


def board_jobs(run_id: str) -> list:
    with RunBoard(None) as board:
        return board.list_jobs(run_id)


def _retry(fleet, stage="normalise", **kw):
    with RunBoard(None) as board:
        return retry_stage(board, "r1", stage, config=_minimal_config(fleet[2]), **kw)


def test_a_failed_stage_is_forked_from_its_first_failed_unit(fleet, monkeypatch):
    fault = _failed_run(fleet, monkeypatch)
    calls = fault.fix()
    retry = _retry(fleet)
    assert retry.retry_id == "r1:stage:normalise:retry:1" and retry.forked_from == "r1:stage:normalise"
    work(fleet, idle_timeout=2)
    assert ("normalise", ["batch-0001"]) in calls
    assert fault.calls == [("normalise", ["batch-0002"])]  # the finished unit's recorded step was kept
    with RunBoard(None) as board:
        rows = {r.id: r for r in board.list_jobs("r1") if r.kind == "stage"}
        stats = board.stats("r1")
        (run,) = board.runs(run_id="r1")
    original, forked = (s.attributes for s in _attempts("normalise"))
    assert forked == original  # the fork is the same stage: owner, digest and all
    assert rows["r1:stage:normalise"].status == "failed"
    assert rows["r1:stage:normalise"].retried_by == "r1:stage:normalise:retry:1"
    assert rows["r1:stage:normalise:retry:1"].status == "done"
    assert rows["r1:stage:normalise:retry:1"].retried_by is None
    assert stats == {"done": 4}  # two batches, the stage's retry, and the coordinator it repaired
    assert run.counts == stats
    assert {r.kind: r.status for r in board_jobs("r1") if r.kind == "downstream"} == {"downstream": "failed"}


def test_a_retry_no_live_worker_serves_is_refused_before_the_fork(fleet, monkeypatch):
    from womblex.cloud.worker_registry import WorkerRegistry

    _failed_run(fleet, monkeypatch)
    with WorkerRegistry(None) as registry:
        registry.remove("standby")
    with pytest.raises(RetryRefusedError) as e:
        _retry(fleet)
    assert e.value.code is RetryCode.NO_WORKER
    assert [s.workflow_id for s in _attempts("normalise")] == ["r1:stage:normalise"]


def test_a_second_retry_is_the_next_attempt(fleet, monkeypatch):
    _failed_run(fleet, monkeypatch)
    _retry(fleet)
    work(fleet, idle_timeout=2)  # the fault is still there: the retry fails too
    assert _retry(fleet).retry_id == "r1:stage:normalise:retry:2"


@pytest.mark.parametrize("stage, code", [("embed", RetryCode.NOT_FOUND), ("normalise", RetryCode.NOT_FAILED)])
def test_a_stage_that_is_unknown_or_succeeded_is_refused(fleet, stage, code):
    _, ingest, _ = fleet
    with RunBoard(None) as board:
        board.enqueue("r1", [batch(ingest)])
        board.enqueue_stages("r1", ["normalise"], PREFIX)
        work(fleet, idle_timeout=2)
    with pytest.raises(RetryRefusedError) as e:
        _retry(fleet, stage)
    assert e.value.code is code


def test_a_stage_still_running_is_refused(fleet):
    with RunBoard(None) as board:
        board._client.enqueue(
            {"workflow_name": dbos_app.STAGE_WORKFLOW, "queue_name": dbos_app.stage_queue("normalise"),
             "workflow_id": dbos_app.stage_id("r1", "normalise"),
             "attributes": {"run_id": "r1", "kind": dbos_app.KIND_STAGE, "stage": "normalise"}},
            "r1", "normalise", PREFIX, 1,
        )
        with pytest.raises(RetryRefusedError) as e:
            retry_stage(board, "r1", "normalise", config=_minimal_config(fleet[2]))
    assert e.value.code is RetryCode.STILL_RUNNING


def test_a_retry_by_another_owner_is_refused(fleet, monkeypatch):
    _failed_run(fleet, monkeypatch)
    with RunBoard(None) as board:
        board.enqueue_stages("r2", ["normalise"], PREFIX, owner="alice")
        with pytest.raises(RunOwnedError):
            board.stage_attempts("r2", owner="bob")


def test_a_retry_that_fails_a_launch_check_forks_nothing(fleet, monkeypatch):
    from womblex.cloud import workflow_order

    _failed_run(fleet, monkeypatch)

    def refuse(*_):
        raise workflow_order.StageOrderError([])

    monkeypatch.setattr(workflow_order, "check_dispatch_order", refuse)
    with pytest.raises(RetryRefusedError) as e:
        _retry(fleet)
    assert e.value.code is RetryCode.STEP_ORDER
    with RunBoard(None) as board:
        assert [s.workflow_id for s in board.stage_attempts("r1")["normalise"]] == ["r1:stage:normalise"]


def test_a_config_that_will_not_load_refuses_the_retry(fleet, monkeypatch):
    from womblex.utils import model_check

    _failed_run(fleet, monkeypatch)

    def unregistered(*_):
        raise ValueError("no such model")

    monkeypatch.setattr(model_check, "check_registered", unregistered)
    with pytest.raises(RetryRefusedError) as e:
        _retry(fleet)
    assert e.value.code is RetryCode.CONFIG_INVALID


def test_stages_that_ran_after_a_failure_are_offered_once_it_is_cured(fleet, monkeypatch, caplog):
    fault = _failed_run(fleet, monkeypatch, stages=("normalise", "chunk"))
    fault.fix()
    from womblex.cli.retry import cmd_retry_stage

    monkeypatch.setattr("womblex.config.load_config", lambda _: _minimal_config(fleet[2]))
    args = argparse.Namespace(run_id="r1", stage="normalise", config=Path("c.yaml"), store=None, dsn=None)
    assert cmd_retry_stage(args) == 0
    assert "womblex retry-stage --run-id r1 --stage chunk" in caplog.text  # offered, to run once cured
    assert cmd_retry_stage(args) == 1  # the retry is still queued: a second is refused
    assert "retry refused" in caplog.text
    with pytest.raises(RetryRefusedError) as e:  # but not before normalise has succeeded
        _retry(fleet, "chunk")
    assert e.value.code is RetryCode.EARLIER_FAILED
    work(fleet, idle_timeout=2)
    fault.fix()
    again = _retry(fleet, "chunk")
    assert again.retry_id == "r1:stage:chunk:retry:1" and again.start_step is None
    assert again.bases == ("batch-0002",)  # only the unit normalise had not finished
    work(fleet, idle_timeout=2)
    assert fault.calls == [("chunk", ["batch-0002"])]  # batch-0001's chunks are not paid for again
    with pytest.raises(RetryRefusedError) as e:  # re-run since: nothing left to offer
        _retry(fleet, "chunk")
    assert e.value.code is RetryCode.NOT_FAILED


def test_stages_offered_after_a_failure_are_in_pipeline_order():
    from types import SimpleNamespace as Row

    from womblex.cloud.retry import _ran_after

    attempts = {s: [Row(status="SUCCESS", attributes={"ran_after_failed": ["normalise"]})]
                for s in ("money", "embed", "chunk", "enrich", "spellfix")}
    attempts["normalise"] = [Row(status="ERROR", attributes={})]
    assert _ran_after(attempts, "normalise") == ("spellfix", "enrich", "chunk", "embed", "money")


def _stage_rows(*specs: tuple[str, str, int]):
    """Fake stage rows of run r1: (workflow id, DBOS status, created_at)."""
    from tests.test_workflows import _Row

    rows = []
    for wid, status, created in specs:
        row = _Row(wid, status, None, kind="stage", created_at=created)
        row.attributes["stage"] = "normalise"
        rows.append(row)
    return rows


@pytest.mark.parametrize(("specs", "chosen"), [
    # the older firing failed and the newer finished: the failure is what a retry reaches
    ([("r1:fire-1:stage:normalise", "ERROR", 1), ("r1:fire-2:stage:normalise", "SUCCESS", 2)],
     ["r1:fire-1:stage:normalise"]),
    # both failed: the newest first, the older once that one is retried successfully
    ([("r1:fire-1:stage:normalise", "ERROR", 1), ("r1:fire-2:stage:normalise", "ERROR", 2)],
     ["r1:fire-2:stage:normalise"]),
    ([("r1:fire-1:stage:normalise", "ERROR", 1), ("r1:fire-2:stage:normalise", "ERROR", 2),
      ("r1:fire-2:stage:normalise:retry:1", "SUCCESS", 3)],
     ["r1:fire-1:stage:normalise"]),
    # every firing finished: the newest
    ([("r1:fire-1:stage:normalise", "SUCCESS", 1), ("r1:fire-2:stage:normalise", "SUCCESS", 2)],
     ["r1:fire-2:stage:normalise"]),
])
def test_stage_attempts_reach_each_failed_firing_in_turn(specs, chosen) -> None:
    from tests.test_workflows import _board

    attempts = _board(_stage_rows(*specs)).stage_attempts("r1")
    assert [a.workflow_id for a in attempts["normalise"]][:1] == chosen


def test_a_firings_retry_is_named_under_its_own_firing() -> None:
    base, number = dbos_app.stage_attempt("r1:fire-2:stage:normalise:retry:1")
    assert (base, number) == ("r1:fire-2:stage:normalise", 1)
    assert dbos_app.stage_retry_id(base, number + 1) == "r1:fire-2:stage:normalise:retry:2"


@pytest.mark.parametrize(("firing", "prefix"), [(None, "r1"), ("20261020T0100", "r1:fire-20261020T0100")])
def test_a_coordinator_counts_done_only_once_every_stage_it_ran_has_succeeded(firing, prefix) -> None:
    """A partly repaired run still reads failed; a firing's coordinator is matched to its own stages."""
    from tests.test_workflows import _board, _Row

    coordinator = _Row("r1:downstream", "ERROR", None, kind="downstream", created_at=1)
    coordinator.attributes.update(stages=["normalise", "chunk"], firing=firing)
    rows = [coordinator] + [_Row(f"{prefix}:stage:{wid}", status, None, kind="stage", created_at=n)
                            for n, (wid, status) in enumerate([("normalise", "ERROR"), ("chunk", "ERROR"),
                                                               ("normalise:retry:1", "SUCCESS")], 2)]
    assert _board(rows).stats("r1") == {"failed": 2, "done": 1}  # chunk has not been repaired
    rows.append(_Row(f"{prefix}:stage:chunk:retry:1", "SUCCESS", None, kind="stage", created_at=5))
    assert _board(rows).stats("r1") == {"done": 3}


def _steps(plan, **units):
    """Recorded steps of a stage attempt: its `plan`, then each `unit:<label>` and whether it failed."""
    return [{"function_name": "plan", "output": plan, "error": None}] + [
        {"function_name": f"unit:{label}", "output": {}, "error": RuntimeError("boom") if failed else None}
        for label, failed in units.items()]


@pytest.mark.parametrize(("steps", "unfinished"), [
    (_steps([["batch-0001"], ["batch-0002"]], **{"batch-0001": False, "batch-0002": True}), ["batch-0002"]),
    (_steps([["batch-0001"], ["batch-0002"]], **{"batch-0001": False}), ["batch-0002"]),  # never reached
    (_steps([["batch-0001", "batch-0002"]], run=True), ["batch-0001", "batch-0002"]),  # a whole-run unit
    (_steps([["batch-0001", "batch-0002"]], run=False), []),
    ([], None),  # no plan recorded: every base
])
def test_unfinished_bases_reads_the_attempts_plan_and_units(steps, unfinished) -> None:
    from tests.test_workflows import _board

    assert _board([], {"r1:stage:normalise": steps}).unfinished_bases("r1:stage:normalise") == unfinished


def test_a_stage_that_ran_after_a_failure_redoes_only_what_each_failure_left() -> None:
    from tests.test_workflows import _board, _Row
    from womblex.cloud.retry import _affected

    def row(wid, status, created, **attrs):
        r = _Row(wid, status, None, kind="stage", created_at=created)
        r.attributes.update(attrs)
        return r

    steps = {"r1:stage:normalise": _steps([["batch-0001"], ["batch-0002"]], **{"batch-0001": False, "batch-0002": True}),
             "r1:stage:spellfix": _steps([["batch-0001"], ["batch-0003"]], **{"batch-0001": False, "batch-0003": True})}
    attempts = {"normalise": [row("r1:stage:normalise", "ERROR", 1)],
                "spellfix": [row("r1:stage:spellfix", "ERROR", 2)]}
    chunk = row("r1:stage:chunk", "SUCCESS", 3, ran_after_failed=["normalise", "spellfix"])
    board = _board([], steps)
    assert _affected(board, attempts, chunk, None) == ["batch-0002", "batch-0003"]
    assert _affected(board, attempts, chunk, ["batch-0002", "batch-0004"]) == ["batch-0002"]  # its own bases only
    assert _affected(_board([], {}), attempts, chunk, ["batch-0002"]) == ["batch-0002"]  # no plan: all it ran
