"""Replaying a coordinator that caught a failed child (#192), and `--status` on an unmigrated Postgres."""

from __future__ import annotations

import argparse

import pytest

pytest.importorskip("dbos")

from tests.test_stage_retry import Fault
from tests.test_workflows import batch, fleet, work  # noqa: F401
from womblex.cloud import dbos_app
from womblex.cloud.jobs import RunBoard


def _crash_after_the_end(board: RunBoard, workflow_id: str) -> None:
    """Put a finished workflow back as a worker that died with it unfinished leaves it."""
    import sqlalchemy as sa
    from dbos._schemas.system_database import SystemSchema

    ws = SystemSchema.workflow_status
    with board._client._sys_db.engine.begin() as c:
        c.execute(sa.update(ws).where(ws.c.workflow_uuid == workflow_id)
                  .values(status="PENDING", error=None, output=None, completed_at=None))


def test_a_replayed_coordinator_reads_its_failed_child_again_and_runs_nothing_twice(
    fleet, monkeypatch,  # noqa: F811
) -> None:
    _, ingest, _ = fleet
    fault = Fault(monkeypatch, "normalise", ["batch-0001"])
    with RunBoard(None) as board:
        board.enqueue("r1", [batch(ingest)])
        board.enqueue_stages("r1", ["normalise", "chunk"], "runs/r1/documents", max_attempts=1)
        work(fleet, idle_timeout=2)
        before = {j.id: (j.status, j.error) for j in board.list_jobs("r1")}
        ran = list(fault.calls)
        (coordinator,) = [j.id for j in board.list_jobs("r1") if j.kind == "downstream"]
        assert before[coordinator][0] == "failed" and before["r1:stage:chunk"][0] == "done"

        _crash_after_the_end(board, coordinator)
        work(fleet, idle_timeout=2)  # recovery replays the coordinator over its recorded steps
        after = {j.id: (j.status, j.error) for j in board.list_jobs("r1")}
    assert after == before  # the caught child failure, the later stage and the final error, as first time
    assert fault.calls == ran  # no unit ran again
    assert sorted(i for i in after if dbos_app.stage_attempt(i)) == ["r1:stage:chunk", "r1:stage:normalise"]


def test_status_on_a_postgres_database_no_writer_has_migrated_prints_nothing(capsys, monkeypatch) -> None:
    from tests._postgres import fresh_database
    from womblex.cli.workflow import cmd_schedule_workflow

    with fresh_database() as url:
        args = argparse.Namespace(workflow="x", dsn=url, remove=False, status=True)
        assert cmd_schedule_workflow(args) == 0
    assert capsys.readouterr().out == ""
