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

_NORMALISED = {"processing": {"text_source": "normalised"}}


def _dispatch(fleet, steps, run_id="r1", **kw):  # noqa: F811
    store, _, _ = fleet
    config = kw.pop("config", {})
    return dispatch_workflow(
        _stored(steps, config=config), resolve_preset=_resolve, run_id=run_id,
        store_uri=str(store), dsn=None, **kw,
    )


@pytest.mark.parametrize(("steps", "config", "code"), [
    (["pii"], {}, RefusalCode.STAGE_NOT_DISPATCHABLE),
    (["chunk", "normalise"], _NORMALISED, RefusalCode.ORDER_NOT_PIPELINE),
    (["chunk"], _NORMALISED, RefusalCode.STEP_ORDER),
])
def test_refusals_come_before_any_row(fleet, steps, config, code):  # noqa: F811
    with pytest.raises(WorkflowRefusedError) as e:
        _dispatch(fleet, steps, config=config)
    assert e.value.code is code
    with RunBoard(None) as board:
        assert board.list_jobs("r1") == []


def test_dry_run_checks_and_queues_nothing(fleet):  # noqa: F811
    result = _dispatch(fleet, ["normalise", "chunk"], config=_NORMALISED, dry_run=True)
    assert result.stages == ("normalise", "chunk") and result.newly_enqueued == 0
    with RunBoard(None) as board:
        assert board.list_jobs("r1") == []


def test_a_workflow_runs_on_the_fleet_and_a_different_one_cannot_follow_it(fleet):  # noqa: F811
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
    with pytest.raises(WorkflowRefusedError) as e:  # same run, other recipe
        _dispatch(fleet, ["normalise", "chunk"], config={**config, "chunking": {"chunk_size": 9}})
    assert e.value.code is RefusalCode.RUN_CONFLICT
