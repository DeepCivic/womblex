"""The fleet skips a unit only when its outputs were published for the same work."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("dbos")

from tests.test_cloud import _minimal_config
from tests.test_workflows import batch, fleet, work  # noqa: F401
from womblex.cloud import dbos_app
from womblex.cloud.jobs import RunBoard
from womblex.cloud.stage_contracts import STAGE_CONTRACTS
from womblex.cloud.unit_receipt import already_published
from womblex.cloud.workflow_runner import config_payload
from womblex.store.remote import RemoteStore

_PREFIX = "runs/r1/documents"


def _stage_result(board: RunBoard, firing: str | None) -> dict:
    scope = dbos_app.stage_scope(firing, None)
    (row,) = board._client.list_workflows(workflow_ids=[dbos_app.stage_id("r1", "normalise", scope)],
                                          load_input=False, load_output=True)
    return row.output


def test_a_redispatch_skips_published_units_and_a_change_runs_them(fleet) -> None:  # noqa: F811
    store_dir, ingest, tmp = fleet
    config = _minimal_config(tmp)
    sidecar = Path(str(store_dir)) / _PREFIX / "batch-0001.normalised_text.parquet"
    with RunBoard(None) as board:
        board.enqueue("r1", [batch(ingest)], config=config_payload(config))
        board.enqueue_stages("r1", ["normalise"], _PREFIX, config=config_payload(config))
        work(fleet, idle_timeout=2)
        assert _stage_result(board, None)["skipped"] == 0
        written = sidecar.stat().st_mtime_ns

        # Same config, same inputs, new stage ids: nothing runs again.
        board.enqueue_stages("r1", ["normalise"], _PREFIX, config=config_payload(config),
                             firing="again", bases=["batch-0001"])
        work(fleet, idle_timeout=2)
        assert _stage_result(board, "again") | {"checksums": {}} == {
            "stage": "normalise", "units": 1, "published": 0, "skipped": 1, "checksums": {}}
        assert sidecar.stat().st_mtime_ns == written

        # The stage's own config changed: the unit runs.
        changed = config.model_copy(update={"normalise": config.normalise.model_copy(
            update={"unicode_hygiene": not config.normalise.unicode_hygiene})})
        board.enqueue_stages("r1", ["normalise"], _PREFIX, config=config_payload(changed),
                             firing="changed", bases=["batch-0001"])
        work(fleet, idle_timeout=2)
        assert _stage_result(board, "changed")["skipped"] == 0

        # An input rewritten under the output (an upstream re-run): the unit runs.
        elements = Path(str(store_dir)) / _PREFIX / "batch-0001.elements.parquet"
        os.utime(elements, ns=(elements.stat().st_atime_ns, elements.stat().st_mtime_ns + 10**9))
        board.enqueue_stages("r1", ["normalise"], _PREFIX, config=config_payload(changed),
                             firing="touched", bases=["batch-0001"])
        work(fleet, idle_timeout=2)
        assert _stage_result(board, "touched")["skipped"] == 0
        assert all(r.status == "done" for r in board.list_jobs("r1") if r.kind == "stage")


def test_never_skipped_without_receipt_or_for_an_in_place_stage(fleet) -> None:  # noqa: F811
    store_dir, ingest, tmp = fleet
    config = _minimal_config(tmp)
    store = RemoteStore.from_uri(str(store_dir))
    with RunBoard(None) as board:
        board.enqueue("r1", [batch(ingest)], config=config_payload(config))
        work(fleet, idle_timeout=2)
    assert not already_published(STAGE_CONTRACTS["normalise"], config, store, _PREFIX, ["batch-0001"])  # no output
    assert not already_published(STAGE_CONTRACTS["layout"], config, store, _PREFIX, ["batch-0001"])  # in place

    # An output written outside a recorded unit (`womblex normalise --shards`) carries no receipt.
    from womblex.process.normalise_stage import normalise_shards

    local = Path(str(store_dir)) / _PREFIX
    normalise_shards(local, config.normalise)
    assert (local / "batch-0001.normalised_text.parquet").exists()
    assert not already_published(STAGE_CONTRACTS["normalise"], config, store, _PREFIX, ["batch-0001"])
