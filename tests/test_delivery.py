"""Workflow destinations: checked before launch, delivered after the last step, recorded on the run."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests._synthetic import REGISTER_CSV
from tests.test_workflow_runner import _resolve, _stored
from womblex.batch import process_batch
from womblex.cloud import delivery
from womblex.cloud.stage_contracts import STAGE_CONTRACTS
from womblex.cloud.workflow_runner import RefusalCode, WorkflowRefusedError, run_workflow_local
from womblex.cloud.workflow_store import validate_workflow_spec, workflow_digest
from womblex.config import DatasetConfig, PathsConfig, WomblexConfig
from womblex.store.run_manifest import read_run_record
from womblex.store.run_stamp import RunStamp
from womblex.store.source_provenance import IngestProvenance


@pytest.fixture
def run_root(tmp_path: Path) -> Path:
    """A finished local run of one synthetic register, ingested from its own directory."""
    cfg = WomblexConfig(
        dataset=DatasetConfig(name="t"),
        paths=PathsConfig(input_root=tmp_path, output_root=tmp_path, checkpoint_dir=tmp_path),
    )
    root = tmp_path / "run-7"
    (root / "documents").mkdir(parents=True)
    process_batch(
        [REGISTER_CSV], cfg, batch_num=1, shard_dir=root / "documents",
        provenance=IngestProvenance.declare(REGISTER_CSV.parent, "t"),
        stamp=RunStamp.declare("run-7", cfg, stage="extract"),
    )
    return root


def _to(tmp_path: Path, name: str = "archive", **extra) -> dict:
    return {"kind": "object_store", "name": name, "uri": str(tmp_path / "out" / name), **extra}


def test_a_run_is_delivered_after_its_last_step_and_the_run_record_lists_it(run_root: Path, tmp_path: Path):
    stored = _stored(["normalise"], destinations=[_to(tmp_path)])

    result = run_workflow_local(stored, run_root / "documents", resolve_preset=_resolve)

    assert result.exit_code == 0
    (outcome,) = result.deliveries
    assert (outcome.status, outcome.attempts, outcome.after_failed) == ("delivered", 1, [])
    bundle = tmp_path / "out" / "archive" / "run-7"
    assert list((bundle / "corpus").glob("*.normalised_text.parquet"))  # the steps' output is in it
    assert json.loads((bundle / "egress_manifest.json").read_text())["delivery"] == {
        "workflow": "wf", "destination": "archive", "after_failed": []}
    assert list((bundle / "sources").iterdir())  # sources included unless the destination says not
    record = read_run_record(run_root / "manifest.parquet")
    assert [(d["destination"], d["status"]) for d in record["deliveries"]] == [("archive", "delivered")]


def test_a_delivery_after_a_failed_step_names_it(run_root: Path, tmp_path: Path, monkeypatch):
    def crash(*_a) -> None:
        raise RuntimeError("boom")

    monkeypatch.setitem(STAGE_CONTRACTS, "normalise", replace(STAGE_CONTRACTS["normalise"], run=crash))
    stored = _stored(["normalise"], destinations=[_to(tmp_path, include_sources=False)])

    result = run_workflow_local(stored, run_root / "documents", resolve_preset=_resolve)

    assert result.exit_code == 1
    assert result.deliveries[0].after_failed == ["normalise"]
    bundle = tmp_path / "out" / "archive" / "run-7"
    assert json.loads((bundle / "egress_manifest.json").read_text())["delivery"]["after_failed"] == ["normalise"]
    assert not (bundle / "sources").exists()


def test_a_failing_destination_is_retried_then_recorded_and_the_run_completes(
    run_root: Path, tmp_path: Path, monkeypatch,
):
    calls: list[int] = []

    def refused(*_a, **_k):
        calls.append(1)
        raise OSError("destination refused the write")

    monkeypatch.setattr("womblex.store.egress.build_bundle", refused)
    monkeypatch.setattr(delivery.time, "sleep", lambda _s: None)
    stored = _stored(["normalise"], destinations=[_to(tmp_path, "bad"), _to(tmp_path, "also-bad")])

    result = run_workflow_local(stored, run_root / "documents", resolve_preset=_resolve)

    assert [s.status for s in result.steps] == ["ok"]
    assert [(d.destination, d.status, d.attempts) for d in result.deliveries] == [
        ("bad", "failed", delivery.DELIVERY_ATTEMPTS), ("also-bad", "failed", delivery.DELIVERY_ATTEMPTS)]
    assert len(calls) == 2 * delivery.DELIVERY_ATTEMPTS
    assert result.exit_code == 1
    saved = json.loads((run_root / "deliveries" / "bad.json").read_text())
    assert saved["status"] == "failed" and "refused the write" in saved["error"]


def test_a_redelivery_writes_the_next_record_and_keeps_the_one_it_replaced(run_root: Path, tmp_path: Path):
    stored = _stored(["normalise"], destinations=[_to(tmp_path, include_sources=False)])
    for _ in range(2):
        run_workflow_local(stored, run_root / "documents", resolve_preset=_resolve)
    assert sorted(p.name for p in (run_root / "deliveries").iterdir()) == ["archive.1.json", "archive.json"]
    listed = read_run_record(run_root / "manifest.parquet")["deliveries"]
    assert [(d["seq"], d["replaces"], d["replaced_by"]) for d in listed] == [(1, None, 2), (2, 1, None)]


@pytest.mark.parametrize("destination, problem", [
    ({"credentials": {"key_env": "WOMBLEX_TEST_UNSET_KEY", "secret_env": "WOMBLEX_TEST_UNSET_SECRET"}},
     "WOMBLEX_TEST_UNSET_KEY, WOMBLEX_TEST_UNSET_SECRET not set"),
    ({"uri": "az://container/x"}, "cannot write to az://"),  # adlfs is not installed
])
def test_a_destination_that_cannot_be_reached_refuses_before_any_step(
    run_root: Path, tmp_path: Path, monkeypatch, destination: dict, problem: str,
):
    ran: list[str] = []
    monkeypatch.setitem(STAGE_CONTRACTS, "normalise", replace(
        STAGE_CONTRACTS["normalise"], run=lambda *_: ran.append("normalise")))
    stored = _stored(["normalise"], destinations=[{**_to(tmp_path), **destination}])

    with pytest.raises(WorkflowRefusedError, match=problem) as refused:
        run_workflow_local(stored, run_root / "documents", resolve_preset=_resolve)
    assert refused.value.code is RefusalCode.DESTINATION_UNAVAILABLE
    assert ran == []


def test_a_workflow_that_delivers_runs_over_a_runs_documents(run_root: Path, tmp_path: Path):
    from womblex.cloud.workflow_validate import validate_workflow

    other = tmp_path / "loose"
    (run_root / "documents").rename(other)
    stored = _stored(["normalise"], destinations=[_to(tmp_path)])
    with pytest.raises(WorkflowRefusedError, match="documents/") as refused:
        run_workflow_local(stored, other, resolve_preset=_resolve)
    assert refused.value.code is RefusalCode.DESTINATION_UNAVAILABLE
    report = validate_workflow(stored, resolve_preset=_resolve, shard_dir=other)  # refused alike
    assert (report.valid, report.code) == (False, RefusalCode.DESTINATION_UNAVAILABLE.value)


@pytest.mark.parametrize("destinations, message", [
    ([{"kind": "object_store", "name": "a", "uri": "s3://key:secret@bucket/x"}], "names no credentials"),
    ([{"kind": "object_store", "name": "a", "uri": "/x"}, {"kind": "object_store", "name": "a", "uri": "/y"}],
     "more than once"),
    ([{"kind": "object_store", "name": "a", "uri": "s3:/bucket/x"}], "needs '://'"),
    ([{"kind": "object_store", "name": "a", "uri": "ftp://host/x"}], "not one of"),
])
def test_a_destination_never_holds_a_secret_and_names_are_unique(destinations: list, message: str):
    with pytest.raises(ValueError, match=message):
        validate_workflow_spec({"preset": "p", "steps": [{"stage": "chunk"}], "destinations": destinations})


def test_an_unknown_destination_kind_is_refused():
    with pytest.raises(ValidationError):
        validate_workflow_spec({"preset": "p", "steps": [{"stage": "chunk"}],
                                "destinations": [{"kind": "ftp", "name": "a", "uri": "/x"}]})


def test_a_workflow_with_no_destinations_keeps_its_digest():
    """The digest this spec had before destinations existed, so its stage ids and stamps carry on."""
    spec = validate_workflow_spec({"preset": "p", "steps": [{"stage": "chunk"}]})
    assert workflow_digest(spec) == "sha256:8215ebc4678e4e432680e153ff2675b3c54462c3871584eb1d4b0051b232d7d6"
    named = validate_workflow_spec({"preset": "p", "steps": [{"stage": "chunk"}],
                                    "destinations": [{"kind": "object_store", "name": "a", "uri": "/x"}]})
    assert workflow_digest(named) != workflow_digest(spec)


def test_dispatch_checks_destinations_before_any_row(tmp_path: Path):
    from womblex.cloud.workflow_dispatch import dispatch_workflow

    unset = _to(tmp_path, credentials={"key_env": "WOMBLEX_TEST_UNSET_KEY", "secret_env": "WOMBLEX_TEST_UNSET_SECRET"})
    with pytest.raises(WorkflowRefusedError, match="WOMBLEX_TEST_UNSET_KEY") as refused:
        dispatch_workflow(_stored(["normalise"], destinations=[unset]), resolve_preset=_resolve,
                          run_id="r1", store_uri=None, dsn=None, dry_run=True)
    assert refused.value.code is RefusalCode.DESTINATION_UNAVAILABLE


def test_womblex_deliver_redelivers_and_refuses_what_it_cannot(run_root: Path, tmp_path: Path):
    import argparse

    import yaml

    from womblex.cli.workflow import cmd_deliver

    def deliver(destinations: list) -> int:
        path = tmp_path / "wf.yaml"
        path.write_text(yaml.safe_dump({"name": "wf", "preset": "saved-offline", "steps": [{"stage": "normalise"}],
                                        "destinations": destinations}))
        return cmd_deliver(argparse.Namespace(workflow=path, shards=run_root / "documents", run_id=None, store=None,
                                              output_prefix=None, dsn=None))

    assert deliver([_to(tmp_path, include_sources=False)]) == 0
    assert (tmp_path / "out" / "archive" / "run-7" / "egress_manifest.json").is_file()
    assert read_run_record(run_root / "manifest.parquet")["deliveries"][0]["status"] == "delivered"
    assert deliver([]) == 1
    assert deliver([{**_to(tmp_path), "uri": "nosuchscheme://x"}]) == 1


def test_womblex_deliver_on_the_fleet_rewrites_the_manifest_before_it_delivers(tmp_path: Path, monkeypatch):
    """A retry made since the run finished changed files the manifest lists, so the bundle must
    carry a manifest written after it, as the coordinator's does."""
    import argparse

    import yaml

    import womblex.cloud.delivery as delivery_module
    import womblex.cloud.finalize as finalize_module
    from womblex.cli.workflow import cmd_deliver

    calls: list[str] = []
    monkeypatch.setattr(finalize_module, "finalize_from_board", lambda *_a: calls.append("finalize"))
    monkeypatch.setattr(delivery_module, "deliver", lambda *_a, **_k: calls.append("deliver") or [])
    monkeypatch.setattr(delivery_module, "record_deliveries", lambda *_a: calls.append("record"))
    path = tmp_path / "wf.yaml"
    path.write_text(yaml.safe_dump({"name": "wf", "preset": "saved-offline", "steps": [{"stage": "normalise"}],
                                    "destinations": [_to(tmp_path)]}))
    assert cmd_deliver(argparse.Namespace(workflow=path, shards=None, run_id="r1", store=str(tmp_path / "store"),
                                          output_prefix=None, dsn=None)) == 0
    assert calls == ["finalize", "deliver", "record", "finalize"]


def test_a_retry_names_the_workflow_to_redeliver():
    from womblex.cloud.retry import _delivering

    spec = {"steps": [{"stage": "chunk"}], "destinations": [{"kind": "object_store", "name": "a", "uri": "/x"}]}
    assert _delivering({"workflow": {"id": "nightly", "spec": spec}}) == "nightly"
    assert _delivering({"workflow": {"id": "nightly", "spec": {**spec, "destinations": []}}}) is None
    assert _delivering({"config": {}}) is None
