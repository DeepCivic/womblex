"""Conditions: labels over manifest columns, written per batch and logged; never a gate."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from tests.test_workflow_runner import _resolve, _stored, shards  # noqa: F401
from womblex.cloud.workflow_conditions import label_manifest
from womblex.cloud.workflow_runner import RefusalCode, WorkflowRefusedError, run_workflow_local
from womblex.cloud.workflow_store import StoredWorkflowSpec, WorkflowCondition, validate_workflow_spec
from womblex.cloud.workflow_validate import validate_workflow
from womblex.store.conditions_output import OUTCOMES_KEY, read_conditions
from womblex.store.contract import SENSITIVITY_KEY
from womblex.store.output import read_manifest


def _cond(column: str, op: str, value: object) -> dict:
    return {"column": column, "op": op, "value": value}


def _status(shard_dir: Path) -> str:
    return read_manifest(shard_dir).column("status").to_pylist()[0]


@pytest.mark.parametrize("condition", [
    _cond("filename_lower", "eq", "x"),          # not a manifest column
    _cond("status", "gt", "a"),                  # order operator on text
    _cond("elements_count", "eq", "many"),       # text against a number column
    _cond("status", "in", "ok"),                 # in needs a list
    _cond("status", "in", []),
    {**_cond("status", "eq", "ok"), "stage": "pii"},  # a condition cannot name a step
])
def test_a_condition_that_could_never_be_evaluated_is_not_stored(condition: dict) -> None:
    with pytest.raises((ValueError, ValidationError)):
        validate_workflow_spec({"preset": "p", "steps": [{"stage": "normalise"}], "conditions": [condition]})


def test_each_document_is_labelled_and_a_null_cell_is_unevaluable_for_every_operator() -> None:
    table = pa.table({"source_hash": ["h1", "h2", "h3"], "doc_id": ["a", "b", "c"],
                      "status": ["ok", "error", None]})
    for op, value, want in (
        ("eq", "ok", ["holds", "fails", "unevaluable"]),
        ("ne", "ok", ["fails", "holds", "unevaluable"]),
        ("in", ["ok"], ["holds", "fails", "unevaluable"]),
        ("not_in", ["ok"], ["fails", "holds", "unevaluable"]),
    ):
        rows = label_manifest([WorkflowCondition(column="status", op=op, value=value)], table)
        assert [r["outcome"] for r in rows] == want
        assert rows[2]["cell"] is None and rows[0]["value"] == json.dumps(value)
    count = pa.table({"source_hash": ["h"], "doc_id": ["a"], "elements_count": pa.array([None], pa.int64())})
    assert label_manifest([WorkflowCondition(column="elements_count", op="lt", value=9)], count)[0]["outcome"] == "unevaluable"


def test_a_failing_condition_labels_the_documents_and_the_run_goes_on(
    shards: Path, caplog: pytest.LogCaptureFixture,  # noqa: F811
) -> None:
    ok = _status(shards)
    stored = _stored(["normalise"], conditions=[_cond("status", "ne", ok), _cond("elements_count", "gt", 0)])
    with caplog.at_level(logging.INFO):
        result = run_workflow_local(stored, shards, resolve_preset=_resolve)
    assert result.exit_code == 0 and list(shards.glob("*.normalised_text.parquet"))  # the step ran
    labels = read_conditions(shards)
    n = read_manifest(shards).num_rows
    assert labels.num_rows == 2 * n
    by_condition = {i: set(labels.filter(pc.equal(labels["condition_index"], i))["outcome"].to_pylist())
                    for i in (0, 1)}
    assert by_condition == {0: {"fails"}, 1: {"holds"}}
    footer = pq.read_schema(next(shards.glob("*.conditions.parquet"))).metadata
    assert json.loads(footer[OUTCOMES_KEY.encode()]) == {"holds": n, "fails": n, "unevaluable": 0}
    assert footer[SENSITIVITY_KEY.encode()] == b"none"
    assert f"{n} of {n} document(s) fail it" in caplog.text and "the run goes on" in caplog.text


def test_no_manifest_rows_label_nothing_and_refuse_nothing(
    shards: Path, caplog: pytest.LogCaptureFixture,  # noqa: F811
) -> None:
    for f in shards.glob("*._manifest.parquet"):
        empty = read_manifest(f.parent / f.name.removesuffix("._manifest.parquet")).slice(0, 0)
        pq.write_table(empty, f)
    stored = _stored(["normalise"], conditions=[_cond("elements_count", "gte", 0)])
    assert validate_workflow(stored, resolve_preset=_resolve, shard_dir=shards).valid
    with caplog.at_level(logging.WARNING):
        assert run_workflow_local(stored, shards, resolve_preset=_resolve).exit_code == 0
    assert read_conditions(shards).num_rows == 0
    assert "no manifest rows" in caplog.text


def test_a_labelling_failure_after_launch_is_logged_and_the_steps_still_run(
    shards: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch,  # noqa: F811
) -> None:
    from womblex.cloud import workflow_conditions

    def boom(*_: object) -> None:
        raise OSError("disk gone")

    monkeypatch.setattr(workflow_conditions, "label_shard_dir", boom)
    stored = _stored(["normalise"], conditions=[_cond("status", "eq", _status(shards))])
    assert run_workflow_local(stored, shards, resolve_preset=_resolve).exit_code == 0
    assert list(shards.glob("*.normalised_text.parquet"))
    assert "conditions not labelled" in caplog.text


def test_no_entry_point_refuses_a_failing_condition(
    shards: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,  # noqa: F811
) -> None:
    pytest.importorskip("dbos")
    from womblex.cloud.workflow_dispatch import dispatch_workflow
    from womblex.cloud.workflow_plan import plan_workflow

    monkeypatch.setenv("WOMBLEX_DBOS_PATH", str(tmp_path / "dbos.sqlite"))
    store = tmp_path / "store"
    shutil.copytree(shards, store / "runs/r1/documents")
    stored = _stored(["normalise"], conditions=[_cond("status", "ne", _status(shards))])
    assert validate_workflow(stored, resolve_preset=_resolve, shard_dir=shards).valid
    assert plan_workflow(stored, shards, resolve_preset=_resolve).steps
    with caplog.at_level(logging.WARNING):
        sent = dispatch_workflow(stored, resolve_preset=_resolve, run_id="r1",
                                 store_uri=str(store), dsn=None, dry_run=True)
    assert sent.stages == ("normalise",)
    assert "conditions stored but not acted on by dispatch" in caplog.text
    assert not list(shards.glob("*.conditions.parquet"))  # validate, plan and dispatch write nothing


def test_a_row_stored_before_conditions_were_checked_is_refused_on_load(
    shards: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str],  # noqa: F811
) -> None:
    import argparse

    from womblex.cli.workflow import cmd_validate
    from womblex.cloud.workflow_store import WorkflowStore, ensure_workflow_schema

    dsn = f"sqlite:///{tmp_path}/wf.db"
    ensure_workflow_schema(dsn)
    old = StoredWorkflowSpec.model_validate({  # pydantic accepts it; check_spec would not
        "preset": "p", "steps": [{"stage": "normalise"}],
        "conditions": [_cond("manifest.doc_type", "eq", "pdf")],
    })
    with WorkflowStore(dsn) as store:
        store._write("old", old, None)  # as merge 1 stored it, past today's save-time check
        stored = store.get("old")
    assert stored is not None
    with pytest.raises(WorkflowRefusedError) as e:
        run_workflow_local(stored, shards, resolve_preset=_resolve)
    assert e.value.code is RefusalCode.SPEC_INVALID and "manifest.doc_type" in str(e.value)
    args = argparse.Namespace(workflow_id="old", shards=shards, dsn=dsn, presets_dir=None,
                              models_check=None, json=True)
    assert cmd_validate(args) == 1
    body = json.loads(capsys.readouterr().out)  # one JSON object
    assert body["code"] == RefusalCode.SPEC_INVALID.value
