"""Conditions: guards over manifest columns, decided before any step runs."""

from __future__ import annotations

import shutil
from pathlib import Path

import pyarrow as pa
import pytest
from pydantic import ValidationError

from tests.test_workflow_runner import _resolve, _stored, shards  # noqa: F401
from womblex.cloud.workflow_conditions import ConditionRefused, require_conditions
from womblex.cloud.workflow_runner import RefusalCode, WorkflowRefusedError, run_workflow_local
from womblex.cloud.workflow_store import WorkflowCondition, validate_workflow_spec
from womblex.cloud.workflow_validate import validate_workflow
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


def test_null_never_satisfies_a_condition_not_even_ne() -> None:
    table = pa.table({"doc_id": ["a", "b"], "status": ["ok", None]})
    for op in ("eq", "ne"):
        with pytest.raises(ConditionRefused) as e:
            require_conditions([WorkflowCondition(column="status", op=op, value="x" if op == "ne" else "ok")], table)
        assert e.value.code == "condition_failed" and "1 of 2" in str(e.value)


def test_a_holding_condition_runs_and_a_failing_one_stops_before_any_step(shards: Path) -> None:  # noqa: F811
    ok = _status(shards)
    held = _stored(["normalise"], conditions=[_cond("status", "eq", ok), _cond("elements_count", "gt", 0)])
    assert run_workflow_local(held, shards, resolve_preset=_resolve).exit_code == 0
    for f in shards.glob("*.normalised_text.parquet"):
        f.unlink()

    failing = _stored(["normalise"], conditions=[_cond("status", "ne", ok)])
    with pytest.raises(WorkflowRefusedError) as e:
        run_workflow_local(failing, shards, resolve_preset=_resolve)
    assert e.value.code is RefusalCode.CONDITION_FAILED and "status" in str(e.value)
    assert not list(shards.glob("*.normalised_text.parquet"))  # no step ran


def test_no_manifest_rows_cannot_be_evaluated_and_validate_says_when_it_did_not_try(
    shards: Path,  # noqa: F811
) -> None:
    empty = read_manifest(shards).slice(0, 0)
    with pytest.raises(ConditionRefused) as e:
        require_conditions([WorkflowCondition(column="status", op="eq", value="ok")], empty)
    assert e.value.code == "condition_unevaluable"
    stored = _stored(["normalise"], conditions=[_cond("status", "ne", _status(shards))])
    assert validate_workflow(stored, resolve_preset=_resolve).valid  # no shard dir: not evaluated
    report = validate_workflow(stored, resolve_preset=_resolve, shard_dir=shards)
    assert report.code == RefusalCode.CONDITION_FAILED.value


def test_dispatch_evaluates_conditions_against_the_store_before_any_row(
    shards: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,  # noqa: F811
) -> None:
    pytest.importorskip("dbos")
    from womblex.cloud.workflow_dispatch import dispatch_workflow

    monkeypatch.setenv("WOMBLEX_DBOS_PATH", str(tmp_path / "dbos.sqlite"))
    store = tmp_path / "store"
    shutil.copytree(shards, store / "runs/r1/documents")

    def go(op: str) -> object:
        stored = _stored(["normalise"], conditions=[_cond("status", op, _status(shards))])
        return dispatch_workflow(stored, resolve_preset=_resolve, run_id="r1",
                                 store_uri=str(store), dsn=None, dry_run=True)

    assert go("eq").stages == ("normalise",)  # type: ignore[attr-defined]
    with pytest.raises(WorkflowRefusedError) as e:
        go("ne")
    assert e.value.code is RefusalCode.CONDITION_FAILED
    with pytest.raises(WorkflowRefusedError) as e:  # an empty store: nothing to evaluate
        dispatch_workflow(_stored(["normalise"], conditions=[_cond("status", "eq", "ok")]),
                          resolve_preset=_resolve, run_id="r2", store_uri=str(tmp_path / "none"),
                          dsn=None, dry_run=True)
    assert e.value.code is RefusalCode.CONDITION_UNEVALUABLE
