"""Conditions: checked when saved and loaded, and otherwise not acted on."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.test_workflow_runner import _resolve, _stored, shards  # noqa: F401
from womblex.cloud.workflow_runner import RefusalCode, WorkflowRefusedError, run_workflow_local
from womblex.cloud.workflow_store import StoredWorkflowSpec, validate_workflow_spec
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


def test_no_entry_point_acts_on_a_condition(
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
    assert "conditions" in caplog.text and "not acted on" in caplog.text
    assert run_workflow_local(stored, shards, resolve_preset=_resolve).not_acted_on == ("conditions",)
    assert not list(shards.glob("*.conditions.parquet"))  # nothing labels them


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
