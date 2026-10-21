"""The run record as an OpenLineage RunEvent, checked against the published 2-0-2 schema.

The schema is vendored under `fixtures/openlineage/`. No JSON-schema library is
a dependency, so `_conforms` walks the parts of the schema a RunEvent touches:
`$ref`, `allOf`, `anyOf`, `required`, `type`, `enum`, `items`,
`additionalProperties` and the `uuid` / `date-time` / `uri` formats.
"""

from __future__ import annotations

import json
import uuid
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from tests.test_run_record import _extracted, extraction  # noqa: F401
from womblex.cli.pipeline import cmd_manifest
from womblex.store.lineage import LineageError, run_event, run_uuid
from womblex.store.run_manifest import build_run_record, read_run_record, write_run_manifest

_SCHEMA = json.loads((Path(__file__).parent.parent / "fixtures/openlineage/OpenLineage-2-0-2.json").read_text())
_TYPES = {"object": dict, "string": str, "array": list, "boolean": bool, "integer": int}


def _conforms(value: Any, schema: dict, path: str = "$") -> list[str]:
    """Where *value* breaks *schema*, as JSON paths with the reason; empty if it conforms."""
    if "$ref" in schema:
        return _conforms(value, _SCHEMA["$defs"][schema["$ref"].rsplit("/", 1)[1]], path)
    errors = [e for part in schema.get("allOf", []) for e in _conforms(value, part, path)]
    if "anyOf" in schema and all(_conforms(value, part, path) for part in schema["anyOf"]):
        errors.append(f"{path}: matches no anyOf branch")
    if (kind := _TYPES.get(schema.get("type", ""))) and not isinstance(value, kind):
        return [*errors, f"{path}: not {schema['type']}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in enum")
    fmt = schema.get("format")
    try:
        if fmt == "uuid":
            uuid.UUID(value)
        elif fmt == "date-time":
            datetime.fromisoformat(value)
        elif fmt == "uri" and ":" not in value:
            raise ValueError(value)
    except (TypeError, ValueError):
        errors.append(f"{path}: not a {fmt}")
    if isinstance(value, dict):
        errors += [f"{path}.{k}: required" for k in schema.get("required", []) if k not in value]
        props = schema.get("properties", {})
        for k, v in value.items():
            sub = props.get(k, schema.get("additionalProperties"))
            if isinstance(sub, dict):
                errors += _conforms(v, sub, f"{path}.{k}")
    if isinstance(value, list) and "items" in schema:
        errors += [e for i, v in enumerate(value) for e in _conforms(v, schema["items"], f"{path}[{i}]")]
    return errors


def _event_schema() -> dict:
    return {"$ref": "#/$defs/RunEvent"}


def test_the_checker_catches_a_broken_event(tmp_path, extraction) -> None:  # noqa: F811
    event = run_event(read_run_record(write_run_manifest(_extracted(tmp_path, extraction))))
    del event["job"]["name"]
    event["run"]["runId"] = "run-A"
    event["eventType"] = "DONE"
    errors = _conforms(event, _event_schema())
    assert any(e.endswith("job.name: required") for e in errors)
    assert any("runId: not a uuid" in e for e in errors)
    assert any("eventType" in e for e in errors)


def test_a_run_record_maps_to_a_conforming_run_event(tmp_path, extraction) -> None:  # noqa: F811
    record = read_run_record(write_run_manifest(_extracted(tmp_path, extraction)))
    event = run_event(record)
    assert _conforms(event, _event_schema()) == []
    assert event["eventType"] == "COMPLETE" and event["eventTime"] == record["generated_at"]
    assert event["run"]["runId"] == run_uuid("run-A") == run_event(record)["run"]["runId"]
    run_facet = event["run"]["facets"]["womblex_run"]
    assert run_facet["run_id"] == "run-A" and run_facet["partial"] == record["partial"]
    assert event["job"]["facets"]["womblex_job"]["stages"] == record["stages"]  # observed, as recorded
    assert event["inputs"][0]["facets"]["womblex_corpus"]["documents"] == 1


@pytest.mark.parametrize(("state", "event_type"), [("done", "COMPLETE"), ("failed", "FAIL"), ("cancelled", "ABORT")])
def test_the_event_type_follows_the_runs_outcome(tmp_path, extraction, state, event_type) -> None:  # noqa: F811
    record = read_run_record(write_run_manifest(_extracted(tmp_path, extraction)))
    event = run_event(record, state=state)
    assert _conforms(event, _event_schema()) == []
    assert event["eventType"] == event_type and event["run"]["facets"]["womblex_run"]["status"] == state


def test_an_unfinished_run_has_no_outcome_and_no_state_is_unknown(tmp_path, extraction) -> None:  # noqa: F811
    record = read_run_record(write_run_manifest(_extracted(tmp_path, extraction)))
    with pytest.raises(LineageError):
        run_event(record, state="running")
    assert run_event(record)["run"]["facets"]["womblex_run"]["status"] == "unknown"


def test_published_files_become_output_datasets(tmp_path, extraction) -> None:  # noqa: F811
    shards = _extracted(tmp_path, extraction)
    table = pq.read_table(str(write_run_manifest(shards)))
    record = build_run_record(shards, table, checksums={"documents/batch-0001.parquet": "sha256:ab"})
    event = run_event(record)
    assert _conforms(event, _event_schema()) == []
    assert [(o["name"], o["facets"]["womblex_file"]["sha256"]) for o in event["outputs"]] == [
        ("documents/batch-0001.parquet", "sha256:ab")]


def test_manifest_writes_the_event_and_refuses_a_run_with_no_stamp(tmp_path, extraction) -> None:  # noqa: F811
    shards, out = _extracted(tmp_path, extraction), tmp_path / "lineage.json"
    assert cmd_manifest(Namespace(shards=shards, output=None, lineage=out)) == 0
    assert _conforms(json.loads(out.read_text()), _event_schema()) == []

    for path in shards.glob("*.parquet"):
        pq.write_table(pq.read_table(str(path)).replace_schema_metadata({}), str(path))
    out.unlink()
    assert cmd_manifest(Namespace(shards=shards, output=None, lineage=out)) == 1
    assert not out.exists()
    with pytest.raises(LineageError):
        run_event(read_run_record(write_run_manifest(shards)))
