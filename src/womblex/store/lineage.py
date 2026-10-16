"""The run record as an OpenLineage RunEvent.

A mapping, not a second account: every field comes from the run record
(`store/run_manifest.build_run_record`), which is observed from the run's own
files, never from config. The event conforms to the OpenLineage 2-0-2 spec
(`fixtures/openlineage/` holds the published schema the test checks against).
Womblex's own detail rides in `womblex_*` facets; nothing is sent anywhere.

OpenLineage wants a UUID run id, so `runId` is a UUID5 of the Womblex run id,
stable across re-exports, with the run id itself in the `womblex_run` facet. A
run record is written once its manifests are consolidated, so the event is
`COMPLETE`; what the record could not establish is its `partial` list, carried
as is.
"""

from __future__ import annotations

import uuid
from typing import Any

SPEC_URL = "https://openlineage.io/spec/2-0-2/OpenLineage.json"
_EVENT_SCHEMA = f"{SPEC_URL}#/$defs/RunEvent"
_REPO = "https://github.com/DeepCivic/womblex"
_FACET_SCHEMA = f"{_REPO}/blob/main/docs/contract.md"
_RUN_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, _REPO)


class LineageError(ValueError):
    """The record cannot be mapped (it names no run)."""


def run_uuid(run_id: str) -> str:
    """The OpenLineage `runId` for a Womblex run id; the same id always gives the same UUID."""
    return str(uuid.uuid5(_RUN_NAMESPACE, run_id))


def run_event(record: dict[str, Any], *, namespace: str = "womblex") -> dict[str, Any]:
    """The OpenLineage RunEvent for *record*; *namespace* is the job's namespace."""
    run = record.get("run")
    if not run or not run.get("run_id"):
        raise LineageError("the run record names no run (its shards carry no run stamp)")
    producer = f"{_REPO}/tree/{run.get('commit') or run.get('version') or 'unknown'}"

    def facet(**fields: Any) -> dict[str, Any]:
        return {"_producer": producer, "_schemaURL": _FACET_SCHEMA, **fields}

    inputs = record.get("inputs") or {}
    run_id = run["run_id"]
    files = record.get("files")
    outputs = (
        [{"namespace": f"womblex:{run_id}", "name": f["key"],
          "facets": {"womblex_file": facet(sha256=f["sha256"])}} for f in files]
        if files is not None
        else [{"namespace": f"womblex:{run_id}", "name": "documents", "facets": {}}]
    )
    return {
        "eventType": "COMPLETE",
        "eventTime": record["generated_at"],
        "producer": producer,
        "schemaURL": _EVENT_SCHEMA,
        "run": {
            "runId": run_uuid(run_id),
            "facets": {"womblex_run": facet(
                run_id=run_id, version=run.get("version"), commit=run.get("commit"),
                config_digest=run.get("config_digest"),
                record_version=record.get("record_version"),
                image=record.get("image"), model_check=record.get("model_check", []),
                partial=record.get("partial", []),
            )},
        },
        "job": {
            "namespace": namespace,
            "name": inputs.get("collection_id") or run_id,
            "facets": {"womblex_job": facet(
                stages=record.get("stages", []), workflows=record.get("workflows", []),
                slot_models=record.get("slot_models", []),
                local_models=record.get("local_models", []),
                services=record.get("services", []),
            )},
        },
        "inputs": [{
            "namespace": inputs.get("ingest_root") or "womblex",
            "name": inputs.get("collection_id") or "corpus",
            "facets": {"womblex_corpus": facet(
                documents=inputs.get("documents"),
                extraction_methods=inputs.get("extraction_methods", {}),
                statuses=inputs.get("statuses", {}),
            )},
        }],
        "outputs": outputs,
    }


__all__ = ["SPEC_URL", "LineageError", "run_event", "run_uuid"]
