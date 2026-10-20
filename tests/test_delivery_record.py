"""Delivery records: read back into the run record, and a delivered bundle marked with what it follows."""

from __future__ import annotations

import json
from pathlib import Path

from tests.test_egress import _build_run
from womblex.store.delivery_record import delivery_key, dump_delivery, read_delivery_records
from womblex.store.egress import EGRESS_MANIFEST_FILENAME, build_bundle
from womblex.store.remote import RemoteStore
from womblex.store.run_manifest import RUN_RECORD_VERSION, read_run_record, write_run_manifest


def _record(run_root: Path, destination: str, **fields) -> None:
    path = run_root / delivery_key(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_delivery({"destination": destination, **fields}))


def test_the_run_record_lists_every_delivery_by_destination(tmp_path: Path):
    run_root, _ = _build_run(tmp_path)
    _record(run_root, "warehouse", status="failed", after_failed=["enrich"])
    _record(run_root, "archive", status="delivered", after_failed=[])
    (run_root / "deliveries" / "torn.json").write_text("{not json")

    write_run_manifest(run_root / "documents")

    record = read_run_record(run_root / "manifest.parquet")
    assert record["record_version"] == RUN_RECORD_VERSION == 8
    assert [(d["destination"], d["status"]) for d in record["deliveries"]] == [
        ("archive", "delivered"), ("warehouse", "failed")]


def test_a_run_never_delivered_lists_none(tmp_path: Path):
    run_root, _ = _build_run(tmp_path)
    assert read_delivery_records(run_root) == []
    assert read_run_record(run_root / "manifest.parquet")["deliveries"] == []


def test_a_delivered_bundle_says_what_it_follows(tmp_path: Path):
    run_root, _ = _build_run(tmp_path)
    marker = {"workflow": "nightly", "destination": "archive", "after_failed": ["enrich"]}
    build_bundle(run_root, RemoteStore.from_uri(str(tmp_path / "dest")), run_id="run-A",
                 include_sources=False, delivery=marker)
    descriptor = json.loads((tmp_path / "dest" / "run-A" / EGRESS_MANIFEST_FILENAME).read_text())
    assert descriptor["delivery"] == marker
    build_bundle(run_root, RemoteStore.from_uri(str(tmp_path / "plain")), run_id="run-A", include_sources=False)
    assert "delivery" not in json.loads((tmp_path / "plain" / "run-A" / EGRESS_MANIFEST_FILENAME).read_text())
