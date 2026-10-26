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
    assert record["record_version"] == RUN_RECORD_VERSION == 9
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


def _outcome(status: str = "delivered"):
    from womblex.cloud.delivery import DeliveryOutcome

    return DeliveryOutcome("archive", "object_store", "mem://x", "wf", status, 1)


def test_a_redelivery_keeps_the_record_it_replaces_and_says_what_moved(tmp_path: Path):
    import pyarrow.parquet as pq

    from womblex.cloud.delivery import record_deliveries

    run_root, _ = _build_run(tmp_path)
    write_run_manifest(run_root / "documents")
    run = RemoteStore.from_uri(str(run_root))
    record_deliveries(run, [_outcome()])
    first = json.loads((run_root / "deliveries" / "archive.json").read_text())
    assert (first["seq"], first["replaces"], first["changed_documents"]) == (1, None, None)
    assert set(first["document_digests"]) == set(pq.read_table(str(run_root / "manifest.parquet"))["source_hash"].to_pylist())

    table = pq.read_table(str(run_root / "manifest.parquet"))
    moved = table["source_hash"].to_pylist()[0]
    digests = ["different" if h == moved else d for h, d in zip(
        table["source_hash"].to_pylist(), table["content_digest"].to_pylist(), strict=True)]
    pq.write_table(table.set_column(table.schema.get_field_index("content_digest"), "content_digest",
                                    __import__("pyarrow").array(digests)), str(run_root / "manifest.parquet"))
    record_deliveries(run, [_outcome("failed")])

    assert json.loads((run_root / "deliveries" / "archive.1.json").read_text()) == first  # kept, not rewritten
    second = json.loads((run_root / "deliveries" / "archive.json").read_text())
    assert (second["seq"], second["replaces"], second["changed_documents"]) == (2, 1, [moved])

    listed = read_delivery_records(run_root)
    assert [(d["seq"], d["replaces"], d["replaced_by"], d["documents_changed"]) for d in listed] == [
        (1, None, 2, None), (2, 1, None, 1)]
    assert all("document_digests" not in d and "changed_documents" not in d for d in listed)


def test_a_record_written_before_sequences_is_the_first_delivery_and_a_missing_manifest_is_not_fatal(tmp_path: Path):
    from womblex.cloud.delivery import record_deliveries

    run_root, _ = _build_run(tmp_path)
    (run_root / "manifest.parquet").unlink()
    _record(run_root, "archive", status="delivered")
    record_deliveries(RemoteStore.from_uri(str(run_root)), [_outcome()])
    assert json.loads((run_root / "deliveries" / "archive.1.json").read_text())["status"] == "delivered"
    latest = json.loads((run_root / "deliveries" / "archive.json").read_text())
    assert (latest["seq"], latest["replaces"], latest["document_digests"], latest["changed_documents"]) == (
        2, 1, None, None)
