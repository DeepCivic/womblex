"""Tests for the egress bundle builder (store/egress.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from womblex.store.egress import CORPUS_DIRNAME, EGRESS_MANIFEST_FILENAME, build_bundle
from womblex.store.egress_output import SOURCE_INDEX_FILENAME, raw_key_for
from womblex.store.output import ELEMENT_SCHEMA, MANIFEST_SCHEMA, _source_hash, _write_rows
from womblex.store.remote import RemoteStore
from womblex.store.run_manifest import write_run_manifest
from womblex.store.run_stamp import RunStamp
from womblex.store.source_provenance import qualify_root

_STAMP = RunStamp("run-A", "1.2.3", "c" * 40, "sha256:abc", "extract", preset="corpus")


def _manifest_row(source_hash: str, doc_id: str, filename: str, ext: str, *,
                   ingest_root: str, relpath: str, extraction_method: str = "native") -> dict:
    return {
        "source_hash": source_hash, "collection_id": "", "doc_id": doc_id,
        "ingest_root": ingest_root, "source_relpath": relpath, "filename": filename,
        "ext": ext, "extraction_method": extraction_method, "elements_count": 1,
        "table_cells_count": 0, "form_fields_count": 0, "status": "ok", "error": "",
        "extracted_at_iso": "2026-01-01T00:00:00Z", "parser_version": "1.0",
    }


def _build_run(tmp_path: Path) -> tuple[Path, Path]:
    """A finished local run plus its corpus directory, with one resolvable
    document, one duplicate-hash document, one unresolvable (missing file)
    document, and one records-hashed (unsupported basis) document."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "report.pdf").write_bytes(b"%PDF-1.4 report bytes")
    report_hash = _source_hash(str(corpus_dir / "report.pdf"))
    root = qualify_root(corpus_dir)

    run_root = tmp_path / "run"
    shard_dir = run_root / "documents"
    shard_dir.mkdir(parents=True)

    rows = [
        _manifest_row(report_hash, "doc-1", "report.pdf", ".pdf",
                      ingest_root=root, relpath="report.pdf"),
        _manifest_row(report_hash, "doc-1-dup", "report-copy.pdf", ".pdf",
                      ingest_root=root, relpath="report.pdf"),
        _manifest_row("missing" * 8, "doc-2", "gone.pdf", ".pdf",
                      ingest_root=root, relpath="gone.pdf"),
        _manifest_row("recid" * 8, "doc-3", "record-42.txt", ".txt",
                      ingest_root=root, relpath="record-42.txt", extraction_method="records"),
    ]
    footer = _STAMP.footer_metadata()
    _write_rows([], shard_dir / "batch-0001.elements.parquet", ELEMENT_SCHEMA, metadata=footer)
    _write_rows(rows, shard_dir / "batch-0001._manifest.parquet", MANIFEST_SCHEMA, metadata=footer)
    write_run_manifest(shard_dir)
    return run_root, corpus_dir


def test_missing_manifest_raises(tmp_path: Path):
    run_root = tmp_path / "run"
    (run_root / "documents").mkdir(parents=True)
    store = RemoteStore.from_uri(str(tmp_path / "dest"))

    with pytest.raises(FileNotFoundError, match="finalise"):
        build_bundle(run_root, store, run_id="run-A")


def test_corpus_is_mirrored_flat_under_the_bundle(tmp_path: Path):
    run_root, _ = _build_run(tmp_path)
    dest = tmp_path / "dest"
    store = RemoteStore.from_uri(str(dest))

    build_bundle(run_root, store, run_id="run-A")

    bundle_corpus = dest / "run-A" / CORPUS_DIRNAME
    assert (bundle_corpus / "manifest.parquet").is_file()
    assert (bundle_corpus / "batch-0001.elements.parquet").is_file()
    assert (bundle_corpus / "batch-0001._manifest.parquet").is_file()


def test_resolved_source_is_copied_and_indexed(tmp_path: Path):
    run_root, corpus_dir = _build_run(tmp_path)
    dest = tmp_path / "dest"
    store = RemoteStore.from_uri(str(dest))

    result = build_bundle(run_root, store, run_id="run-A")

    report_hash = _source_hash(str(corpus_dir / "report.pdf"))
    raw_key = raw_key_for(report_hash, ".pdf")
    assert (dest / "run-A" / raw_key).read_bytes() == (corpus_dir / "report.pdf").read_bytes()

    index = pq.read_table(str(dest / "run-A" / SOURCE_INDEX_FILENAME)).to_pylist()
    by_doc = {r["doc_id"]: r for r in index}
    assert by_doc["doc-1"]["status"] == "resolved"
    assert by_doc["doc-1"]["raw_key"] == raw_key
    assert by_doc["doc-2"]["status"] == "not_found"
    assert by_doc["doc-2"]["raw_key"] is None
    assert by_doc["doc-3"]["status"] == "unsupported_basis"
    assert result.documents == 4


def test_duplicate_hash_is_uploaded_once_but_indexed_per_document(tmp_path: Path):
    run_root, corpus_dir = _build_run(tmp_path)
    store = RemoteStore.from_uri(str(tmp_path / "dest"))

    result = build_bundle(run_root, store, run_id="run-A")

    index = pq.read_table(str(tmp_path / "dest" / "run-A" / SOURCE_INDEX_FILENAME)).to_pylist()
    resolved_keys = {r["raw_key"] for r in index if r["status"] == "resolved"}
    assert resolved_keys == {raw_key_for(_source_hash(str(corpus_dir / "report.pdf")), ".pdf")}
    assert result.sources_copied == 1  # one upload, not two, for the shared hash


def test_source_index_is_stamped_with_the_exported_run_at_stage_egress(tmp_path: Path):
    run_root, _ = _build_run(tmp_path)
    dest = tmp_path / "dest"
    store = RemoteStore.from_uri(str(dest))

    build_bundle(run_root, store, run_id="run-A")

    meta = pq.read_schema(str(dest / "run-A" / SOURCE_INDEX_FILENAME)).metadata
    assert meta[b"womblex.run_id"] == b"run-A"
    assert meta[b"womblex.stage"] == b"egress"


def test_egress_manifest_reports_the_export(tmp_path: Path):
    run_root, _ = _build_run(tmp_path)
    dest = tmp_path / "dest"
    store = RemoteStore.from_uri(str(dest))

    result = build_bundle(run_root, store, run_id="run-A")

    descriptor = json.loads((dest / "run-A" / EGRESS_MANIFEST_FILENAME).read_text())
    assert descriptor["run_id"] == "run-A"
    assert descriptor["documents"] == 4
    assert descriptor["sources_copied"] == result.sources_copied == 1
    assert descriptor["sources_by_status"]["not_found"] == 1
    assert descriptor["sources_by_status"]["unsupported_basis"] == 1


def test_corpus_only_export_skips_sources_entirely(tmp_path: Path):
    run_root, _ = _build_run(tmp_path)
    dest = tmp_path / "dest"
    store = RemoteStore.from_uri(str(dest))

    result = build_bundle(run_root, store, run_id="run-A", include_sources=False)

    assert not (dest / "run-A" / "sources").exists()
    assert not (dest / "run-A" / SOURCE_INDEX_FILENAME).exists()
    assert (dest / "run-A" / EGRESS_MANIFEST_FILENAME).exists()
    assert result.sources_copied == 0
    assert result.sources_by_status == {}


def test_bundle_prefix_overrides_the_run_id_as_the_export_folder(tmp_path: Path):
    run_root, _ = _build_run(tmp_path)
    dest = tmp_path / "dest"
    store = RemoteStore.from_uri(str(dest))

    build_bundle(run_root, store, run_id="run-A", bundle_prefix="handoff/2026-09-26")

    assert (dest / "handoff" / "2026-09-26" / CORPUS_DIRNAME / "manifest.parquet").is_file()
    assert not (dest / "run-A").exists()
