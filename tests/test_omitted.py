"""The files a run passes over are recorded, with a reason (SMS uplift D2)."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from tests._synthetic import REGISTER_CSV
from womblex.cli._shared import NestedCorpusError, Omitted, discover_selection, select_files, select_supported
from womblex.store.omitted_output import OMITTED_FILENAME, OMITTED_SCHEMA, publish_omitted, read_omitted, write_omitted


def test_the_files_a_run_passes_over_are_named_with_a_reason():
    supported, omitted = select_files(
        ["b.csv", "a.pdf", "notes.txt", ".DS_Store", "logs/run.log", "logs/deep/x.json"], location="inbox")

    assert supported == ["a.pdf", "b.csv"]
    assert omitted == [
        Omitted(".DS_Store", "unsupported_type"), Omitted("logs/deep/x.json", "nested"),
        Omitted("logs/run.log", "nested"), Omitted("notes.txt", "unsupported_type"),
    ]
    assert select_supported(["b.csv", "notes.txt"], location="inbox") == ["b.csv"]


def test_a_supported_document_a_level_down_is_still_refused_not_omitted():
    with pytest.raises(NestedCorpusError):
        select_files(["a.pdf", "2026/b.pdf", "notes.txt"], location="inbox")


def test_omitted_is_written_with_its_schema_and_an_empty_one_is_still_a_file(tmp_path: Path):
    path = write_omitted([Omitted("notes.txt", "unsupported_type")], tmp_path)
    assert path.name == OMITTED_FILENAME
    assert read_omitted(path).schema.equals(OMITTED_SCHEMA)
    assert read_omitted(path).to_pylist() == [{"source_relpath": "notes.txt", "reason": "unsupported_type"}]

    assert read_omitted(write_omitted([], tmp_path / "other")).num_rows == 0  # "omitted nothing", not "not recorded"
    meta = pq.read_schema(str(path)).metadata
    assert meta[b"womblex.sensitivity"] == b"none"


def test_a_local_run_writes_omitted_at_its_root(tmp_path: Path):
    from tests.test_pipeline import _write_minimal_config
    from womblex.cli.pipeline import cmd_run

    inbox = tmp_path / "in"
    (inbox / "scratch").mkdir(parents=True)
    shutil.copy(REGISTER_CSV, inbox / REGISTER_CSV.name)
    (inbox / "readme.txt").write_text("x")
    (inbox / "scratch" / "tmp.bin").write_bytes(b"x")
    cfg = _write_minimal_config(tmp_path, inbox)

    rc = cmd_run(argparse.Namespace(config=cfg, resume=False, limit=None, skip=0, batch_size=None, run_id="r1"))

    assert rc == 0
    assert read_omitted(tmp_path / "out" / "r1" / OMITTED_FILENAME).to_pylist() == [
        {"source_relpath": "readme.txt", "reason": "unsupported_type"},
        {"source_relpath": "scratch/tmp.bin", "reason": "nested"},
    ]
    assert discover_selection(inbox)[1] == [Omitted("readme.txt", "unsupported_type"),
                                            Omitted("scratch/tmp.bin", "nested")]


def test_enqueue_records_the_omitted_files_under_the_ingest_prefix_in_the_store(tmp_path: Path, monkeypatch):
    import womblex.cloud.jobs as jobs_module
    from tests.test_cloud import _CapturingQueue, _enqueue_args
    from womblex.cli import cloud
    from womblex.cloud import dbos_app

    monkeypatch.setattr(jobs_module, "RunBoard", _CapturingQueue)
    monkeypatch.setattr(dbos_app, "ensure_schema", lambda _dsn: None)
    monkeypatch.setattr("womblex.cloud.worker_registry.require_served", lambda _dsn, _needed: None)
    (tmp_path / "inbox" / "2026-08").mkdir(parents=True)
    (tmp_path / "inbox" / "2026-08" / "a.pdf").write_bytes(b"%PDF-1.4\n")
    (tmp_path / "inbox" / "2026-08" / "cover.txt").write_text("x")

    assert cloud.cmd_enqueue(_enqueue_args(tmp_path, "2026-08")) == 0

    omitted = read_omitted(tmp_path / "store" / "runs" / "run-cli" / OMITTED_FILENAME).to_pylist()
    assert omitted == [{"source_relpath": "2026-08/cover.txt", "reason": "unsupported_type"}]


def test_the_api_enqueue_records_them_too(tmp_path: Path, monkeypatch):
    from tests.test_dispatch import _Board, _seed, _settings
    from womblex.cloud import dispatch

    monkeypatch.setattr("womblex.cloud.jobs.RunBoard", _Board)
    monkeypatch.setattr("womblex.cloud.dispatch.ensure_schema", lambda _dsn: None)
    monkeypatch.setattr("womblex.cloud.worker_registry.require_served", lambda _dsn, _needed: None)

    settings = _settings(tmp_path)
    _seed(settings, "a.pdf")
    (Path(str(settings.ingest_uri)) / "cover.txt").write_text("x")

    dispatch.enqueue_extraction(settings, run_id="r1")

    assert read_omitted(tmp_path / "store" / "runs" / "r1" / OMITTED_FILENAME).to_pylist() == [
        {"source_relpath": "cover.txt", "reason": "unsupported_type"}]


def test_a_record_that_cannot_be_written_is_a_warning_and_never_holds_the_run(caplog):
    class _Broken:
        def upload_file(self, *_a: object, **_k: object) -> None:
            raise OSError("store refused the write")

    with caplog.at_level(logging.WARNING):
        publish_omitted([Omitted("notes.txt", "unsupported_type")], _Broken(), "runs/r1")  # type: ignore[arg-type]

    assert "could not record the omitted files under runs/r1" in caplog.text
