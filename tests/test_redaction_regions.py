"""Where a source withheld content is kept, region by region (SMS uplift D3)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests._pdf_builders import PdfBuilder
from tests._synthetic import NOTICE_PDF, REGISTER_CSV
from womblex.batch import process_batch
from womblex.config import DatasetConfig, PathsConfig, RedactionConfig, WomblexConfig
from womblex.ingest.pdf.types import Rect
from womblex.redact.batch import annotate_redactions_for_shards
from womblex.redact.detector import RedactionInfo
from womblex.redact.stage import RedactionReport, region_rows
from womblex.store.manifest_rows import UNREADABLE_METHOD
from womblex.store.output import read_manifest
from womblex.store.redaction_regions_output import (
    REDACTION_REGIONS_SCHEMA,
    read_redaction_regions,
    redaction_regions_path_for,
)


def _shard(shard_dir: Path, pdf_dir: Path, name: str, source_hash: str, batch: str = "0001") -> None:
    pq.write_table(pa.table({"source_hash": [source_hash], "filename": [name]}),
                   shard_dir / f"batch-{batch}._manifest.parquet")
    pq.write_table(pa.table({"source_hash": [source_hash], "elem_order": pa.array([0], type=pa.int32()),
                             "page": pa.array([0], type=pa.int32())}),
                   shard_dir / f"batch-{batch}.elements.parquet")


def test_pixels_at_the_detection_dpi_become_points_on_the_page():
    report = RedactionReport(dpi=144, page_redactions={
        2: [RedactionInfo(bbox=(144, 288, 432, 360), page=2, area_px=288 * 72)]})

    (row,) = region_rows("h", report)

    assert row["page"] == 2 and row["source_hash"] == "h"
    assert row["bbox"] == {"x": 72.0, "y": 144.0, "width": 144.0, "height": 36.0}
    assert row["area"] == 144.0 * 36.0


def test_the_batch_annotator_writes_each_region_with_its_page_and_box(tmp_path: Path):
    shards, pdfs = tmp_path / "shards", tmp_path / "pdfs"
    shards.mkdir()
    pdfs.mkdir()
    PdfBuilder(pdfs / "redacted.pdf").page(600, 400).rect(Rect(100, 100, 300, 140)).save()
    _shard(shards, pdfs, "redacted.pdf", "h1")

    annotate_redactions_for_shards(shards, pdfs)

    table = read_redaction_regions(shards / "batch-0001.parquet")
    assert table.schema.equals(REDACTION_REGIONS_SCHEMA)
    (row,) = table.to_pylist()
    assert (row["source_hash"], row["page"]) == ("h1", 0)
    assert row["bbox"] == pytest.approx({"x": 100, "y": 100, "width": 200, "height": 40}, abs=1.5)
    assert row["area"] == pytest.approx(200 * 40, rel=0.05)


def test_a_batch_with_no_redaction_still_writes_an_empty_file(tmp_path: Path):
    shards, pdfs = tmp_path / "shards", tmp_path / "pdfs"
    shards.mkdir()
    pdfs.mkdir()
    PdfBuilder(pdfs / "blank.pdf").page().save()
    _shard(shards, pdfs, "blank.pdf", "h1")

    annotate_redactions_for_shards(shards, pdfs)

    assert redaction_regions_path_for(shards / "batch-0001.parquet").exists()
    assert read_redaction_regions(shards).num_rows == 0


def _config(tmp_path: Path, *, enabled: bool) -> WomblexConfig:
    return WomblexConfig(
        dataset=DatasetConfig(name="t"),
        paths=PathsConfig(input_root=tmp_path, output_root=tmp_path, checkpoint_dir=tmp_path),
        redaction=RedactionConfig(enabled=enabled),
    )


def test_a_batch_run_with_redaction_writes_regions_keyed_by_its_manifest(tmp_path: Path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    shutil.copy(NOTICE_PDF, inbox / "notice.pdf")
    shutil.copy(REGISTER_CSV, inbox / "register.csv")
    shards = tmp_path / "documents"

    outcome = process_batch(
        [inbox / "notice.pdf", inbox / "register.csv", inbox / "vanished.pdf"], _config(tmp_path, enabled=True),
        batch_num=1, shard_dir=shards,
    )

    rows = {r["filename"]: r for r in read_manifest(outcome.shard_path).to_pylist()}
    assert rows["vanished.pdf"]["extraction_method"] == UNREADABLE_METHOD  # failed beside it, and nothing broke
    regions = read_redaction_regions(outcome.shard_path).to_pylist()
    assert regions, "the redacted notice has withheld regions"
    assert {r["source_hash"] for r in regions} == {rows["notice.pdf"]["source_hash"]}
    assert all(r["bbox"]["width"] > 0 and r["bbox"]["height"] > 0 and r["page"] >= 0 for r in regions)


def test_a_batch_run_without_redaction_writes_no_regions_file(tmp_path: Path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    shutil.copy(NOTICE_PDF, inbox / "notice.pdf")

    outcome = process_batch([inbox / "notice.pdf"], _config(tmp_path, enabled=False), batch_num=1,
                            shard_dir=tmp_path / "documents")

    assert not redaction_regions_path_for(outcome.shard_path).exists()  # a missing file means detection did not run
