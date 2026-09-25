"""Tests for the egress bundle's source_index.parquet (store/egress_output.py)."""

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from womblex.store.egress_output import (
    SOURCE_INDEX_FILENAME,
    SOURCE_INDEX_SCHEMA,
    read_source_index,
    write_source_index,
)

_ROWS = [
    {
        "source_hash": "abc123",
        "doc_id": "doc-1",
        "filename": "report.pdf",
        "ext": ".pdf",
        "raw_key": "abc123.pdf",
        "status": "resolved",
    },
    {
        "source_hash": "def456",
        "doc_id": "doc-2",
        "filename": "register.csv",
        "ext": ".csv",
        "raw_key": None,
        "status": "not_found",
    },
]


def test_write_and_read_roundtrip(tmp_path: Path):
    target = write_source_index(_ROWS, tmp_path)
    assert target == tmp_path / SOURCE_INDEX_FILENAME
    assert target.exists()

    table = read_source_index(tmp_path)
    assert table.schema.equals(SOURCE_INDEX_SCHEMA)
    rows = {r["doc_id"]: r for r in table.to_pylist()}
    assert rows["doc-1"]["raw_key"] == "abc123.pdf"
    assert rows["doc-1"]["status"] == "resolved"
    assert rows["doc-2"]["raw_key"] is None
    assert rows["doc-2"]["status"] == "not_found"


def test_read_source_index_from_file_path(tmp_path: Path):
    target = write_source_index(_ROWS, tmp_path)
    table = read_source_index(target)
    assert table.num_rows == 2


def test_duplicate_source_hash_keeps_both_rows(tmp_path: Path):
    """Two documents sharing one hash (same bytes, two names) each get a row,
    both naming the same raw_key — the index is not deduplicated by hash."""
    rows = [
        {
            "source_hash": "shared",
            "doc_id": "doc-a",
            "filename": "a.pdf",
            "ext": ".pdf",
            "raw_key": "shared.pdf",
            "status": "resolved",
        },
        {
            "source_hash": "shared",
            "doc_id": "doc-b",
            "filename": "b.pdf",
            "ext": ".pdf",
            "raw_key": "shared.pdf",
            "status": "resolved",
        },
    ]
    table = read_source_index(write_source_index(rows, tmp_path))
    assert table.num_rows == 2
    assert {r["raw_key"] for r in table.to_pylist()} == {"shared.pdf"}


def test_empty_rows_is_schema_correct(tmp_path: Path):
    table = read_source_index(write_source_index([], tmp_path))
    assert table.num_rows == 0
    assert table.schema.equals(SOURCE_INDEX_SCHEMA)


def test_write_creates_missing_bundle_dir(tmp_path: Path):
    bundle_dir = tmp_path / "run-123"
    write_source_index(_ROWS, bundle_dir)
    assert (bundle_dir / SOURCE_INDEX_FILENAME).exists()


def test_read_rejects_missing_columns(tmp_path: Path):
    import pyarrow as pa

    bad_schema = pa.schema([("source_hash", pa.string())])
    target = tmp_path / SOURCE_INDEX_FILENAME
    pq.write_table(pa.table({"source_hash": ["x"]}, schema=bad_schema), str(target))
    with pytest.raises(ValueError, match="missing columns"):
        read_source_index(target)
