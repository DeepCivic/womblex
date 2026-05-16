"""Tests for womblex.store.output — fixture-driven, ground-truth only.

Exercises the writer / reader / integrity check round-trip on the
budget-statement DOCX fixture. No synthetic data — every test relies
on a real fixture in womblex-development-fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from womblex.ingest.strategies_file import DocxExtractor
from womblex.store.output import (
    ELEMENT_SCHEMA,
    FORM_FIELDS_SCHEMA,
    MANIFEST_SCHEMA,
    TABLE_CELLS_SCHEMA,
    ShardVerificationError,
    _shard_paths,
    read_elements,
    read_form_fields,
    read_manifest,
    read_table_cells,
    verify_shard_persistence,
    write_results,
)

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "fixtures"
_BUDGET_DOCX = (
    _FIXTURES
    / "womblex-collection"
    / "_documents"
    / "foreign-affairs-and-trade-2025-26-portfolio-budget-statements.docx"
)


@pytest.fixture(scope="module")
def budget_statement_extraction():
    if not _BUDGET_DOCX.exists():
        pytest.skip(f"fixture not present: {_BUDGET_DOCX}")
    return DocxExtractor().extract_path(_BUDGET_DOCX)


@pytest.fixture
def written_shard(tmp_path, budget_statement_extraction):
    shard = tmp_path / "batch-0001.parquet"
    write_results(
        [("budget", str(_BUDGET_DOCX), budget_statement_extraction)],
        shard,
        collection_id="test",
    )
    return shard


class TestShardLayout:
    def test_four_sibling_files_written(self, written_shard):
        paths = _shard_paths(written_shard)
        for role, p in paths.items():
            assert p.exists(), f"{role} not written: {p}"
            assert p.stat().st_size > 0, f"{role} is empty: {p}"

    def test_schemas_round_trip(self, written_shard):
        # Each file matches the canonical schema (read_* enforces).
        assert read_elements(written_shard).schema.equals(ELEMENT_SCHEMA)
        assert read_table_cells(written_shard).schema.equals(TABLE_CELLS_SCHEMA)
        assert read_form_fields(written_shard).schema.equals(FORM_FIELDS_SCHEMA)
        assert read_manifest(written_shard).schema.equals(MANIFEST_SCHEMA)


class TestVerbatimRoundTrip:
    def test_first_table_headers_match_source(self, written_shard, budget_statement_extraction):
        # First kind='table' element's row 0 (header row) reads back verbatim
        # against the legacy derived TableData.headers view.
        e_table = read_elements(written_shard)
        kinds = e_table.column("kind").to_pylist()
        first_table_pos = kinds.index("table")
        parent_order = e_table.column("elem_order")[first_table_pos].as_py()
        src_hash = e_table.column("source_hash")[first_table_pos].as_py()

        tc = read_table_cells(written_shard)
        header_cells = [
            (r["col"], r["value"])
            for r in tc.to_pylist()
            if r["source_hash"] == src_hash
            and r["parent_elem_order"] == parent_order
            and r["row"] == 0
        ]
        header_cells.sort()
        cell_headers = [v for _, v in header_cells]
        assert cell_headers == budget_statement_extraction.tables[0].headers


class TestIntegrity:
    def test_passes_on_real_fixture(self, written_shard):
        cumulative = verify_shard_persistence(
            written_shard, expected_docs=1, prev_total_size=0,
        )
        assert cumulative > 0

    def test_detects_missing_shard(self, written_shard, tmp_path):
        paths = _shard_paths(written_shard)
        paths["elements"].unlink()
        with pytest.raises(ShardVerificationError, match="shard missing"):
            verify_shard_persistence(written_shard, 1, 0)

    def test_detects_manifest_row_mismatch(self, written_shard):
        with pytest.raises(ShardVerificationError, match="manifest row count mismatch"):
            verify_shard_persistence(written_shard, expected_docs=99, prev_total_size=0)


class TestManifestColumns:
    def test_manifest_records_source_metadata(self, written_shard):
        m = read_manifest(written_shard)
        assert m.num_rows == 1
        row = m.to_pylist()[0]
        assert row["filename"] == _BUDGET_DOCX.name
        assert row["ext"] == ".docx"
        assert row["extraction_method"] == "docx"
        assert row["elements_count"] > 0
        assert row["table_cells_count"] > 0
        assert row["status"] == "completed"
        assert row["error"] == ""
        assert row["collection_id"] == "test"
        assert len(row["source_hash"]) == 64  # sha256 hex
