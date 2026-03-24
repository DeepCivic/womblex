"""Tests for womblex.store.output -- Parquet writer."""

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from womblex.ingest.extract import (
    ExtractionMetadata,
    ExtractionResult,
    FormField,
    ImageData,
    PageResult,
    Position,
    TableData,
    TextBlock,
)
from womblex.store.output import EXTRACTION_SCHEMA, read_results, write_results


def _make_result(text: str = "Sample text", method: str = "native_narrative") -> ExtractionResult:
    """Build a minimal ExtractionResult for testing."""
    pos = Position(x=0.1, y=0.1, width=0.8, height=0.1)
    return ExtractionResult(
        pages=[PageResult(page_number=0, text=text, method="native")],
        method=method,
        tables=[TableData(headers=["A", "B"], rows=[["1", "2"]], position=pos, confidence=0.85)],
        forms=[FormField(field_name="Name", value="Test", position=pos, confidence=0.9)],
        images=[ImageData(alt_text="photo", position=pos, confidence=0.7)],
        text_blocks=[TextBlock(text=text, position=pos, block_type="paragraph", confidence=0.9)],
        metadata=ExtractionMetadata(
            extraction_strategy=method,
            confidence=0.95,
            processing_time=0.5,
            page_count=1,
            text_coverage=1.0,
        ),
    )


class TestWriteResults:
    def test_writes_single_result(self, tmp_path: Path) -> None:
        out = tmp_path / "test.parquet"
        result = _make_result()
        write_results([("doc1", "/path/doc1.pdf", result)], out)

        assert out.exists()
        table = pq.read_table(str(out))
        assert len(table) == 1
        assert table.column("document_id")[0].as_py() == "doc1"
        assert "Sample text" in table.column("text")[0].as_py()

    def test_writes_multiple_results(self, tmp_path: Path) -> None:
        out = tmp_path / "multi.parquet"
        rows = [
            ("doc1", "/a.pdf", _make_result("First")),
            ("doc2", "/b.pdf", _make_result("Second")),
        ]
        write_results(rows, out)

        table = pq.read_table(str(out))
        assert len(table) == 2

    def test_writes_empty_list(self, tmp_path: Path) -> None:
        out = tmp_path / "empty.parquet"
        write_results([], out)

        assert out.exists()
        table = pq.read_table(str(out))
        assert len(table) == 0

    def test_preserves_tables(self, tmp_path: Path) -> None:
        out = tmp_path / "tables.parquet"
        result = _make_result()
        write_results([("doc1", "/a.pdf", result)], out)

        table = pq.read_table(str(out))
        tables_col = table.column("tables")[0].as_py()
        assert len(tables_col) == 1
        assert tables_col[0]["headers"] == ["A", "B"]
        assert tables_col[0]["rows"] == [["1", "2"]]

    def test_preserves_forms(self, tmp_path: Path) -> None:
        out = tmp_path / "forms.parquet"
        result = _make_result()
        write_results([("doc1", "/a.pdf", result)], out)

        table = pq.read_table(str(out))
        forms_col = table.column("forms")[0].as_py()
        assert len(forms_col) == 1
        assert forms_col[0]["field_name"] == "Name"

    def test_preserves_metadata(self, tmp_path: Path) -> None:
        out = tmp_path / "meta.parquet"
        result = _make_result()
        write_results([("doc1", "/a.pdf", result)], out)

        table = pq.read_table(str(out))
        meta = table.column("metadata")[0].as_py()
        assert meta["extraction_strategy"] == "native_narrative"
        assert meta["confidence"] == pytest.approx(0.95)

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        out = tmp_path / "sub" / "dir" / "test.parquet"
        write_results([("doc1", "/a.pdf", _make_result())], out)
        assert out.exists()

    def test_schema_matches(self, tmp_path: Path) -> None:
        out = tmp_path / "schema.parquet"
        write_results([("doc1", "/a.pdf", _make_result())], out)

        table = pq.read_table(str(out))
        for field in EXTRACTION_SCHEMA:
            assert field.name in table.schema.names


class TestReadResults:
    def test_roundtrip(self, tmp_path: Path) -> None:
        out = tmp_path / "roundtrip.parquet"
        result = _make_result()
        write_results([("doc1", "/a.pdf", result)], out)

        table = read_results(out)
        assert len(table) == 1
        assert table.column("document_id")[0].as_py() == "doc1"


class TestResultWithoutMetadata:
    def test_handles_no_metadata(self, tmp_path: Path) -> None:
        out = tmp_path / "nometa.parquet"
        result = ExtractionResult(
            pages=[PageResult(page_number=0, text="Hello", method="native")],
            method="native_narrative",
        )
        write_results([("doc1", "/a.pdf", result)], out)

        table = pq.read_table(str(out))
        meta = table.column("metadata")[0].as_py()
        assert meta["extraction_strategy"] == "native_narrative"
        assert meta["confidence"] == 0.0
