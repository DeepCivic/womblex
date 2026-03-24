"""Tests for womblex.pipeline — orchestration.

Tests use real fixtures. No synthetic data.
"""

from pathlib import Path

import pytest

from womblex.config import PipelineConfig, load_config
from womblex.pipeline import BatchResult, DocumentResult, process_batch, process_file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(sample_config_path: Path) -> PipelineConfig:
    return load_config(sample_config_path)


# ---------------------------------------------------------------------------
# process_file
# ---------------------------------------------------------------------------


class TestProcessFile:
    def test_nonexistent_file_errors(self, tmp_path: Path, sample_config_path: Path) -> None:
        config = _make_config(sample_config_path)
        results = process_file(tmp_path / "missing.pdf", config)

        assert len(results) == 1
        assert results[0].status == "error"
        assert results[0].error is not None
        assert results[0].profile is None

    def test_processes_real_spreadsheet(self, spreadsheet_dir: Path, sample_config_path: Path) -> None:
        csv_path = spreadsheet_dir / "Approved-providers-au-export_20260204.csv"
        if not csv_path.exists():
            pytest.skip("CSV fixture not available")

        config = _make_config(sample_config_path)
        results = process_file(csv_path, config)

        assert len(results) >= 1
        ok = sum(1 for r in results if r.status == "completed")
        assert ok >= 1


# ---------------------------------------------------------------------------
# process_batch
# ---------------------------------------------------------------------------


class TestProcessBatch:
    def test_empty_batch(self, sample_config_path: Path) -> None:
        config = _make_config(sample_config_path)
        batch = process_batch([], config)

        assert batch.succeeded == 0
        assert batch.failed == 0
        assert len(batch.results) == 0

    def test_batch_with_missing_file(self, tmp_path: Path, sample_config_path: Path) -> None:
        config = _make_config(sample_config_path)
        missing = tmp_path / "missing.pdf"
        batch = process_batch([missing], config)

        assert batch.failed == 1
        assert batch.succeeded == 0


# ---------------------------------------------------------------------------
# DocumentResult / BatchResult
# ---------------------------------------------------------------------------


class TestDocumentResult:
    def test_default_status(self) -> None:
        r = DocumentResult(path=Path("/tmp/test.pdf"), doc_id="test")
        assert r.status == "pending"
        assert r.profile is None
        assert r.extraction is None
        assert r.error is None
        assert r.chunks == []


class TestBatchResult:
    def test_empty_batch(self) -> None:
        b = BatchResult()
        assert b.succeeded == 0
        assert b.failed == 0

    def test_counts(self) -> None:
        b = BatchResult(
            results=[
                DocumentResult(path=Path("/a.pdf"), doc_id="a", status="completed"),
                DocumentResult(path=Path("/b.pdf"), doc_id="b", status="error"),
                DocumentResult(path=Path("/c.pdf"), doc_id="c", status="completed"),
            ]
        )
        assert b.succeeded == 2
        assert b.failed == 1
