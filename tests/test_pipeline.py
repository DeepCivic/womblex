"""Tests for womblex.operations — independent operations.


Tests use real fixtures. No synthetic data.
"""


from pathlib import Path


import pytest


from womblex.config import WomblexConfig, load_config

from womblex.operations import BatchResult, DocumentResult, run_extraction, run_chunking, write_batch_parquet



# ---------------------------------------------------------------------------

# Helpers

# ---------------------------------------------------------------------------



def _make_config(sample_config_path: Path) -> WomblexConfig:
    return load_config(sample_config_path)



# ---------------------------------------------------------------------------

# run_extraction

# ---------------------------------------------------------------------------



class TestRunExtraction:

    def test_nonexistent_file_errors(self, tmp_path: Path, sample_config_path: Path) -> None:

        config = _make_config(sample_config_path)

        results = run_extraction([tmp_path / "missing.pdf"], config)


        assert len(results) == 1

        assert results[0].status == "error"

        assert results[0].error is not None

        assert results[0].profile is None


    def test_extracts_real_spreadsheet(self, spreadsheet_dir: Path, sample_config_path: Path) -> None:

        csv_path = spreadsheet_dir / "Approved-providers-au-export_20260204.csv"

        if not csv_path.exists():

            pytest.skip("CSV fixture not available")


        config = _make_config(sample_config_path)

        results = run_extraction([csv_path], config)


        assert len(results) >= 1

        ok = sum(1 for r in results if r.status == "completed")

        assert ok >= 1


    def test_empty_list(self, sample_config_path: Path) -> None:

        config = _make_config(sample_config_path)

        results = run_extraction([], config)

        assert len(results) == 0



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



# ---------------------------------------------------------------------------

# Composition: extract then chunk

# ---------------------------------------------------------------------------


from womblex.config import ChunkingConfig, DatasetConfig, PathsConfig


FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "fixtures" / "womblex-collection"

_CSV_FILE = FIXTURE_DIR / "_spreadsheets" / "Approved-providers-au-export_20260204.csv"



class TestComposition:

    """Operations compose correctly when called in sequence."""


    def test_extract_then_chunk(self) -> None:

        if not _CSV_FILE.exists():

            pytest.skip("CSV fixture not available")

        config = WomblexConfig(
            dataset=DatasetConfig(name="t"),
            paths=PathsConfig(input_root=Path("."), output_root=Path("."), checkpoint_dir=Path(".")),

            chunking=ChunkingConfig(enabled=True, chunk_size=480),
        )

        results = run_extraction([_CSV_FILE], config)

        assert any(r.status == "completed" for r in results)


        results = run_chunking(results, config)

        has_chunks = any(len(r.chunks) > 0 for r in results)

        assert has_chunks


    def test_extract_only(self) -> None:

        if not _CSV_FILE.exists():

            pytest.skip("CSV fixture not available")

        config = WomblexConfig(
            dataset=DatasetConfig(name="t"),
            paths=PathsConfig(input_root=Path("."), output_root=Path("."), checkpoint_dir=Path(".")),
            chunking=ChunkingConfig(enabled=False),
        )

        results = run_extraction([_CSV_FILE], config)

        assert any(r.status == "completed" for r in results)

        # No chunking called — no chunks.

        assert all(len(r.chunks) == 0 for r in results)

