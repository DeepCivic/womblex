"""Integration tests using fixtures/womblex-collection as real-world fixtures.

Exercises detection, extraction, chunking, and Parquet output against
the curated document collection. Only tests non-OCR flows since
PaddleOCR ONNX models may not be bundled in CI.
"""

from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq
import semchunk

from womblex.config import (
    ChunkingConfig,
    DatasetConfig,
    DetectionConfig,
    PathsConfig,
    PipelineConfig,
)
from womblex.ingest.detect import DocumentType, detect_file_type
from womblex.ingest.extract import extract_text
from womblex.pipeline import process_batch, process_file, write_batch_parquet
from womblex.process.chunker import chunk_document, create_chunker, table_to_markdown

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "fixtures" / "womblex-collection"
PDF_DIR = FIXTURE_DIR / "_documents"
CSV_DIR = FIXTURE_DIR / "_spreadsheets"

_CSV_FILE = CSV_DIR / "Approved-providers-au-export_20260204.csv"
_REDACTED_PDF = PDF_DIR / "00768-213A-270825-Throsby-Out-of-School-Care-Administrative-Decision-Other-Notice-and-Direction_Redacted.pdf"


def _word_token_counter(text: str) -> int:
    """Simple word-count token counter for integration tests."""
    return len(text.split())


def _native_pdfs() -> list[Path]:
    """Return PDFs detected as native types (no OCR required)."""
    config = DetectionConfig()
    native_types = {
        DocumentType.NATIVE_NARRATIVE,
        DocumentType.NATIVE_WITH_STRUCTURED,
        DocumentType.STRUCTURED,
    }
    found = []
    for f in sorted(PDF_DIR.iterdir()):
        if f.suffix.lower() != ".pdf":
            continue
        profile = detect_file_type(f, config)
        if profile.doc_type in native_types:
            found.append(f)
    return found


class TestRealDocumentDetection:
    def test_all_pdfs_detected(self) -> None:
        config = DetectionConfig()
        pdfs = [f for f in PDF_DIR.iterdir() if f.suffix.lower() == ".pdf"]
        for pdf in pdfs:
            profile = detect_file_type(pdf, config)
            assert profile.doc_type != DocumentType.UNKNOWN, f"{pdf.name} classified as UNKNOWN"
            assert profile.page_count > 0

    def test_csv_detected_as_spreadsheet(self) -> None:
        profile = detect_file_type(_CSV_FILE, DetectionConfig())
        assert profile.doc_type == DocumentType.SPREADSHEET

    def test_docx_detected(self) -> None:
        docx_files = list(PDF_DIR.glob("*.docx"))
        assert docx_files, "no DOCX fixture in _documents"
        profile = detect_file_type(docx_files[0], DetectionConfig())
        assert profile.doc_type == DocumentType.DOCX


class TestRealSpreadsheetExtraction:
    def test_csv_extracts_table(self) -> None:
        profile = detect_file_type(_CSV_FILE, DetectionConfig())
        results = extract_text(_CSV_FILE, profile)

        assert len(results) >= 1
        for r in results:
            assert r.error is None, f"CSV extraction error: {r.error}"
            assert r.method == "spreadsheet"
            assert r.metadata is not None
            assert r.metadata.extraction_strategy == "spreadsheet"

    def test_csv_parquet_roundtrip(self, tmp_path: Path) -> None:
        config = PipelineConfig(
            dataset=DatasetConfig(name="csv_test"),
            paths=PathsConfig(
                input_root=CSV_DIR,
                output_root=tmp_path / "out",
                checkpoint_dir=tmp_path / "ckpt",
            ),
            chunking=ChunkingConfig(enabled=False),
        )
        batch = process_batch([_CSV_FILE], config)
        assert batch.failed == 0
        assert batch.succeeded > 0

        out = tmp_path / "csv.parquet"
        write_batch_parquet(batch, out)
        table = pq.read_table(str(out))
        assert len(table) == batch.succeeded
        tables_col = table.column("tables")[0].as_py()
        assert len(tables_col) >= 1
        assert len(tables_col[0]["headers"]) > 0


class TestRealDocumentExtraction:
    def test_native_pdfs_extract_text(self) -> None:
        pdfs = _native_pdfs()
        assert len(pdfs) > 0, "No native PDFs found"

        for pdf in pdfs:
            profile = detect_file_type(pdf, DetectionConfig())
            results = extract_text(pdf, profile)
            result = results[0]

            assert result.error is None, f"{pdf.name}: {result.error}"
            assert len(result.full_text) > 0, f"{pdf.name}: no text extracted"
            assert result.metadata is not None
            assert result.metadata.processing_time >= 0
            assert result.metadata.page_count > 0

    def test_native_pdfs_produce_text_blocks(self) -> None:
        pdfs = _native_pdfs()
        for pdf in pdfs:
            profile = detect_file_type(pdf, DetectionConfig())
            result = extract_text(pdf, profile)[0]
            assert len(result.text_blocks) > 0, f"{pdf.name}: no text blocks"
            for block in result.text_blocks:
                assert 0 <= block.position.x <= 1.1  # small float tolerance
                assert block.confidence > 0



class TestRealDocumentParquet:
    def test_batch_to_parquet_roundtrip(self, tmp_path: Path) -> None:
        pdfs = _native_pdfs()[:3]  # Limit to 3 for speed
        assert len(pdfs) > 0

        config = PipelineConfig(
            dataset=DatasetConfig(name="integration_test"),
            paths=PathsConfig(
                input_root=PDF_DIR,
                output_root=tmp_path / "out",
                checkpoint_dir=tmp_path / "ckpt",
            ),
            chunking=ChunkingConfig(enabled=False),
        )

        batch = process_batch(pdfs, config)
        assert batch.succeeded == len(pdfs)
        assert batch.failed == 0

        out = tmp_path / "extraction.parquet"
        write_batch_parquet(batch, out)
        assert out.exists()

        table = pq.read_table(str(out))
        assert len(table) == len(pdfs)

        for i in range(len(table)):
            meta = table.column("metadata")[i].as_py()
            assert meta["extraction_strategy"] in (
                "native_narrative",
                "native_with_structured",
                "structured",
            )
            assert meta["confidence"] > 0
            assert meta["page_count"] > 0
            assert len(table.column("text")[i].as_py()) > 0



# ---------------------------------------------------------------------------
# Chunking integration — CSV
# ---------------------------------------------------------------------------


class TestCSVChunkingIntegration:
    """End-to-end: CSV detection → extraction → chunking."""

    def test_csv_extracts_and_chunks_tables(self) -> None:
        profile = detect_file_type(_CSV_FILE, DetectionConfig())
        assert profile.doc_type == DocumentType.SPREADSHEET

        results = extract_text(_CSV_FILE, profile)
        assert len(results) >= 1
        for r in results:
            assert r.error is None

        all_tables = [t for r in results for t in r.tables]
        assert len(all_tables) >= 1

        chunker = create_chunker(tokenizer=_word_token_counter, chunk_size=100)
        full_text = "\n\n".join(r.full_text for r in results if r.full_text.strip())
        chunks = chunk_document(full_text, chunker, tables=all_tables)
        assert len(chunks) > 0

    def test_csv_table_to_markdown_roundtrip(self) -> None:
        profile = detect_file_type(_CSV_FILE, DetectionConfig())
        results = extract_text(_CSV_FILE, profile)
        all_tables = [t for r in results for t in r.tables]
        assert len(all_tables) >= 1
        tbl = all_tables[0]

        md = table_to_markdown(tbl.headers, tbl.rows)
        assert len(md) > 0

        for hdr in tbl.headers:
            assert hdr in md, f"Header '{hdr}' missing from markdown"

        lines = md.strip().split("\n")
        assert len(lines) >= 2 + min(len(tbl.rows), 10)

    def test_csv_pipeline_with_chunking(self, tmp_path: Path) -> None:
        """Full pipeline: CSV → detect → extract → chunk via process_file."""
        config = PipelineConfig(
            dataset=DatasetConfig(name="csv_chunk_test"),
            paths=PathsConfig(
                input_root=CSV_DIR,
                output_root=tmp_path / "out",
                checkpoint_dir=tmp_path / "ckpt",
            ),
            chunking=ChunkingConfig(
                tokenizer="not-used",
                chunk_size=100,
                enabled=True,
                chunk_tables=True,
            ),
        )
        word_chunker = semchunk.chunkerify(_word_token_counter, chunk_size=100)
        with patch("womblex.pipeline.create_chunker", return_value=word_chunker):
            doc_results = process_file(_CSV_FILE, config)

        assert all(r.status == "completed" for r in doc_results)
        all_chunks = [c for r in doc_results for c in r.chunks]
        assert len(all_chunks) > 0
        table_chunks = [c for c in all_chunks if c.content_type == "table"]
        assert len(table_chunks) >= 1

    def test_csv_chunking_disabled(self, tmp_path: Path) -> None:
        """When chunking.enabled=False, no chunks are produced."""
        config = PipelineConfig(
            dataset=DatasetConfig(name="csv_no_chunk"),
            paths=PathsConfig(
                input_root=CSV_DIR,
                output_root=tmp_path / "out",
                checkpoint_dir=tmp_path / "ckpt",
            ),
            chunking=ChunkingConfig(enabled=False),
        )
        word_chunker = semchunk.chunkerify(_word_token_counter, chunk_size=100)
        with patch("womblex.pipeline.create_chunker", return_value=word_chunker):
            doc_results = process_file(_CSV_FILE, config)

        assert all(r.status == "completed" for r in doc_results)
        assert all(len(r.chunks) == 0 for r in doc_results)


# ---------------------------------------------------------------------------
# Chunking integration — Redacted PDF
# ---------------------------------------------------------------------------


class TestRedactedPDFChunkingIntegration:
    """End-to-end: redacted PDF detection → extraction → chunking."""

    def test_redacted_pdf_extracts_text(self) -> None:
        profile = detect_file_type(_REDACTED_PDF, DetectionConfig())
        assert profile.page_count > 0

        result = extract_text(_REDACTED_PDF, profile)[0]
        assert result.error is None
        assert len(result.full_text) > 0

    def test_redacted_pdf_chunks_have_valid_offsets(self) -> None:
        profile = detect_file_type(_REDACTED_PDF, DetectionConfig())
        result = extract_text(_REDACTED_PDF, profile)[0]

        chunker = create_chunker(tokenizer=_word_token_counter, chunk_size=80)
        chunks = chunk_document(result.full_text, chunker, tables=result.tables)

        assert len(chunks) > 0

        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

        for chunk in chunks:
            assert chunk.start_char >= 0
            assert chunk.end_char >= chunk.start_char

    def test_redacted_pdf_pipeline_produces_chunks(self, tmp_path: Path) -> None:
        """Full pipeline run on a redacted PDF produces chunks."""
        config = PipelineConfig(
            dataset=DatasetConfig(name="redacted_chunk_test"),
            paths=PathsConfig(
                input_root=PDF_DIR,
                output_root=tmp_path / "out",
                checkpoint_dir=tmp_path / "ckpt",
            ),
            chunking=ChunkingConfig(
                tokenizer="not-used",
                chunk_size=80,
                enabled=True,
                chunk_tables=True,
            ),
        )
        word_chunker = semchunk.chunkerify(_word_token_counter, chunk_size=80)
        with patch("womblex.pipeline.create_chunker", return_value=word_chunker):
            doc_results = process_file(_REDACTED_PDF, config)

        assert len(doc_results) == 1
        doc_result = doc_results[0]
        assert doc_result.status == "completed"
        assert len(doc_result.chunks) > 0

        for chunk in doc_result.chunks:
            assert len(chunk.text.strip()) > 0

    def test_redacted_pdf_chunk_tables_flag(self, tmp_path: Path) -> None:
        """When chunk_tables=False, only narrative chunks are produced."""
        config = PipelineConfig(
            dataset=DatasetConfig(name="redacted_no_tables"),
            paths=PathsConfig(
                input_root=PDF_DIR,
                output_root=tmp_path / "out",
                checkpoint_dir=tmp_path / "ckpt",
            ),
            chunking=ChunkingConfig(
                tokenizer="not-used",
                chunk_size=80,
                enabled=True,
                chunk_tables=False,
            ),
        )
        word_chunker = semchunk.chunkerify(_word_token_counter, chunk_size=80)
        with patch("womblex.pipeline.create_chunker", return_value=word_chunker):
            doc_results = process_file(_REDACTED_PDF, config)

        assert len(doc_results) == 1
        doc_result = doc_results[0]
        assert doc_result.status == "completed"
        for chunk in doc_result.chunks:
            assert chunk.content_type == "narrative"
