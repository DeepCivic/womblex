"""Integration tests using fixtures/womblex-collection as real-world fixtures.


Exercises detection, extraction, chunking, and Parquet output against

the curated document collection. Only tests non-OCR flows since

PaddleOCR ONNX models may not be bundled in CI.
"""


from pathlib import Path

from unittest.mock import patch



import pytest

import semchunk


from womblex.config import (

    ChunkingConfig,

    DatasetConfig,

    DetectionConfig,

    PathsConfig,
    WomblexConfig,
)

from womblex.ingest.detect import DocumentType, detect_file_type

from womblex.ingest.extract import extract_text

from womblex.operations import run_extraction, run_chunking, write_batch_parquet, BatchResult

from womblex.process.chunker import (
    ChunkInput,
    chunk_batch,
    create_chunker,
    table_to_markdown,
)
from womblex.utils.availability import isaacus_available

# run_chunking sizes chunks with the Kanon-2 tokeniser (Isaacus API only) and
# skips when the API isn't configured; tests asserting it produced chunks
# require Isaacus. Direct create_chunker(callable) tests are unaffected.
requires_isaacus = pytest.mark.skipif(
    not isaacus_available(),
    reason="run_chunking needs the Kanon-2 tokeniser (isaacus SDK + ISAACUS_API_KEY)",
)


def _chunk_doc(full_text, chunker, tables=None):
    """Test-only shim mimicking the deleted chunk_document via chunk_batch."""
    ci = ChunkInput(
        source_hash="d",
        narrative=full_text or "",
        tables=[(None, table_to_markdown(t.headers, t.rows)) for t in (tables or [])],
    )
    return chunk_batch([ci], chunker).get("d", [])


FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "fixtures" / "womblex-collection"

PDF_DIR = FIXTURE_DIR / "_documents"

CSV_DIR = FIXTURE_DIR / "_spreadsheets"


_CSV_FILE = CSV_DIR / "Approved-providers-au-export_20260204.csv"

# Every test here reads a real document from the collection. Skip cleanly when
# the fixtures repo is not cloned (e.g. CI). See THIRD_PARTY_DATA.md.
pytestmark = pytest.mark.skipif(
    not FIXTURE_DIR.exists(),
    reason="womblex-benchmark not cloned (see THIRD_PARTY_DATA.md)",
)

_REDACTED_PDF = PDF_DIR / "00768-213A-270825-Throsby-Out-of-School-Care-Administrative-Decision-Other-Notice-and-Direction_Redacted.pdf"



def _word_token_counter(text: str) -> int:

    """Simple word-count token counter for integration tests."""

    return len(text.split())



# Fast-tier page bound. Native extraction OCRs embedded image regions and
# detects tables per page (~0.8s/page on these government PDFs), so the full
# 406-page Auditor-General report takes minutes — a benchmark-scale input, not
# a fast-tier unit. Its vendored `-First-30-Pages` truncation is the fast-tier
# proxy; the full report is exercised by the benchmark accuracy suite.
_FAST_TIER_MAX_PAGES = 60


def _native_pdfs() -> list[Path]:

    """Native-typed PDFs small enough for the fast tier (see _FAST_TIER_MAX_PAGES)."""

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

        if profile.doc_type in native_types and profile.page_count <= _FAST_TIER_MAX_PAGES:

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
        # The new output writes four sibling shards. Verify the manifest
        # has one row per source and the elements shard has at least one
        # sheet_cell per source.
        from womblex.store.output import read_elements, read_manifest

        config = WomblexConfig(
            dataset=DatasetConfig(name="csv_test"),
            paths=PathsConfig(

                input_root=CSV_DIR,

                output_root=tmp_path / "out",

                checkpoint_dir=tmp_path / "ckpt",

            ),

            chunking=ChunkingConfig(enabled=False),
        )

        results = run_extraction([_CSV_FILE], config)

        batch = BatchResult(results=results)

        assert batch.failed == 0
        assert batch.succeeded > 0

        out = tmp_path / "csv.parquet"
        write_batch_parquet(batch, out)

        manifest = read_manifest(out)
        assert manifest.num_rows == batch.succeeded

        elements = read_elements(out)
        kinds = elements.column("kind").to_pylist()
        assert "sheet_cell" in kinds
        assert "sheet_meta" in kinds



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


        config = WomblexConfig(
            dataset=DatasetConfig(name="integration_test"),
            paths=PathsConfig(

                input_root=PDF_DIR,

                output_root=tmp_path / "out",

                checkpoint_dir=tmp_path / "ckpt",

            ),

            chunking=ChunkingConfig(enabled=False),
        )


        results = run_extraction(pdfs, config)

        batch = BatchResult(results=results)

        assert batch.succeeded == len(pdfs)

        assert batch.failed == 0


        # The new output writes four sibling shards (elements / table_cells /
        # form_fields / manifest). Verify the manifest carries one row per
        # PDF with sane extraction metadata; the elements shard has rows.
        from womblex.store.output import _shard_paths, read_elements, read_manifest

        out = tmp_path / "extraction.parquet"
        write_batch_parquet(batch, out)

        paths = _shard_paths(out)
        for role, p in paths.items():
            assert p.exists(), f"{role} shard not written: {p}"

        manifest = read_manifest(out)
        assert manifest.num_rows == len(pdfs)
        statuses = manifest.column("status").to_pylist()
        assert all(s == "completed" for s in statuses)
        methods = manifest.column("extraction_method").to_pylist()
        # Method is the DocumentType enum value (e.g. native_narrative,
        # native_with_structured, structured, hybrid, scanned_machinewritten).
        assert all(m for m in methods)

        elements = read_elements(out)
        assert elements.num_rows > 0




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

        chunks = _chunk_doc(full_text, chunker, tables=all_tables)

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


    @requires_isaacus
    def test_csv_pipeline_with_chunking(self, tmp_path: Path) -> None:

        """Full run: CSV → detect → extract → chunk."""

        config = WomblexConfig(
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

        with patch("womblex.operations.chunk.create_chunker", return_value=word_chunker):

            doc_results = run_extraction([_CSV_FILE], config)

            doc_results = run_chunking(doc_results, config)


        assert all(r.status == "completed" for r in doc_results)

        all_chunks = [c for r in doc_results for c in r.chunks]

        assert len(all_chunks) > 0

        table_chunks = [c for c in all_chunks if c.content_type == "table"]

        assert len(table_chunks) >= 1


    def test_csv_chunking_disabled(self, tmp_path: Path) -> None:

        """When chunking.enabled=False, no chunks are produced."""

        config = WomblexConfig(
            dataset=DatasetConfig(name="csv_no_chunk"),
            paths=PathsConfig(

                input_root=CSV_DIR,

                output_root=tmp_path / "out",

                checkpoint_dir=tmp_path / "ckpt",

            ),

            chunking=ChunkingConfig(enabled=False),
        )

        word_chunker = semchunk.chunkerify(_word_token_counter, chunk_size=100)

        with patch("womblex.operations.chunk.create_chunker", return_value=word_chunker):

            doc_results = run_extraction([_CSV_FILE], config)


        assert all(r.status == "completed" for r in doc_results)

        assert all(len(r.chunks) == 0 for r in doc_results)



# ---------------------------------------------------------------------------

# Chunking integration — Redacted PDF

# ---------------------------------------------------------------------------



@pytest.mark.skipif(
    not _REDACTED_PDF.exists(),
    reason="Throsby 213A redacted PDF not vendored (research-use only; lives in the external womblex-development-fixtures repo — see THIRD_PARTY_DATA.md)",
)
class TestRedactedPDFChunkingIntegration:

    """End-to-end: redacted PDF detection → extraction → chunking."""

    # The Throsby redacted PDF is a womblex-benchmark fixture, not part of the
    # vendored minimal set (the womblex-collection dir exists via _spreadsheets,
    # so the module-level guard passes — this needs the specific file). Skip
    # cleanly on a bare checkout. See THIRD_PARTY_DATA.md.
    pytestmark = pytest.mark.skipif(
        not _REDACTED_PDF.exists(),
        reason="redacted PDF fixture (Throsby) needs womblex-benchmark (see THIRD_PARTY_DATA.md)",
    )

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

        chunks = _chunk_doc(result.full_text, chunker, tables=result.tables)


        assert len(chunks) > 0


        indices = [c.chunk_index for c in chunks]

        assert indices == list(range(len(chunks)))


        for chunk in chunks:

            assert chunk.start_char >= 0

            assert chunk.end_char >= chunk.start_char


    @requires_isaacus
    def test_redacted_pdf_pipeline_produces_chunks(self, tmp_path: Path) -> None:

        """Full pipeline run on a redacted PDF produces chunks."""

        config = WomblexConfig(
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

        with patch("womblex.operations.chunk.create_chunker", return_value=word_chunker):

            doc_results = run_extraction([_REDACTED_PDF], config)

            doc_results = run_chunking(doc_results, config)


        assert len(doc_results) == 1

        doc_result = doc_results[0]

        assert doc_result.status == "completed"

        assert len(doc_result.chunks) > 0


        for chunk in doc_result.chunks:

            assert len(chunk.text.strip()) > 0


    @requires_isaacus
    def test_redacted_pdf_chunk_tables_flag(self, tmp_path: Path) -> None:

        """When chunk_tables=False, only narrative chunks are produced."""

        config = WomblexConfig(
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

        with patch("womblex.operations.chunk.create_chunker", return_value=word_chunker):

            doc_results = run_extraction([_REDACTED_PDF], config)

            doc_results = run_chunking(doc_results, config)


        assert len(doc_results) == 1

        doc_result = doc_results[0]

        assert doc_result.status == "completed"

        for chunk in doc_result.chunks:

            assert chunk.content_type == "narrative"

