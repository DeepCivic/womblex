"""Tests for womblex.ingest.extract — extraction strategies.

Tests use real fixtures from fixtures/. OCR-dependent extractors are
tested only where PaddleOCR ONNX models are available.
"""

from pathlib import Path

import pytest

from womblex.ingest.detect import DocumentProfile, DocumentType
from womblex.ingest.extract import (
    ExtractionMetadata,
    ExtractionResult,
    FormField,
    ImageData,
    PageResult,
    Position,
    TableData,
    TextBlock,
    _count_blocks_in_bbox,
    _find_native_tables,
    _normalise_bbox,
    get_extractor,
)
from womblex.ingest.strategies import (
    DocxExtractor,
    ImageExtractor,
    TextExtractor,
)
from womblex.ingest.spreadsheet import SpreadsheetExtractor


# ---------------------------------------------------------------------------
# Data model tests
# ---------------------------------------------------------------------------


class TestPosition:
    def test_fields(self) -> None:
        p = Position(x=0.1, y=0.2, width=0.5, height=0.3)
        assert p.x == 0.1
        assert p.width == 0.5

    def test_normalise_bbox(self) -> None:
        pos = _normalise_bbox((100, 200, 400, 600), 800, 1000)
        assert pos.x == pytest.approx(0.125)
        assert pos.y == pytest.approx(0.2)
        assert pos.width == pytest.approx(0.375)
        assert pos.height == pytest.approx(0.4)


class TestExtractionResult:
    def test_full_text_concatenation(self) -> None:
        result = ExtractionResult(
            pages=[
                PageResult(page_number=0, text="Hello", method="native"),
                PageResult(page_number=1, text="World", method="native"),
            ],
            method="native",
        )
        assert result.full_text == "Hello\n\nWorld"

    def test_full_text_skips_empty(self) -> None:
        result = ExtractionResult(
            pages=[
                PageResult(page_number=0, text="Hello", method="native"),
                PageResult(page_number=1, text="", method="native"),
                PageResult(page_number=2, text="End", method="native"),
            ],
            method="native",
        )
        assert result.full_text == "Hello\n\nEnd"

    def test_page_count(self) -> None:
        result = ExtractionResult(
            pages=[PageResult(page_number=i, text=f"p{i}", method="native") for i in range(5)],
            method="native",
        )
        assert result.page_count == 5

    def test_empty_result(self) -> None:
        result = ExtractionResult()
        assert result.full_text == ""
        assert result.page_count == 0
        assert result.error is None
        assert result.tables == []
        assert result.forms == []
        assert result.images == []
        assert result.text_blocks == []

    def test_structured_fields(self) -> None:
        # The legacy structured fields (tables/forms/images/text_blocks) are
        # now read-only derived views over ExtractionResult.elements. Build
        # an element stream and confirm each view projects correctly.
        from womblex.ingest.elements import Cell, Element, FieldEntry

        pos = Position(x=0.0, y=0.0, width=1.0, height=1.0)
        elements = [
            Element(
                order=0, kind="paragraph", extractor="native_text",
                bbox=pos, text="Test", confidence=0.9,
            ),
            Element(
                order=1, kind="table", extractor="native_text",
                bbox=pos, confidence=0.8,
                cells=[
                    Cell(row=0, col=0, value="A"),
                    Cell(row=0, col=1, value="B"),
                    Cell(row=1, col=0, value="1"),
                    Cell(row=1, col=1, value="2"),
                ],
                header_rows=[0],
            ),
            Element(
                order=2, kind="form", extractor="form_acroform",
                bbox=pos, confidence=0.9,
                fields=[FieldEntry(name="Name", value="Alice")],
            ),
            Element(
                order=3, kind="image", extractor="figure_image",
                bbox=pos, alt_text="photo", confidence=0.7,
            ),
        ]
        result = ExtractionResult(
            pages=[PageResult(page_number=0, text="Test", method="native")],
            elements=elements,
            method="native_with_structured",
        )
        assert len(result.tables) == 1
        assert result.tables[0].headers == ["A", "B"]
        assert len(result.forms) == 1
        assert result.forms[0].field_name == "Name"
        assert len(result.images) == 1
        assert len(result.text_blocks) == 1


class TestFindNativeTablesGate:
    """Cross-classifier gate inside ``_find_native_tables``.

    Real tables decompose into ≥1 PyMuPDF dict-block per row; prose-as-
    table over-claims rows by carving sub-block whitespace into pseudo-
    rows. The gate rejects candidates where block count < row count.
    """

    def test_count_blocks_in_bbox_uses_block_centre(self) -> None:
        import fitz

        doc = fitz.open()
        page = doc.new_page(width=400, height=600)
        page.insert_text((50, 50), "alpha")
        page.insert_text((50, 200), "beta")
        page.insert_text((50, 500), "gamma")

        bbox_top = fitz.Rect(0, 0, 400, 300)
        assert _count_blocks_in_bbox(page, bbox_top) == 2  # alpha, beta

        bbox_bottom = fitz.Rect(0, 400, 400, 600)
        assert _count_blocks_in_bbox(page, bbox_bottom) == 1  # gamma

        bbox_none = fitz.Rect(0, 300, 400, 400)
        assert _count_blocks_in_bbox(page, bbox_none) == 0

        doc.close()

    def test_gate_rejects_prose_as_table(self) -> None:
        import fitz

        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        # One paragraph block of prose with consistent left-indent. PyMuPDF's
        # text-strategy `find_tables` will read the whitespace pattern as
        # columnar and over-claim many rows; the gate should reject because
        # `n_blocks_in_bbox` (≈1) is much smaller than the claimed row count.
        prose = (
            "1. As you are aware, the Authority has issued this Notice.\n"
            "2. The Provider must comply with the requirements set out below.\n"
            "3. The Authority will continue to monitor compliance.\n"
            "4. Should you have any questions, contact the Director.\n"
            "5. This Notice takes effect immediately upon receipt.\n"
        )
        page.insert_textbox(fitz.Rect(50, 50, 545, 800), prose, fontsize=11)

        tables = _find_native_tables(page)
        # Any text-strategy hit on this page would be over-firing; the gate
        # should leave us with no tables.
        assert tables == []

        doc.close()


class TestExtractionMetadata:
    def test_fields(self) -> None:
        m = ExtractionMetadata(
            extraction_strategy="native_narrative",
            confidence=0.95,
            processing_time=1.2,
            page_count=3,
            text_coverage=0.9,
        )
        assert m.extraction_strategy == "native_narrative"
        assert m.preprocessing_steps == []
        assert m.content_mix == {}


# ---------------------------------------------------------------------------
# get_extractor
# ---------------------------------------------------------------------------


class TestGetExtractor:
    """`get_extractor` is the legacy non-PDF dispatch. Native and scanned PDFs
    route through `extract_pdf_with_plan` (orchestrator) — see
    test_orchestrator-style assertions in test_pipeline.py / test_integration.py
    for that path."""

    def _make_profile(self, doc_type: DocumentType) -> DocumentProfile:
        return DocumentProfile(
            doc_type=doc_type,
            page_count=1,
            has_text_layer=True,
            text_coverage=1.0,
            has_images=False,
            has_tables=False,
            has_handwriting_signals=False,
            ocr_confidence=None,
            glyph_regularity=None,
            stroke_consistency=None,
            confidence=0.9,
        )

    def test_image_returns_correct_extractor(self) -> None:
        ext = get_extractor(self._make_profile(DocumentType.IMAGE))
        assert isinstance(ext, ImageExtractor)

    def test_spreadsheet_returns_correct_extractor(self) -> None:
        ext = get_extractor(self._make_profile(DocumentType.SPREADSHEET))
        assert isinstance(ext, SpreadsheetExtractor)

    def test_docx_returns_correct_extractor(self) -> None:
        ext = get_extractor(self._make_profile(DocumentType.DOCX))
        assert isinstance(ext, DocxExtractor)

    def test_text_returns_correct_extractor(self) -> None:
        ext = get_extractor(self._make_profile(DocumentType.TEXT))
        assert isinstance(ext, TextExtractor)

    def test_pdf_type_raises(self) -> None:
        with pytest.raises(ValueError, match="only handles non-PDF"):
            get_extractor(self._make_profile(DocumentType.NATIVE_NARRATIVE))


# ---------------------------------------------------------------------------
# SpreadsheetExtractor (uses real fixture spreadsheets)
# ---------------------------------------------------------------------------


class TestSpreadsheetExtractor:
    """One ExtractionResult per workbook with cell-grained elements.

    The previous one-result-per-row shape was removed; a workbook now
    yields a single result whose ``elements`` are sheet_meta + sheet_cell
    in workbook order.
    """

    def test_extracts_real_csv(self, spreadsheet_dir: Path) -> None:
        csv_path = spreadsheet_dir / "Approved-providers-au-export_20260204.csv"
        if not csv_path.exists():
            pytest.skip("CSV fixture not available")

        ext = SpreadsheetExtractor()
        result = ext.extract_path(csv_path)

        assert result.method == "spreadsheet"
        assert result.error is None
        assert result.metadata is not None
        kinds = [e.kind for e in result.elements]
        assert kinds.count("sheet_meta") >= 1
        assert kinds.count("sheet_cell") >= 1

    def test_extracts_real_xlsx(self, spreadsheet_dir: Path) -> None:
        xlsx_path = spreadsheet_dir / "mso-statistics-sept-qtr-2025.xlsx"
        if not xlsx_path.exists():
            pytest.skip("Excel fixture not available")

        ext = SpreadsheetExtractor()
        result = ext.extract_path(xlsx_path)

        assert result.method == "spreadsheet"
        assert result.metadata is not None
        # Multi-sheet workbook: one sheet_meta per sheet.
        n_sheet_meta = sum(1 for e in result.elements if e.kind == "sheet_meta")
        assert n_sheet_meta >= 1

    def test_handles_missing_file(self, tmp_path: Path) -> None:
        ext = SpreadsheetExtractor()
        result = ext.extract_path(tmp_path / "missing.csv")
        assert result.error is not None
        assert result.elements == []


# ---------------------------------------------------------------------------
# DocxExtractor
# ---------------------------------------------------------------------------


class TestDocxExtractor:
    def test_handles_missing_docx_library(self, tmp_path: Path) -> None:
        ext = DocxExtractor()
        result = ext.extract_path(tmp_path / "test.docx")
        # python-docx not installed in test env
        assert result.error is not None
        assert "docx" in result.error.lower()
