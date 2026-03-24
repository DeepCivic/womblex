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
    _normalise_bbox,
    _normalise_text,
    get_extractor,
)
from womblex.ingest.strategies import (
    DocxExtractor,
    HybridExtractor,
    ImageExtractor,
    NativeNarrativeExtractor,
    NativeWithStructuredExtractor,
    ScannedHandwrittenExtractor,
    ScannedMachinewrittenExtractor,
    ScannedMixedExtractor,
    StructuredExtractor,
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


# ---------------------------------------------------------------------------
# _normalise_text
# ---------------------------------------------------------------------------


class TestNormaliseText:
    # -- RES-001: apostrophe + dollar/euro → apostrophe + s --

    def test_apostrophe_dollar_no_space(self) -> None:
        assert _normalise_text("Children'$ Education") == "Children's Education"

    def test_curly_apostrophe_dollar_no_space(self) -> None:
        assert _normalise_text("Children\u2019$ Education") == "Children\u2019s Education"

    def test_apostrophe_dollar_with_space(self) -> None:
        assert _normalise_text("the child' $ specific health") == "the child's specific health"

    def test_curly_apostrophe_dollar_with_space(self) -> None:
        assert _normalise_text("the child\u2019 $ specific health") == "the child\u2019s specific health"

    def test_apostrophe_euro(self) -> None:
        assert _normalise_text("Witness C'\u20ac statement") == "Witness C's statement"

    def test_curly_apostrophe_euro(self) -> None:
        assert _normalise_text("Witness C\u2019\u20ac statement") == "Witness C\u2019s statement"

    def test_dollar_sign_preserved_in_normal_context(self) -> None:
        assert _normalise_text("cost is $500") == "cost is $500"

    # -- RES-002: URL scheme corruption --

    def test_url_scheme_lL(self) -> None:
        assert _normalise_text("httplLwww.example.com") == "http://www.example.com"

    def test_url_scheme_colon_lL(self) -> None:
        assert _normalise_text("http:lLwww.example.com") == "http://www.example.com"

    def test_correct_url_unchanged(self) -> None:
        assert _normalise_text("http://www.example.com") == "http://www.example.com"

    # -- RES-003: single-line page footers --

    def test_footer_standard(self) -> None:
        assert _normalise_text("body text\n14 | P a g e\nmore text") == "body text\n\nmore text"

    def test_footer_corrupted_letters(self) -> None:
        assert _normalise_text("body text\n5 | F' 2 < F\nmore text") == "body text\n\nmore text"

    def test_footer_ampersand_variants(self) -> None:
        assert _normalise_text("body\n20 | P & g \u20ac\nmore") == "body\n\nmore"

    # -- RES-003 extended: split footers across two lines --

    def test_split_footer_number_pipe_then_page(self) -> None:
        text = "body text\n11 |\nP a &\nmore text"
        result = _normalise_text(text)
        assert "11 |" not in result
        assert "P a &" not in result
        assert "body text" in result
        assert "more text" in result

    def test_split_footer_bare_number_then_page(self) -> None:
        text = "body text\n15\nP a & e\nmore text"
        result = _normalise_text(text)
        assert "P a & e" not in result
        assert "body text" in result
        assert "more text" in result

    def test_split_footer_f_prime_variant(self) -> None:
        text = "body text\n16\nF' 2 < ?\nmore text"
        result = _normalise_text(text)
        assert "F' 2 < ?" not in result
        assert "body text" in result

    def test_split_footer_does_not_match_paragraph_number(self) -> None:
        """Bare number followed by normal text must NOT be removed."""
        text = "5\nProtection from harms and hazards"
        assert _normalise_text(text) == text

    def test_split_footer_does_not_match_body_starting_with_p(self) -> None:
        """Bare number followed by 'Provider' (starts with P but no spaces) must survive."""
        text = "48\nProvider failed to notify"
        assert _normalise_text(text) == text

    def test_no_false_positive_on_dollar_amounts(self) -> None:
        """Lines with dollar amounts near apostrophes should not be mangled."""
        text = "Penalty: $11 400"
        assert _normalise_text(text) == text


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
        pos = Position(x=0.0, y=0.0, width=1.0, height=1.0)
        result = ExtractionResult(
            pages=[PageResult(page_number=0, text="Test", method="native")],
            method="native_with_structured",
            tables=[TableData(headers=["A", "B"], rows=[["1", "2"]], position=pos, confidence=0.8)],
            forms=[FormField(field_name="Name", value="Alice", position=pos, confidence=0.9)],
            images=[ImageData(alt_text="photo", position=pos, confidence=0.7)],
            text_blocks=[TextBlock(text="Test", position=pos, block_type="paragraph", confidence=0.9)],
        )
        assert len(result.tables) == 1
        assert result.tables[0].headers == ["A", "B"]
        assert len(result.forms) == 1
        assert result.forms[0].field_name == "Name"
        assert len(result.images) == 1
        assert len(result.text_blocks) == 1


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

    def test_native_narrative_returns_correct_extractor(self) -> None:
        ext = get_extractor(self._make_profile(DocumentType.NATIVE_NARRATIVE))
        assert isinstance(ext, NativeNarrativeExtractor)

    def test_native_with_structured_returns_correct_extractor(self) -> None:
        ext = get_extractor(self._make_profile(DocumentType.NATIVE_WITH_STRUCTURED))
        assert isinstance(ext, NativeWithStructuredExtractor)

    def test_structured_returns_correct_extractor(self) -> None:
        ext = get_extractor(self._make_profile(DocumentType.STRUCTURED))
        assert isinstance(ext, StructuredExtractor)

    def test_scanned_machinewritten_returns_correct_extractor(self) -> None:
        ext = get_extractor(self._make_profile(DocumentType.SCANNED_MACHINEWRITTEN))
        assert isinstance(ext, ScannedMachinewrittenExtractor)

    def test_scanned_handwritten_returns_correct_extractor(self) -> None:
        ext = get_extractor(self._make_profile(DocumentType.SCANNED_HANDWRITTEN))
        assert isinstance(ext, ScannedHandwrittenExtractor)

    def test_scanned_mixed_returns_correct_extractor(self) -> None:
        ext = get_extractor(self._make_profile(DocumentType.SCANNED_MIXED))
        assert isinstance(ext, ScannedMixedExtractor)

    def test_hybrid_returns_correct_extractor(self) -> None:
        ext = get_extractor(self._make_profile(DocumentType.HYBRID))
        assert isinstance(ext, HybridExtractor)

    def test_image_returns_correct_extractor(self) -> None:
        ext = get_extractor(self._make_profile(DocumentType.IMAGE))
        assert isinstance(ext, ImageExtractor)

    def test_spreadsheet_returns_correct_extractor(self) -> None:
        ext = get_extractor(self._make_profile(DocumentType.SPREADSHEET))
        assert isinstance(ext, SpreadsheetExtractor)

    def test_docx_returns_correct_extractor(self) -> None:
        ext = get_extractor(self._make_profile(DocumentType.DOCX))
        assert isinstance(ext, DocxExtractor)

    def test_unknown_returns_fallback(self) -> None:
        ext = get_extractor(self._make_profile(DocumentType.UNKNOWN))
        assert isinstance(ext, NativeNarrativeExtractor)


# ---------------------------------------------------------------------------
# SpreadsheetExtractor (uses real fixture spreadsheets)
# ---------------------------------------------------------------------------


class TestSpreadsheetExtractor:
    def test_extracts_real_csv(self, spreadsheet_dir: Path) -> None:
        csv_path = spreadsheet_dir / "Approved-providers-au-export_20260204.csv"
        if not csv_path.exists():
            pytest.skip("CSV fixture not available")

        ext = SpreadsheetExtractor()
        results = ext.extract_path(csv_path)

        assert len(results) >= 1
        for r in results:
            assert r.method == "spreadsheet"
            assert r.error is None
            assert r.metadata is not None

    def test_extracts_real_xlsx(self, spreadsheet_dir: Path) -> None:
        xlsx_path = spreadsheet_dir / "mso-statistics-sept-qtr-2025.xlsx"
        if not xlsx_path.exists():
            pytest.skip("Excel fixture not available")

        ext = SpreadsheetExtractor()
        results = ext.extract_path(xlsx_path)

        assert len(results) >= 1
        for r in results:
            assert r.method == "spreadsheet"
            assert r.metadata is not None

    def test_handles_missing_file(self, tmp_path: Path) -> None:
        ext = SpreadsheetExtractor()
        results = ext.extract_path(tmp_path / "missing.csv")
        assert len(results) == 1
        assert results[0].error is not None


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
