"""Integration tests for the spreadsheet-print extractor.

Uses real ACT FOI manifest fixtures from the womblex-development-fixtures repo
(`fixtures/fixtures/womblex-collection/_documents/`). Tests skip gracefully if
those fixtures are unavailable.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import fitz

from womblex.ingest.spreadsheet_print import extract_spreadsheet_print
from womblex.ingest.page_profile import qualify_for_spreadsheet_print, profile_pages


_FIXTURES = (
    Path(__file__).resolve().parent.parent
    / "fixtures" / "fixtures" / "womblex-collection" / "_documents"
)
_MASTER_INDEX = _FIXTURES / (
    "Early-childhood-education-and-care-incident-recordsOrder-to-"
    "TableAssembly-resolution-of-24-June-2025"
    "Index-of-returned-documents-for-part-2aRevised.pdf"
)
_SCHEDULE_2B = _FIXTURES / "Schedule-of-documents-Part-2b.pdf"


def _has_fixtures() -> bool:
    return _MASTER_INDEX.is_file() and _SCHEDULE_2B.is_file()


pytestmark = pytest.mark.skipif(
    not _has_fixtures(),
    reason="ACT FOI Index Files fixtures not available on this host",
)


class TestQualifier:
    """Cheap pre-extraction qualifier should accept index-shaped docs and
    reject vanilla letters."""

    def test_qualifier_accepts_master_index(self) -> None:
        doc = fitz.open(str(_MASTER_INDEX))
        try:
            profiles = profile_pages(doc)
            assert qualify_for_spreadsheet_print(profiles, _MASTER_INDEX.name)
        finally:
            doc.close()

    def test_qualifier_accepts_schedule_2b(self) -> None:
        doc = fitz.open(str(_SCHEDULE_2B))
        try:
            profiles = profile_pages(doc)
            assert qualify_for_spreadsheet_print(profiles, _SCHEDULE_2B.name)
        finally:
            doc.close()

    def test_qualifier_rejects_when_no_table_signal_and_no_hint(self) -> None:
        # Construct a profile manually: native pages but no table signal
        # and a filename without spreadsheet-shaped hints.
        from womblex.ingest.page_profile import PageProfile
        profiles = [
            PageProfile(
                page_number=i, width=595, height=842,
                char_count=2000, image_count=0, vector_drawings=0,
                has_text_layer=True, needs_ocr=False,
                has_table_signal=False, has_form_signal=False,
                has_handwriting_signal=False,
            )
            for i in range(3)
        ]
        assert not qualify_for_spreadsheet_print(profiles, "letter.pdf")

    def test_qualifier_rejects_when_table_signal_low_and_no_hint(self) -> None:
        # 1 of 3 pages has table signal (33 %), filename is letter-shaped.
        from womblex.ingest.page_profile import PageProfile
        profiles = [
            PageProfile(
                page_number=i, width=595, height=842,
                char_count=2000, image_count=0, vector_drawings=0,
                has_text_layer=True, needs_ocr=False,
                has_table_signal=(i == 0), has_form_signal=False,
                has_handwriting_signal=False,
            )
            for i in range(3)
        ]
        assert not qualify_for_spreadsheet_print(profiles, "letter.pdf")

    def test_qualifier_rejects_compliance_notice_shape(self) -> None:
        # Compliance-notice cohort: every page has `has_table_signal=True`
        # (real rules-of-law tables exist) but `has_manifest_signal=False`
        # (tables are small, not manifest-shape). Pre-§2 qualifier tripped
        # on the 50%-of-pages rule and routed the doc through the manifest
        # extractor, producing conf=0.70 garbage in `tables[0]`.
        from womblex.ingest.page_profile import PageProfile
        profiles = [
            PageProfile(
                page_number=i, width=595, height=842,
                char_count=2000, image_count=0, vector_drawings=0,
                has_text_layer=True, needs_ocr=False,
                has_table_signal=True, has_form_signal=False,
                has_handwriting_signal=False,
                has_manifest_signal=False,
            )
            for i in range(3)
        ]
        assert not qualify_for_spreadsheet_print(profiles, "compliance-notice.pdf")

    def test_qualifier_accepts_manifest_signal_with_filename_hint(self) -> None:
        # Filename hint + ≥1 manifest-shape page.
        from womblex.ingest.page_profile import PageProfile
        profiles = [
            PageProfile(
                page_number=0, width=595, height=842,
                char_count=10000, image_count=0, vector_drawings=0,
                has_text_layer=True, needs_ocr=False,
                has_table_signal=True, has_form_signal=False,
                has_handwriting_signal=False,
                has_manifest_signal=True,
            ),
        ]
        assert qualify_for_spreadsheet_print(profiles, "schedule-of-evidence.pdf")

    def test_qualifier_accepts_manifest_signal_majority_pages(self) -> None:
        # No filename hint but ≥50% of pages are manifest-shape.
        from womblex.ingest.page_profile import PageProfile
        profiles = [
            PageProfile(
                page_number=i, width=595, height=842,
                char_count=10000, image_count=0, vector_drawings=0,
                has_text_layer=True, needs_ocr=False,
                has_table_signal=True, has_form_signal=False,
                has_handwriting_signal=False,
                has_manifest_signal=(i < 2),  # 2/3 pages manifest-shape
            )
            for i in range(3)
        ]
        assert qualify_for_spreadsheet_print(profiles, "noname.pdf")


class TestMasterIndex:
    """The 37-page rotated FOI manifest — the corpus's hardest tabular doc."""

    @pytest.fixture(scope="class")
    def extracted(self):
        doc = fitz.open(str(_MASTER_INDEX))
        try:
            tables, doc_meta = extract_spreadsheet_print(
                doc, metadata_location="both",
            )
            return tables, doc_meta
        finally:
            doc.close()

    def test_single_multi_page_table(self, extracted) -> None:
        tables, _ = extracted
        assert len(tables) == 1

    def test_row_count_matches_corpus_scale(self, extracted) -> None:
        # Corpus is ~2615 docs; manifest covers most. Tolerance for
        # post-manifest additions / pagination quirks.
        tables, _ = extracted
        assert 2300 <= len(tables[0].rows) <= 2700

    def test_canonical_columns_present(self, extracted) -> None:
        tables, _ = extracted
        headers = tables[0].headers
        # Exact column list — locks the schema for downstream consumers.
        assert headers == [
            "Unique ID", "Directorate", "Service Name", "File Name",
            "Document Type", "Case Number", "Subsection code",
            "Issue Date", "Author", "Privilege", "Reason for Privilege",
            "Out of Scope Exemption", "Assembly Permitted Redactions",
        ]

    def test_first_row_known_values(self, extracted) -> None:
        tables, _ = extracted
        row = tables[0].rows[0]
        assert row[0] == "00008"
        assert row[1] == "EDU"
        assert row[2] == "360 Early Education Throsby"
        assert row[5] == "CAS-00312020"
        assert row[6] == "2ai"
        assert row[7] == "02/09/2024"
        assert row[8] == "CECA"

    def test_metadata_block_captured(self, extracted) -> None:
        tables, doc_meta = extracted
        # Both surfaces populated when metadata_location="both".
        assert doc_meta == tables[0].context
        assert doc_meta.get("213A reference") == "213A-2025-008"
        assert doc_meta.get("Element #") == "2(a)(i) - 2(a)(iv)"


class TestSchedulePart2b:
    """A simpler, single-page non-rotated example to confirm generality."""

    @pytest.fixture(scope="class")
    def extracted(self):
        doc = fitz.open(str(_SCHEDULE_2B))
        try:
            tables, doc_meta = extract_spreadsheet_print(
                doc, metadata_location="both",
            )
            return tables, doc_meta
        finally:
            doc.close()

    def test_extracts_rows(self, extracted) -> None:
        tables, _ = extracted
        assert len(tables) == 1
        assert 15 <= len(tables[0].rows) <= 30  # ~22 rows on this single page

    def test_metadata_has_213a_reference(self, extracted) -> None:
        _, doc_meta = extracted
        assert doc_meta.get("213A reference") == "213A-2025-008"


class TestMetadataLocation:
    """The three metadata_location modes shape where the metadata lands."""

    def test_table_only(self) -> None:
        doc = fitz.open(str(_SCHEDULE_2B))
        try:
            tables, doc_meta = extract_spreadsheet_print(
                doc, metadata_location="table",
            )
            assert tables[0].context  # populated
            assert doc_meta == {}      # empty
        finally:
            doc.close()

    def test_document_only(self) -> None:
        doc = fitz.open(str(_SCHEDULE_2B))
        try:
            tables, doc_meta = extract_spreadsheet_print(
                doc, metadata_location="document",
            )
            assert tables[0].context == {}  # empty
            assert doc_meta                  # populated
        finally:
            doc.close()

    def test_invalid_value_raises(self) -> None:
        doc = fitz.open(str(_SCHEDULE_2B))
        try:
            with pytest.raises(ValueError):
                extract_spreadsheet_print(doc, metadata_location="invalid")
        finally:
            doc.close()
