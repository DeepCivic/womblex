"""Tests for womblex.redact — detection, masking, stage logic, and utils.

The RedactionDetector operates on numpy image arrays.  Positive-case
tests (detecting known black boxes) need controlled inputs with exact
geometry, so ``redacted_image`` is constructed inline.  Negative-case
tests (no spurious detections) use a real FUNSD benchmark image.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image

from womblex.config import RedactionConfig
from womblex.redact import RedactionDetector, RedactionInfo
from womblex.redact.stage import (
    RedactionReport,
    annotate_chunks,
    annotate_extraction,
    apply_text_redaction,
    build_detector,
)
from womblex.redact.utils import pre_ocr_mask

if TYPE_CHECKING:
    from womblex.process.chunker import TextChunk

_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "fixtures"


@pytest.fixture
def redacted_image() -> np.ndarray:
    """RGB image with two black rectangles at known coordinates."""
    img = np.full((400, 600, 3), 240, dtype=np.uint8)
    img[80:120, 100:500] = 0   # wide bar
    img[200:260, 150:350] = 0  # narrower bar
    return img


@pytest.fixture
def clean_image() -> np.ndarray:
    """Real FUNSD benchmark image (sparse form, 25 words) as grayscale — no redaction boxes."""
    path = _FIXTURES_DIR / "funsd" / "images" / "85540866.png"
    return np.array(Image.open(path).convert("L"))


# ---------------------------------------------------------------------------
# RedactionDetector.detect
# ---------------------------------------------------------------------------


class TestRedactionDetect:
    def setup_method(self) -> None:
        self.detector = RedactionDetector()

    def test_detects_black_rectangles(self, redacted_image: np.ndarray) -> None:
        redactions = self.detector.detect(redacted_image, page=0)
        # We drew two black boxes in the fixture
        assert len(redactions) == 2

    def test_no_redactions_on_clean_image(self, clean_image: np.ndarray) -> None:
        redactions = self.detector.detect(clean_image, page=0)
        assert len(redactions) == 0

    def test_redaction_bboxes_are_reasonable(self, redacted_image: np.ndarray) -> None:
        redactions = self.detector.detect(redacted_image, page=0)
        for r in redactions:
            x1, y1, x2, y2 = r.bbox
            assert x1 < x2
            assert y1 < y2
            assert r.area_px > 0

    def test_page_number_is_set(self, redacted_image: np.ndarray) -> None:
        redactions = self.detector.detect(redacted_image, page=7)
        for r in redactions:
            assert r.page == 7

    def test_grayscale_input(self) -> None:
        """Detector should handle grayscale images too."""
        gray = np.full((400, 600), 240, dtype=np.uint8)
        gray[100:150, 100:400] = 0  # black bar
        redactions = self.detector.detect(gray, page=0)
        assert len(redactions) >= 1

    def test_rejects_tiny_contours(self) -> None:
        """Very small dark spots should not count as redactions."""
        img = np.full((400, 600, 3), 240, dtype=np.uint8)
        # A single pixel is too small
        img[200, 300] = 0
        redactions = self.detector.detect(img, page=0)
        assert len(redactions) == 0

    def test_rejects_full_page_black(self) -> None:
        """An entirely black image should not count as a redaction."""
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        redactions = self.detector.detect(img, page=0)
        # The contour covers > max_area_ratio so should be rejected
        assert len(redactions) == 0

    def test_custom_threshold(self) -> None:
        """Darker threshold should still detect very dark boxes."""
        img = np.full((400, 600, 3), 240, dtype=np.uint8)
        # Box with pixel value 30 (very dark)
        img[100:160, 100:400] = 30
        detector = RedactionDetector(threshold=40)
        redactions = detector.detect(img, page=0)
        assert len(redactions) >= 1

    def test_near_threshold_not_detected(self) -> None:
        """Boxes lighter than threshold should not be detected."""
        img = np.full((400, 600, 3), 240, dtype=np.uint8)
        # Box with value 80, which is above default threshold of 50
        img[100:160, 100:400] = 80
        detector = RedactionDetector(threshold=50)
        redactions = detector.detect(img, page=0)
        assert len(redactions) == 0


# ---------------------------------------------------------------------------
# RedactionDetector.mask
# ---------------------------------------------------------------------------


class TestRedactionMask:
    def setup_method(self) -> None:
        self.detector = RedactionDetector()

    def test_mask_replaces_with_white(self, redacted_image: np.ndarray) -> None:
        redactions = self.detector.detect(redacted_image, page=0)
        masked = self.detector.mask(redacted_image, redactions)

        for r in redactions:
            x1, y1, x2, y2 = r.bbox
            region = masked[y1:y2, x1:x2]
            # All pixels in the masked region should be white (255)
            assert np.all(region == 255)

    def test_mask_preserves_non_redacted_areas(self, redacted_image: np.ndarray) -> None:
        redactions = self.detector.detect(redacted_image, page=0)
        masked = self.detector.mask(redacted_image, redactions)

        # Non-redacted background should remain unchanged
        assert np.array_equal(masked[0, 0], redacted_image[0, 0])

    def test_mask_does_not_modify_original(self, redacted_image: np.ndarray) -> None:
        original = redacted_image.copy()
        redactions = self.detector.detect(redacted_image, page=0)
        self.detector.mask(redacted_image, redactions)

        # Original should be unchanged
        assert np.array_equal(redacted_image, original)

    def test_mask_with_no_redactions(self, clean_image: np.ndarray) -> None:
        masked = self.detector.mask(clean_image, [])
        assert np.array_equal(masked, clean_image)

    def test_mask_grayscale(self) -> None:
        gray = np.full((400, 600), 200, dtype=np.uint8)
        gray[100:150, 100:400] = 0

        redactions = [RedactionInfo(bbox=(100, 100, 400, 150), page=0, area_px=300 * 50)]
        detector = RedactionDetector()
        masked = detector.mask(gray, redactions)

        assert np.all(masked[100:150, 100:400] == 255)
        # Untouched area preserved
        assert masked[0, 0] == 200


# ---------------------------------------------------------------------------
# RedactionInfo dataclass
# ---------------------------------------------------------------------------


class TestRedactionInfo:
    def test_fields(self) -> None:
        r = RedactionInfo(bbox=(10, 20, 100, 50), page=3, area_px=2700)
        assert r.bbox == (10, 20, 100, 50)
        assert r.page == 3
        assert r.area_px == 2700


# ---------------------------------------------------------------------------
# Backwards-compatible import from ingest.redaction
# ---------------------------------------------------------------------------


class TestBackwardsCompatImport:
    def test_import_from_ingest_redaction(self) -> None:
        from womblex.ingest.redaction import RedactionDetector as LegacyDetector
        from womblex.ingest.redaction import RedactionInfo as LegacyInfo

        assert LegacyDetector is RedactionDetector
        assert LegacyInfo is RedactionInfo


# ---------------------------------------------------------------------------
# Stage logic: build_detector
# ---------------------------------------------------------------------------


class TestBuildDetector:
    def test_uses_config_values(self) -> None:
        cfg = RedactionConfig(threshold=30, min_area_ratio=0.01, max_area_ratio=0.5)
        detector = build_detector(cfg)
        assert detector.threshold == 30
        assert detector.min_area_ratio == 0.01
        assert detector.max_area_ratio == 0.5


# ---------------------------------------------------------------------------
# Utils: pre_ocr_mask
# ---------------------------------------------------------------------------


class TestPreOcrMask:
    def test_masks_redactions(self, redacted_image: np.ndarray) -> None:
        detector = RedactionDetector()
        masked, redactions = pre_ocr_mask(redacted_image, page=0, detector=detector)
        assert len(redactions) == 2
        for r in redactions:
            x1, y1, x2, y2 = r.bbox
            assert np.all(masked[y1:y2, x1:x2] == 255)

    def test_no_redactions_returns_original(self, clean_image: np.ndarray) -> None:
        detector = RedactionDetector()
        masked, redactions = pre_ocr_mask(clean_image, page=0, detector=detector)
        assert len(redactions) == 0
        assert np.array_equal(masked, clean_image)


# ---------------------------------------------------------------------------
# Stage logic: RedactionReport
# ---------------------------------------------------------------------------


class TestRedactionReport:
    def test_empty_report(self) -> None:
        report = RedactionReport()
        assert report.total == 0
        assert report.affected_pages == []

    def test_report_with_data(self) -> None:
        r1 = RedactionInfo(bbox=(0, 0, 100, 50), page=0, area_px=5000)
        r2 = RedactionInfo(bbox=(0, 0, 200, 30), page=2, area_px=6000)
        report = RedactionReport(page_redactions={0: [r1], 2: [r2]})
        assert report.total == 2
        assert report.affected_pages == [0, 2]


# ---------------------------------------------------------------------------
# Stage logic: apply_text_redaction
# ---------------------------------------------------------------------------


class TestApplyTextRedaction:
    def _make_pages(self, texts: list[str]) -> list:
        from womblex.ingest.extract import PageResult

        return [PageResult(page_number=i, text=t, method="test") for i, t in enumerate(texts)]

    def _make_report(self, *pages: int) -> RedactionReport:
        return RedactionReport(
            page_redactions={
                p: [RedactionInfo(bbox=(0, 0, 100, 50), page=p, area_px=5000)]
                for p in pages
            }
        )

    def test_flag_mode_no_change(self) -> None:
        pages = self._make_pages(["hello", "world"])
        report = self._make_report(0)
        result = apply_text_redaction(pages, report, mode="flag")
        assert result[0].text == "hello"
        assert result[1].text == "world"

    def test_blackout_mode_prepends_marker(self) -> None:
        pages = self._make_pages(["sensitive text", "clean text"])
        report = self._make_report(0)
        apply_text_redaction(pages, report, mode="blackout")
        assert pages[0].text.startswith("[REDACTED]")
        assert pages[1].text == "clean text"

    def test_blackout_mode_empty_page(self) -> None:
        pages = self._make_pages([""])
        report = self._make_report(0)
        apply_text_redaction(pages, report, mode="blackout")
        assert pages[0].text == "[REDACTED]"

    def test_delete_mode_clears_page(self) -> None:
        pages = self._make_pages(["sensitive text", "clean text"])
        report = self._make_report(0)
        apply_text_redaction(pages, report, mode="delete")
        assert pages[0].text == ""
        assert pages[1].text == "clean text"

    def test_empty_report_no_change(self) -> None:
        pages = self._make_pages(["hello"])
        report = RedactionReport()
        apply_text_redaction(pages, report, mode="blackout")
        assert pages[0].text == "hello"


# ---------------------------------------------------------------------------
# Stage logic: annotate_chunks
# ---------------------------------------------------------------------------


class TestAnnotateChunks:
    def _make_chunk(self, index: int, **kwargs: object) -> TextChunk:
        from womblex.process.chunker import TextChunk

        chunk = TextChunk(
            text=f"chunk {index}",
            start_char=0,
            end_char=7,
            chunk_index=index,
        )
        for k, v in kwargs.items():
            setattr(chunk, k, v)
        return chunk

    def _make_report(self, pages: dict[int, int]) -> RedactionReport:
        """Build a report with *n* redactions per page."""
        page_redactions: dict[int, list[RedactionInfo]] = {}
        for page, count in pages.items():
            page_redactions[page] = [
                RedactionInfo(bbox=(0, 0, 100, 50), page=page, area_px=5000)
                for _ in range(count)
            ]
        return RedactionReport(page_redactions=page_redactions)

    def test_empty_report_returns_unchanged(self) -> None:
        chunks = [self._make_chunk(0), self._make_chunk(1)]
        report = RedactionReport()
        result = annotate_chunks(chunks, report)
        assert result is chunks
        assert not hasattr(chunks[0], "has_redaction")
        assert not hasattr(chunks[1], "has_redaction")

    def test_flags_chunk_with_source_pages(self) -> None:
        chunk = self._make_chunk(0, source_pages=[0, 1])
        report = self._make_report({1: 1})
        annotate_chunks([chunk], report)
        assert chunk.has_redaction is True

    def test_skips_chunk_with_unaffected_source_pages(self) -> None:
        chunk = self._make_chunk(0, source_pages=[2, 3])
        report = self._make_report({0: 1})
        annotate_chunks([chunk], report)
        assert not hasattr(chunk, "has_redaction")

    def test_flags_chunk_with_page_number(self) -> None:
        chunk = self._make_chunk(0, page_number=5)
        report = self._make_report({5: 2})
        annotate_chunks([chunk], report)
        assert chunk.has_redaction is True

    def test_skips_chunk_with_unaffected_page_number(self) -> None:
        chunk = self._make_chunk(0, page_number=3)
        report = self._make_report({0: 1})
        annotate_chunks([chunk], report)
        assert not hasattr(chunk, "has_redaction")

    def test_source_pages_takes_precedence_over_page_number(self) -> None:
        """When both attributes exist, source_pages is checked first."""
        chunk = self._make_chunk(0, source_pages=[0], page_number=5)
        report = self._make_report({5: 1})
        annotate_chunks([chunk], report)
        # source_pages=[0] doesn't overlap affected page 5, so no flag
        assert not hasattr(chunk, "has_redaction")

    def test_chunk_without_page_attrs_is_skipped(self) -> None:
        chunk = self._make_chunk(0)
        report = self._make_report({0: 1})
        annotate_chunks([chunk], report)
        assert not hasattr(chunk, "has_redaction")

    def test_mixed_chunks(self) -> None:
        c0 = self._make_chunk(0, source_pages=[1])
        c1 = self._make_chunk(1, page_number=2)
        c2 = self._make_chunk(2, page_number=9)
        report = self._make_report({1: 1, 2: 1})
        annotate_chunks([c0, c1, c2], report)
        assert c0.has_redaction is True
        assert c1.has_redaction is True
        assert not hasattr(c2, "has_redaction")

    def test_empty_source_pages_falls_through(self) -> None:
        """source_pages=[] is falsy — should fall through to page_number."""
        chunk = self._make_chunk(0, source_pages=[], page_number=0)
        report = self._make_report({0: 1})
        annotate_chunks([chunk], report)
        assert chunk.has_redaction is True


# ---------------------------------------------------------------------------
# Stage logic: annotate_extraction
# ---------------------------------------------------------------------------


class TestAnnotateExtraction:
    def test_adds_warnings(self) -> None:
        from womblex.ingest.extract import ExtractionResult

        extraction = ExtractionResult(pages=[], method="test")
        r1 = RedactionInfo(bbox=(0, 0, 100, 50), page=0, area_px=5000)
        report = RedactionReport(page_redactions={0: [r1]})

        annotate_extraction(extraction, report)
        assert len(extraction.warnings) == 1
        assert "page 0" in extraction.warnings[0]
        assert "1 redacted region" in extraction.warnings[0]

    def test_no_warnings_for_empty_report(self) -> None:
        from womblex.ingest.extract import ExtractionResult

        extraction = ExtractionResult(pages=[], method="test")
        report = RedactionReport()

        annotate_extraction(extraction, report)
        assert len(extraction.warnings) == 0
