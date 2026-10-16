"""Tests for womblex.redact detection and masking.

RedactionDetector, RedactionInfo, build_detector and pre_ocr_mask.

Positive-case tests (detecting known black boxes) need controlled inputs with
exact geometry, so ``redacted_image`` is constructed inline.  Negative-case
tests (no spurious detections) use a synthetic scanned page.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from tests._synthetic import SCANS_DIR
from womblex.config import RedactionConfig
from womblex.redact import RedactionDetector, RedactionInfo
from womblex.redact.stage import build_detector
from womblex.redact.utils import pre_ocr_mask


@pytest.fixture
def redacted_image() -> np.ndarray:
    """RGB image with two black rectangles at known coordinates."""
    img = np.full((400, 600, 3), 240, dtype=np.uint8)
    img[80:120, 100:500] = 0   # wide bar
    img[200:260, 150:350] = 0  # narrower bar
    return img


@pytest.fixture
def clean_image() -> np.ndarray:
    """A synthetic scanned page (a heading and one line) as grayscale — no redaction boxes."""
    return np.array(Image.open(SCANS_DIR / "page-sparse.png").convert("L"))


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

    def test_exclude_rects_drops_centred_candidates(self, redacted_image: np.ndarray) -> None:
        """A candidate whose centre falls inside an exclude rect is dropped."""
        # The wide bar in the fixture is at rows 80-120, cols 100-500 → centre (300, 100).
        # The narrower bar is at rows 200-260, cols 150-350 → centre (250, 230).
        # Exclude the wide-bar region; expect only the narrower bar to survive.
        exclude = [(50, 50, 550, 150)]  # covers wide bar's centre, not narrower bar's
        redactions = self.detector.detect(redacted_image, page=0, exclude_rects=exclude)
        assert len(redactions) == 1
        # Surviving bar should be the narrower one (y around 200-260)
        _x1, y1, _x2, _y2 = redactions[0].bbox
        assert 180 <= y1 <= 220

    def test_exclude_rects_none_is_noop(self, redacted_image: np.ndarray) -> None:
        """exclude_rects=None matches the un-filtered detect()."""
        baseline = self.detector.detect(redacted_image, page=0)
        with_none = self.detector.detect(redacted_image, page=0, exclude_rects=None)
        assert len(baseline) == len(with_none) == 2

    def test_exclude_rects_empty_is_noop(self, redacted_image: np.ndarray) -> None:
        """exclude_rects=[] matches the un-filtered detect()."""
        baseline = self.detector.detect(redacted_image, page=0)
        with_empty = self.detector.detect(redacted_image, page=0, exclude_rects=[])
        assert len(baseline) == len(with_empty) == 2


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
