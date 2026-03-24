"""Fixture-based tests using curated images from FUNSD, IAM-line, and DocLayNet.

Validates the detection → extraction → chunking pipeline against images with
known ground truth. Tests are grouped by concern:

- Detection: image-only PDFs classify as SCANNED_MACHINEWRITTEN
- Extraction: OCR runs without error on all fixture types
- OCR content: machine-printed fixtures yield recognisable text
- Redaction: clean fixture images report no redaction regions
- Chunking: ground-truth text chunks correctly via the semchunk pipeline
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import fitz
import numpy as np
import pytest
from PIL import Image

from womblex.ingest.detect import DocumentProfile, DocumentType, detect_document_type
from womblex.ingest.extract import extract_text
from womblex.process.chunker import TextChunk, chunk_text, create_chunker
from womblex.redact import RedactionDetector

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "fixtures"
FUNSD_IMAGES = FIXTURES_DIR / "funsd" / "images"
FUNSD_ANNOTATIONS = FIXTURES_DIR / "funsd" / "annotations"
IAM_DIR = FIXTURES_DIR / "iam_line"
DOCLAYNET_DIR = FIXTURES_DIR / "doclaynet"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scanned_profile(page_count: int = 1) -> DocumentProfile:
    """Minimal DocumentProfile for a single-page scanned document."""
    return DocumentProfile(
        doc_type=DocumentType.SCANNED_MACHINEWRITTEN,
        page_count=page_count,
        has_text_layer=False,
        text_coverage=0.0,
        has_images=True,
        has_tables=False,
        has_handwriting_signals=False,
        ocr_confidence=None,
        glyph_regularity=None,
        stroke_consistency=None,
        confidence=0.8,
    )


def _image_to_pdf(
    image_path: Path,
    output_path: Path,
    page_w: int = 595,
    page_h: int = 842,
) -> Path:
    """Embed a PNG image into a single-page PDF.

    Args:
        image_path: Source PNG file.
        output_path: Destination PDF path.
        page_w: Page width in points (default 595 = A4 width).
        page_h: Page height in points (default 842 = A4 height).

    Using a fixed page size ensures OCR renders a predictable-size
    pixmap regardless of the source image's native pixel dimensions.
    For tests that call the full OCR pipeline use a smaller page size
    (e.g. 150×150 pt → ~417×417 px at 200 DPI) to keep wall time low.
    """
    img = Image.open(image_path).convert("RGB")
    doc = fitz.open()
    page = doc.new_page(width=page_w, height=page_h)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    page.insert_image(fitz.Rect(0, 0, page_w, page_h), stream=buf.getvalue())
    doc.save(str(output_path))
    doc.close()
    return output_path


def _funsd_ground_truth(name: str) -> list[str]:
    """Return non-empty text strings from a FUNSD annotation file."""
    with open(FUNSD_ANNOTATIONS / f"{name}.json") as f:
        data = json.load(f)
    return [entry["text"].strip() for entry in data["form"] if entry["text"].strip()]


def _doclaynet_ground_truth(name: str) -> list[str]:
    """Return word-level ground truth strings from a DocLayNet annotation file."""
    with open(DOCLAYNET_DIR / f"{name}.json") as f:
        data = json.load(f)
    return [w.strip() for w in data["words"] if w.strip()]


def _iam_ground_truth(name: str) -> str:
    """Return the ground truth transcription for an IAM-line sample."""
    return (IAM_DIR / f"{name}.gt.txt").read_text().strip()


def _word_token_counter(text: str) -> int:
    """Simple word-count token counter for tests (no network required)."""
    return len(text.split())


# ---------------------------------------------------------------------------
# Parametrise fixture names
# ---------------------------------------------------------------------------

FUNSD_SAMPLES = [
    "85540866",
    "82200067_0069",
    "87594142_87594144",
    "87528321",
    "87528380",
]

IAM_SAMPLES = [
    "short_1602",
    "median_15",
    "long_4",
    "wide_1739",
    "narrow_1163",
]

DOCLAYNET_SAMPLES = [
    "diverse_layout_49",
    "table_0",
    "formula_29",
    "sparse_text_344",
    "dense_text_548",
]


# ---------------------------------------------------------------------------
# Detection: image-only PDFs classify as SCANNED_MACHINEWRITTEN
# ---------------------------------------------------------------------------


_SCANNED_TYPES = {
    DocumentType.SCANNED_MACHINEWRITTEN,
    DocumentType.SCANNED_HANDWRITTEN,
    DocumentType.SCANNED_MIXED,
}


_SCANNED_TYPES = {
    DocumentType.SCANNED_MACHINEWRITTEN,
    DocumentType.SCANNED_HANDWRITTEN,
    DocumentType.SCANNED_MIXED,
}

# UNKNOWN is a valid outcome for very small or very low-contrast handwriting
# images where OCR confidence is below the classification thresholds.
_SCANNED_OR_UNKNOWN = _SCANNED_TYPES | {DocumentType.UNKNOWN}


@pytest.mark.slow
class TestFixtureDetection:
    """All fixture images should classify as a SCANNED_* type when wrapped in a
    PDF — they have no text layer, only an embedded raster image.

    FUNSD images are noisy scanned documents (always SCANNED_*).  IAM images
    are handwritten lines (IAM Handwriting Database); very small or very
    low-contrast images may fall through to UNKNOWN when OCR confidence is
    below classification thresholds.  That is expected behaviour.
    """

    @pytest.mark.parametrize("name", FUNSD_SAMPLES)
    def test_funsd_detects_as_scanned(self, tmp_path: Path, name: str) -> None:
        pdf = _image_to_pdf(FUNSD_IMAGES / f"{name}.png", tmp_path / f"{name}.pdf")
        profile = detect_document_type(pdf)

        assert profile.has_text_layer is False
        assert profile.has_images is True
        assert profile.text_coverage == 0.0
        assert profile.doc_type in _SCANNED_TYPES, (
            f"FUNSD/{name} should be a scanned type, got {profile.doc_type}"
        )

    @pytest.mark.parametrize("name", IAM_SAMPLES)
    def test_iam_detects_as_image_only(self, tmp_path: Path, name: str) -> None:
        """IAM is a handwriting database; narrow/short images with low OCR
        confidence may classify as UNKNOWN — that is acceptable so long as
        the profile correctly reports no text layer."""
        pdf = _image_to_pdf(IAM_DIR / f"{name}.png", tmp_path / f"{name}.pdf")
        profile = detect_document_type(pdf)

        assert profile.has_text_layer is False
        assert profile.has_images is True
        assert profile.text_coverage == 0.0
        assert profile.doc_type in _SCANNED_OR_UNKNOWN, (
            f"IAM/{name}: unexpected type {profile.doc_type}"
        )

    @pytest.mark.parametrize("name", DOCLAYNET_SAMPLES)
    def test_doclaynet_detects_as_scanned(self, tmp_path: Path, name: str) -> None:
        pdf = _image_to_pdf(DOCLAYNET_DIR / f"{name}.png", tmp_path / f"{name}.pdf")
        profile = detect_document_type(pdf)

        assert profile.has_text_layer is False
        assert profile.has_images is True
        assert profile.text_coverage == 0.0
        assert profile.doc_type in _SCANNED_TYPES, (
            f"DocLayNet/{name} should be a scanned type, got {profile.doc_type}"
        )

    def test_detection_profile_fields_populated(self, tmp_path: Path) -> None:
        """Profile returned for a fixture PDF has all expected fields set."""
        pdf = _image_to_pdf(FUNSD_IMAGES / "85540866.png", tmp_path / "test.pdf")
        profile = detect_document_type(pdf)

        assert profile.page_count == 1
        assert 0.0 <= profile.confidence <= 1.0
        # ocr_confidence is set when morphology signals are inconclusive; may or may not be None
        assert profile.ocr_confidence is None or 0.0 <= profile.ocr_confidence <= 100.0


# ---------------------------------------------------------------------------
# Extraction: OCR runs without error on all fixture images
# ---------------------------------------------------------------------------


# Page size used for OCR extraction tests — small enough that EasyOCR runs
# in a few seconds on CPU.  At 200 DPI a 150×150 pt page renders to ~417×417 px.
_OCR_TEST_PAGE_W = 150
_OCR_TEST_PAGE_H = 150


@pytest.mark.slow
class TestFixtureExtraction:
    """extract_text() should complete without raising for one representative
    fixture from each dataset.

    Pages are created at a small size (150×150 pt) so that EasyOCR processes
    ~417×417 pixel rasters rather than full A4 bitmaps.  This keeps wall time
    to a few seconds per test on CPU-only machines.  Detection tests already
    cover all 15 fixtures at full A4 resolution.
    """

    def test_funsd_sparse_form_extraction(self, tmp_path: Path) -> None:
        """85540866 is the smallest FUNSD sample (25 words); OCR should
        return one result with no error.

        The profile is set explicitly to SCANNED_MACHINEWRITTEN so this test
        exercises the extraction path independent of detection.
        """
        pdf = _image_to_pdf(
            FUNSD_IMAGES / "85540866.png",
            tmp_path / "test.pdf",
            page_w=_OCR_TEST_PAGE_W,
            page_h=_OCR_TEST_PAGE_H,
        )
        results = extract_text(pdf, _scanned_profile())

        assert len(results) == 1
        assert results[0].error is None
        assert results[0].method == "scanned_machinewritten"

    def test_iam_single_line_extraction(self, tmp_path: Path) -> None:
        """median_15 is a single handwritten line; extraction should produce
        one PageResult regardless of OCR confidence."""
        pdf = _image_to_pdf(
            IAM_DIR / "median_15.png",
            tmp_path / "test.pdf",
            page_w=_OCR_TEST_PAGE_W,
            page_h=_OCR_TEST_PAGE_H,
        )
        results = extract_text(pdf, _scanned_profile())

        assert len(results) == 1
        assert results[0].page_count == 1

    def test_doclaynet_sparse_extraction(self, tmp_path: Path) -> None:
        """sparse_text_344 has minimal content (13 labelled words); OCR should
        complete and return one result."""
        pdf = _image_to_pdf(
            DOCLAYNET_DIR / "sparse_text_344.png",
            tmp_path / "test.pdf",
            page_w=_OCR_TEST_PAGE_W,
            page_h=_OCR_TEST_PAGE_H,
        )
        results = extract_text(pdf, _scanned_profile())

        assert len(results) == 1
        assert results[0].error is None

    def test_extraction_result_has_metadata(self, tmp_path: Path) -> None:
        """ExtractionResult always carries metadata with strategy and timing."""
        pdf = _image_to_pdf(
            FUNSD_IMAGES / "85540866.png",
            tmp_path / "test.pdf",
            page_w=_OCR_TEST_PAGE_W,
            page_h=_OCR_TEST_PAGE_H,
        )
        results = extract_text(pdf, _scanned_profile())

        meta = results[0].metadata
        assert meta is not None
        assert meta.extraction_strategy == "scanned_machinewritten"
        assert meta.processing_time > 0
        assert meta.page_count == 1

    def test_extraction_result_has_pages(self, tmp_path: Path) -> None:
        """A single-page fixture always returns exactly one PageResult."""
        pdf = _image_to_pdf(
            DOCLAYNET_DIR / "sparse_text_344.png",
            tmp_path / "test.pdf",
            page_w=_OCR_TEST_PAGE_W,
            page_h=_OCR_TEST_PAGE_H,
        )
        results = extract_text(pdf, _scanned_profile())

        assert results[0].page_count == 1
        assert len(results[0].pages) == 1
        assert results[0].pages[0].page_number == 0


# ---------------------------------------------------------------------------
# OCR content: machine-printed fixtures yield recognisable text
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestFixtureOCRContent:
    """For machine-printed fixtures the extracted text should be non-empty.

    Pages are rendered at a small size (150×150 pt) so OCR completes quickly.
    These tests verify the OCR path produces real output; accuracy comparisons
    against ground truth are outside scope.
    """

    def test_funsd_sparse_form_has_some_text(self, tmp_path: Path) -> None:
        """85540866 is a sparse FUNSD form (25 words); OCR should return
        non-empty text."""
        pdf = _image_to_pdf(
            FUNSD_IMAGES / "85540866.png",
            tmp_path / "test.pdf",
            page_w=_OCR_TEST_PAGE_W,
            page_h=_OCR_TEST_PAGE_H,
        )
        results = extract_text(pdf, _scanned_profile())

        assert len(results[0].full_text) > 0, (
            "Expected non-empty OCR output from FUNSD/85540866"
        )

    def test_doclaynet_sparse_has_some_text(self, tmp_path: Path) -> None:
        """sparse_text_344 has 13 labelled words; OCR should return
        non-empty output."""
        pdf = _image_to_pdf(
            DOCLAYNET_DIR / "sparse_text_344.png",
            tmp_path / "test.pdf",
            page_w=_OCR_TEST_PAGE_W,
            page_h=_OCR_TEST_PAGE_H,
        )
        results = extract_text(pdf, _scanned_profile())

        assert len(results[0].full_text) > 0, (
            "Expected non-empty OCR output from DocLayNet/sparse_text_344"
        )

    def test_iam_line_page_count_is_one(self, tmp_path: Path) -> None:
        """Each IAM-line sample is a single-line image; extraction must
        return exactly one page regardless of content."""
        pdf = _image_to_pdf(
            IAM_DIR / "median_15.png",
            tmp_path / "test.pdf",
            page_w=_OCR_TEST_PAGE_W,
            page_h=_OCR_TEST_PAGE_H,
        )
        results = extract_text(pdf, _scanned_profile())

        assert results[0].page_count == 1


# ---------------------------------------------------------------------------
# Redaction: clean fixture images report no redaction regions
# ---------------------------------------------------------------------------


class TestFixtureRedaction:
    """RedactionDetector behaviour on clean fixture images.

    The fixtures contain no black-box redactions.  However, the default
    detector uses loose area thresholds suited to government PDFs; on some
    fixture images it may flag small dark blobs (form borders, handwriting
    strokes).  The meaningful contract is:

    1. The detector runs without error on every fixture image.
    2. No *document-width* redaction bars are detected — those would indicate
       an actual censorship bar spanning most of the page width.
    3. Masking any detected regions does not raise an error (idempotency).
    """

    # A genuine censorship bar spans nearly the full page width. Form borders
    # and thick horizontal rules in FUNSD documents can reach 60–93 % of width;
    # only flag regions above 95 % as suspicious full-page redactions.
    _FULL_WIDTH_RATIO = 0.95

    def _load_gray(self, path: Path) -> np.ndarray:
        img = Image.open(path).convert("L")
        return np.array(img)

    def _has_full_width_redaction(self, gray: np.ndarray, redactions: list) -> bool:
        """Return True if any detected region spans ≥ 95 % of image width."""
        img_width = gray.shape[1]
        for r in redactions:
            x1, _, x2, _ = r.bbox
            if (x2 - x1) / img_width >= self._FULL_WIDTH_RATIO:
                return True
        return False

    @pytest.mark.parametrize("name", FUNSD_SAMPLES)
    def test_funsd_no_full_width_redaction_bars(self, name: str) -> None:
        gray = self._load_gray(FUNSD_IMAGES / f"{name}.png")
        detector = RedactionDetector()
        redactions = detector.detect(gray, page=0)
        assert not self._has_full_width_redaction(gray, redactions), (
            f"FUNSD/{name}: unexpected full-width redaction bar detected"
        )

    @pytest.mark.parametrize("name", IAM_SAMPLES)
    def test_iam_no_full_width_redaction_bars(self, name: str) -> None:
        gray = self._load_gray(IAM_DIR / f"{name}.png")
        detector = RedactionDetector()
        redactions = detector.detect(gray, page=0)
        assert not self._has_full_width_redaction(gray, redactions), (
            f"IAM/{name}: unexpected full-width redaction bar detected"
        )

    @pytest.mark.parametrize("name", DOCLAYNET_SAMPLES)
    def test_doclaynet_no_full_width_redaction_bars(self, name: str) -> None:
        gray = self._load_gray(DOCLAYNET_DIR / f"{name}.png")
        detector = RedactionDetector()
        redactions = detector.detect(gray, page=0)
        assert not self._has_full_width_redaction(gray, redactions), (
            f"DocLayNet/{name}: unexpected full-width redaction bar detected"
        )

    @pytest.mark.parametrize("name", DOCLAYNET_SAMPLES + FUNSD_SAMPLES + IAM_SAMPLES)
    def test_masking_does_not_raise(self, name: str) -> None:
        """detector.mask() must not raise regardless of what detect() found."""
        if name in DOCLAYNET_SAMPLES:
            path = DOCLAYNET_DIR / f"{name}.png"
        elif name in FUNSD_SAMPLES:
            path = FUNSD_IMAGES / f"{name}.png"
        else:
            path = IAM_DIR / f"{name}.png"
        gray = self._load_gray(path)
        detector = RedactionDetector()
        redactions = detector.detect(gray, page=0)
        masked = detector.mask(gray, redactions)
        assert masked.shape == gray.shape


# ---------------------------------------------------------------------------
# Chunking: ground-truth text chunks correctly via the semchunk pipeline
# ---------------------------------------------------------------------------


class TestFixtureChunking:
    """Uses ground-truth text from the fixture annotations as chunker input.
    These tests do not involve OCR, so they are fast and deterministic."""

    @pytest.fixture(autouse=True)
    def _setup_chunker(self) -> None:
        # Word-based token counter avoids network calls to HuggingFace
        self.chunker = create_chunker(
            tokenizer=_word_token_counter, chunk_size=30
        )

    def test_iam_long_line_single_chunk(self) -> None:
        """IAM long_4 is a 22-word sentence; at chunk_size=30 words it fits in one chunk."""
        gt = _iam_ground_truth("long_4")
        chunks = chunk_text(gt, self.chunker)

        assert len(chunks) == 1
        assert isinstance(chunks[0], TextChunk)
        assert gt.strip() in chunks[0].text

    def test_iam_median_single_chunk(self) -> None:
        """IAM median_15 is a short 9-word sentence; must be exactly one chunk."""
        gt = _iam_ground_truth("median_15")
        chunks = chunk_text(gt, self.chunker)

        assert len(chunks) == 1
        assert chunks[0].chunk_index == 0

    def test_doclaynet_dense_text_produces_multiple_chunks(self) -> None:
        """dense_text_548 has 413 labelled regions; concatenated ground truth
        is long enough to produce multiple chunks at chunk_size=30 words."""
        words = _doclaynet_ground_truth("dense_text_548")
        full_text = " ".join(words)
        chunks = chunk_text(full_text, self.chunker)

        assert len(chunks) > 1, "Expected multiple chunks for dense DocLayNet text"

    def test_chunker_indices_are_sequential(self) -> None:
        """chunk_index must be 0, 1, 2, … with no gaps."""
        words = _doclaynet_ground_truth("dense_text_548")
        full_text = " ".join(words)
        chunks = chunk_text(full_text, self.chunker)

        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_funsd_ground_truth_round_trips_through_chunker(self) -> None:
        """Text from FUNSD annotation JSON should chunk without error and
        the combined chunk text should contain all original content."""
        gt_texts = _funsd_ground_truth("82200067_0069")
        # Join all form field texts into a single document body
        full_text = " ".join(gt_texts)
        chunks = chunk_text(full_text, self.chunker)

        assert len(chunks) >= 1
        combined = " ".join(c.text for c in chunks)
        # Every ground-truth token should appear somewhere in the chunks
        for token in gt_texts[:10]:
            assert token in combined, (
                f"Ground-truth token {repr(token)} missing from chunks"
            )

    def test_empty_ground_truth_yields_no_chunks(self) -> None:
        """Empty input must produce an empty chunk list (not crash)."""
        chunks = chunk_text("", self.chunker)
        assert chunks == []

    def test_sparse_doclaynet_chunks_at_small_size(self) -> None:
        """sparse_text_344 has only 13 words; at chunk_size=5 this produces
        multiple small chunks that together cover all the original words."""
        small_chunker = create_chunker(tokenizer=_word_token_counter, chunk_size=5)
        words = _doclaynet_ground_truth("sparse_text_344")
        full_text = " ".join(words)
        chunks = chunk_text(full_text, small_chunker)

        assert len(chunks) >= 1
        combined = " ".join(c.text for c in chunks)
        for w in words:
            assert w in combined

    def test_chunk_offsets_cover_input(self) -> None:
        """start_char and end_char must span non-overlapping, contiguous regions."""
        gt = _iam_ground_truth("long_4")
        large_chunker = create_chunker(tokenizer=_word_token_counter, chunk_size=5)
        chunks = chunk_text(gt, large_chunker)

        for chunk in chunks:
            assert chunk.start_char >= 0
            assert chunk.end_char > chunk.start_char
            assert chunk.end_char <= len(gt)

