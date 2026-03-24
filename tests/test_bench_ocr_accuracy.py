"""OCR accuracy benchmarks: WER and CER against fixture ground truth.

Run with:
    uv run --extra dev pytest -m benchmark -s -v tests/test_bench_ocr_accuracy.py

The -s flag is required to see the per-sample metric table printed to stdout.
These tests are excluded from the normal test suite (no -m benchmark flag).

Page sizes are computed to preserve each image's aspect ratio at a target
width of 595 pt (A4 width), so OCR sees the same proportions as the source.
IAM-line images are ~128px tall, so their pages are very short — correct
behaviour for single-line recognition.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import fitz
import pytest
from PIL import Image

from womblex.ingest.detect import DocumentProfile, DocumentType
from womblex.ingest.extract import extract_text
from womblex.utils.metrics import cer, wer

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "fixtures"
FUNSD_IMAGES = FIXTURES_DIR / "funsd" / "images"
FUNSD_ANNOTATIONS = FIXTURES_DIR / "funsd" / "annotations"
IAM_DIR = FIXTURES_DIR / "iam_line"
DOCLAYNET_DIR = FIXTURES_DIR / "doclaynet"

# Standard page width in points — height is computed per image to preserve AR.
_TARGET_W = 595


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scanned_profile() -> DocumentProfile:
    return DocumentProfile(
        doc_type=DocumentType.SCANNED_MACHINEWRITTEN,
        page_count=1,
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


def _ar_page_size(image_path: Path, target_w: int = _TARGET_W) -> tuple[int, int]:
    """Return (width, height) in points that preserves the image aspect ratio."""
    img = Image.open(image_path)
    w, h = img.size
    page_h = max(1, round(target_w * h / w))
    return target_w, page_h


def _image_to_pdf(image_path: Path, output_path: Path) -> Path:
    """Wrap a PNG in a single-page PDF, preserving aspect ratio at _TARGET_W pt."""
    page_w, page_h = _ar_page_size(image_path)
    img = Image.open(image_path).convert("RGB")
    doc = fitz.open()
    page = doc.new_page(width=page_w, height=page_h)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    page.insert_image(fitz.Rect(0, 0, page_w, page_h), stream=buf.getvalue())
    doc.save(str(output_path))
    doc.close()
    return output_path


def _iam_gt(name: str) -> str:
    return (IAM_DIR / f"{name}.gt.txt").read_text().strip()


def _funsd_gt(name: str) -> str:
    with open(FUNSD_ANNOTATIONS / f"{name}.json") as f:
        data = json.load(f)
    return " ".join(e["text"].strip() for e in data["form"] if e["text"].strip())


def _doclaynet_gt(name: str) -> str:
    with open(DOCLAYNET_DIR / f"{name}.json") as f:
        data = json.load(f)
    return " ".join(w.strip() for w in data["words"] if w.strip())


def _run_ocr(image_path: Path, tmp_path: Path) -> str:
    """Return full extracted text for an image fixture."""
    pdf = _image_to_pdf(image_path, tmp_path / f"{image_path.stem}.pdf")
    results = extract_text(pdf, _scanned_profile())
    return results[0].full_text if results else ""


def _fmt(label: str, hyp: str, ref: str) -> str:
    w = wer(ref, hyp)
    c = cer(ref, hyp)
    ref_words = len(ref.split())
    hyp_words = len(hyp.split())
    return (
        f"  {label:<35}  WER={w:.3f}  CER={c:.3f}"
        f"  ref={ref_words}w  hyp={hyp_words}w"
    )


# ---------------------------------------------------------------------------
# IAM-line benchmarks
# ---------------------------------------------------------------------------

IAM_SAMPLES = [
    "short_1602",
    "median_15",
    "long_4",
    "wide_1739",
    "narrow_1163",
]


@pytest.mark.benchmark
class TestIAMAccuracy:
    """Single-line handwriting recognition accuracy vs IAM ground truth.

    IAM images are 128px tall single-line strips.  EasyOCR is a general-purpose
    engine, not a specialist HTR model, so WER > 1.0 is expected on handwriting.
    These numbers establish a baseline; improvements to pre-processing should
    move them down.
    """

    @pytest.mark.parametrize("name", IAM_SAMPLES)
    def test_iam_wer_cer(self, tmp_path: Path, name: str) -> None:
        ref = _iam_gt(name)
        hyp = _run_ocr(IAM_DIR / f"{name}.png", tmp_path)
        w = wer(ref, hyp)
        c = cer(ref, hyp)
        print(f"\n{_fmt(f'IAM/{name}', hyp, ref)}")
        print(f"    ref: {ref!r}")
        print(f"    hyp: {hyp!r}")
        # Sanity only — we're measuring, not enforcing accuracy
        assert w >= 0.0
        assert c >= 0.0


# ---------------------------------------------------------------------------
# FUNSD benchmarks
# ---------------------------------------------------------------------------

FUNSD_SAMPLES = [
    "85540866",
    "82200067_0069",
    "87594142_87594144",
    "87528321",
    "87528380",
]


@pytest.mark.benchmark
class TestFUNSDAccuracy:
    """Machine-printed form OCR accuracy vs FUNSD annotation ground truth.

    Ground truth is the concatenation of all non-empty form field texts.
    FUNSD forms have scattered layout (labels, boxes, lines) which challenges
    reading-order recovery; WER reflects both recognition errors and ordering.
    """

    @pytest.mark.parametrize("name", FUNSD_SAMPLES)
    def test_funsd_wer_cer(self, tmp_path: Path, name: str) -> None:
        ref = _funsd_gt(name)
        hyp = _run_ocr(FUNSD_IMAGES / f"{name}.png", tmp_path)
        w = wer(ref, hyp)
        c = cer(ref, hyp)
        print(f"\n{_fmt(f'FUNSD/{name}', hyp, ref)}")
        assert w >= 0.0
        assert c >= 0.0


# ---------------------------------------------------------------------------
# DocLayNet benchmarks
# ---------------------------------------------------------------------------

DOCLAYNET_SAMPLES = [
    "diverse_layout_49",
    "table_0",
    "formula_29",
    "sparse_text_344",
    "dense_text_548",
]


@pytest.mark.benchmark
class TestDocLayNetAccuracy:
    """Machine-printed document OCR accuracy vs DocLayNet word-level ground truth.

    Ground truth is the word list joined with spaces.  DocLayNet covers diverse
    layout types (tables, formulas, dense text) which stress different aspects
    of the extraction pipeline.
    """

    @pytest.mark.parametrize("name", DOCLAYNET_SAMPLES)
    def test_doclaynet_wer_cer(self, tmp_path: Path, name: str) -> None:
        ref = _doclaynet_gt(name)
        hyp = _run_ocr(DOCLAYNET_DIR / f"{name}.png", tmp_path)
        w = wer(ref, hyp)
        c = cer(ref, hyp)
        print(f"\n{_fmt(f'DocLayNet/{name}', hyp, ref)}")
        assert w >= 0.0
        assert c >= 0.0
