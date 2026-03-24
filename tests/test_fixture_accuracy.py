"""Fixture accuracy benchmarks for OCR, layout analysis, and form extraction.

Runs the rapidocr-onnxruntime pipeline against curated fixture datasets
(FUNSD, IAM handwriting, DocLayNet) and reports CER, WER, detection rates,
and layout label accuracy.  Each dataset is tested both **raw** (PaddleOCR
alone) and **preprocessed** (deskew + binarise — the same
pipeline used by the extraction strategies).

Results are written to ``docs/accuracy/EXTRACTION.md``.

Usage:
    pytest tests/test_fixture_accuracy.py -s --tb=short
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pytest

from womblex.ingest.paddle_ocr import (
    LayoutRegion,
    get_layout_analyzer,
    get_paddle_reader,
    preprocess_for_ocr,
)

logger = logging.getLogger(__name__)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "fixtures"
FUNSD_DIR = FIXTURES / "funsd"
IAM_DIR = FIXTURES / "iam_line"
DOCLAYNET_DIR = FIXTURES / "doclaynet"
WOMBLEX_DIR = FIXTURES / "womblex-collection"

# Womblex fixture paths — PDFs/DOCX with human-proofread transcripts.
_WOMBLEX_FIXTURES: list[dict[str, str]] = [
    {
        "name": "Throsby",
        "file": "_documents/00768-213A-270825-Throsby-Out-of-School-Care-"
                "Administrative-Decision-Other-Notice-and-Direction_Redacted.pdf",
        "transcript": "_documents/00768-213A-270825-Throsby-Out-of-School-Care-"
                      "Administrative-Decision-Other-Notice-and-Direction_Redacted_transcript.txt",
    },
    {
        "name": "Auditor-General",
        "file": "_documents/Auditor-General_Report_2020-21_19.pdf",
        "transcript": "_documents/Auditor-General_Report_2020-21_19_transcript.txt",
    },
    {
        "name": "DFAT-Corporate-Plan",
        "file": "_documents/dfat-corporate-plan-2025-26.docx",
        "transcript": "_documents/dfat-corporate-plan-2025-26_transcript.txt",
    },
]

# DocLayNet integer label → string
DOCLAYNET_LABELS: dict[int, str] = {
    1: "Caption",
    2: "Footnote",
    3: "Formula",
    4: "List-item",
    5: "Page-footer",
    6: "Page-header",
    7: "Picture",
    8: "Section-header",
    9: "Table",
    10: "Text",
    11: "Title",
}

# Map DocLayNet labels to womblex block_type equivalents
DOCLAYNET_TO_WOMBLEX: dict[str, str] = {
    "Caption": "caption",
    "Footnote": "paragraph",
    "Formula": "formula",
    "List-item": "list_item",
    "Page-footer": "footer",
    "Page-header": "header",
    "Picture": "figure",
    "Section-header": "heading",
    "Table": "table",
    "Text": "paragraph",
    "Title": "heading",
}


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation edges."""
    import re

    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def char_error_rate(predicted: str, reference: str) -> float:
    """Levenshtein-based character error rate (CER)."""
    pred = _normalise(predicted)
    ref = _normalise(reference)
    if not ref:
        return 0.0 if not pred else 1.0
    d = _levenshtein(pred, ref)
    return d / len(ref)


def word_error_rate(predicted: str, reference: str) -> float:
    """Word error rate (WER)."""
    pred_words = _normalise(predicted).split()
    ref_words = _normalise(reference).split()
    if not ref_words:
        return 0.0 if not pred_words else 1.0
    d = _levenshtein_seq(pred_words, ref_words)
    return d / len(ref_words)


def _levenshtein(s1: str, s2: str) -> int:
    """Levenshtein distance between two strings (uses rapidfuzz C backend)."""
    from rapidfuzz.distance import Levenshtein
    return Levenshtein.distance(s1, s2)


def _levenshtein_seq(s1: list[str], s2: list[str]) -> int:
    """Levenshtein distance between two word sequences."""
    from rapidfuzz.distance import Levenshtein
    return Levenshtein.distance(s1, s2)


def _spatial_sort_text(
    words_with_boxes: list[tuple[str, tuple[float, float, float, float]]],
    line_tolerance: float = 0.5,
) -> str:
    """Sort words by spatial position and join into a string.

    Sorts by vertical centroid first (top-to-bottom), then horizontal centroid
    (left-to-right) within the same line. Two words are considered on the same
    line when their vertical centroids are within ``line_tolerance`` times the
    average word height.

    This is a general-purpose utility: any OCR output or annotation set with
    per-word bounding boxes can be spatially reordered to separate recognition
    accuracy from reading-order accuracy.

    Args:
        words_with_boxes: List of (text, (x0, y0, x1, y1)) tuples.
        line_tolerance: Fraction of average word height used to group words
            into the same line.

    Returns:
        Space-joined string of words in spatial order.
    """
    if not words_with_boxes:
        return ""

    # Calculate centroids
    items: list[tuple[str, float, float]] = []
    heights: list[float] = []
    for text, (x0, y0, x1, y1) in words_with_boxes:
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        items.append((text, cx, cy))
        heights.append(y1 - y0)

    avg_height = sum(heights) / len(heights) if heights else 1.0
    threshold = avg_height * line_tolerance

    # Sort by y-centroid first, then x-centroid within same line
    items.sort(key=lambda t: (t[2], t[1]))

    # Group into lines: consecutive items within threshold of first item in group
    lines: list[list[tuple[str, float, float]]] = []
    current_line: list[tuple[str, float, float]] = [items[0]]
    for item in items[1:]:
        if abs(item[2] - current_line[0][2]) <= threshold:
            current_line.append(item)
        else:
            lines.append(current_line)
            current_line = [item]
    lines.append(current_line)

    # Sort each line by x-centroid, then join
    sorted_words: list[str] = []
    for line in lines:
        line.sort(key=lambda t: t[1])
        sorted_words.extend(word for word, _, _ in line)

    return " ".join(sorted_words)


def iou(box_a: tuple[float, ...], box_b: tuple[float, ...]) -> float:
    """Intersection-over-union for two axis-aligned boxes (x0,y0,x1,y1)."""
    x0 = max(box_a[0], box_b[0])
    y0 = max(box_a[1], box_b[1])
    x1 = min(box_a[2], box_b[2])
    y1 = min(box_a[3], box_b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Result accumulator
# ---------------------------------------------------------------------------

_results: dict[str, list[dict]] = {
    "funsd_raw": [], "funsd_preprocessed": [],
    "iam": [],
    "doclaynet_raw": [], "doclaynet_preprocessed": [],
    "womblex": [],
}


def _ocr_image(reader: object, img: np.ndarray) -> tuple[str, float, int]:
    """Run OCR on an image, return (text, avg_confidence, num_regions)."""
    ocr_results = reader.readtext(img)  # type: ignore[union-attr]
    text = " ".join(t for _, t, _ in ocr_results)
    avg_conf = sum(c for _, _, c in ocr_results) / len(ocr_results) if ocr_results else 0.0
    return text, avg_conf, len(ocr_results)


def _ocr_words_with_boxes(
    ocr_results: list[tuple],
) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Convert OCR results to (word, bbox) tuples for spatial sorting.

    Each OCR region may contain multiple words. This splits them and assigns
    proportional bounding boxes so spatial sorting works at word level.
    """
    words_with_boxes: list[tuple[str, tuple[float, float, float, float]]] = []
    for bbox, text, _ in ocr_results:
        region_words = text.split()
        if not region_words:
            continue
        # bbox from rapidocr: [[x0,y0],[x1,y1],[x2,y2],[x3,y3]] (quadrilateral)
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        if len(region_words) == 1:
            words_with_boxes.append((region_words[0], (x0, y0, x1, y1)))
        else:
            # Split region horizontally across words proportional to char count
            total_chars = sum(len(w) for w in region_words)
            cur_x = x0
            for w in region_words:
                frac = len(w) / total_chars
                w_x1 = cur_x + frac * (x1 - x0)
                words_with_boxes.append((w, (cur_x, y0, w_x1, y1)))
                cur_x = w_x1
    return words_with_boxes


def _aggregate_doclaynet_blocks(
    bboxes: list[list[float]], labels: list[str],
) -> list[tuple[tuple[float, float, float, float], str]]:
    """Merge adjacent DocLayNet text spans with the same label into blocks.

    DocLayNet annotates per text-line; layout models predict page-level blocks.
    This groups consecutive spans sharing a label into a single bounding box.
    """
    if not bboxes:
        return []

    blocks: list[tuple[tuple[float, float, float, float], str]] = []
    cur_label = labels[0]
    x0, y0, x1, y1 = bboxes[0]

    for i in range(1, len(bboxes)):
        if labels[i] == cur_label:
            # Extend the current block
            bx0, by0, bx1, by1 = bboxes[i]
            x0 = min(x0, bx0)
            y0 = min(y0, by0)
            x1 = max(x1, bx1)
            y1 = max(y1, by1)
        else:
            blocks.append(((x0, y0, x1, y1), cur_label))
            cur_label = labels[i]
            x0, y0, x1, y1 = bboxes[i]

    blocks.append(((x0, y0, x1, y1), cur_label))
    return blocks


# ---------------------------------------------------------------------------
# FUNSD — form OCR accuracy
# ---------------------------------------------------------------------------


class TestFUNSD:
    """OCR accuracy on FUNSD form images — raw and preprocessed."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.reader = get_paddle_reader(lang="eng")

    @pytest.mark.parametrize(
        "stem",
        ["85540866", "82200067_0069", "87594142_87594144", "87528321", "87528380"],
    )
    def test_funsd_ocr(self, stem: str) -> None:
        img_path = FUNSD_DIR / "images" / f"{stem}.png"
        ann_path = FUNSD_DIR / "annotations" / f"{stem}.json"
        if not img_path.exists():
            pytest.skip(f"Fixture missing: {img_path}")

        img = cv2.imread(str(img_path))
        assert img is not None, f"Failed to load {img_path}"

        annotation = json.loads(ann_path.read_text())
        gt_words: list[str] = []
        gt_words_with_boxes: list[tuple[str, tuple[float, float, float, float]]] = []
        gt_fields: int = len(annotation["form"])
        for field in annotation["form"]:
            for w in field.get("words", []):
                gt_words.append(w["text"])
                box = w["box"]  # [x0, y0, x1, y1]
                gt_words_with_boxes.append((w["text"], (box[0], box[1], box[2], box[3])))
        gt_text = " ".join(gt_words)
        gt_text_sorted = _spatial_sort_text(gt_words_with_boxes)

        # --- Raw OCR ---
        raw_ocr_results = self.reader.readtext(img)
        raw_text = " ".join(t for _, t, _ in raw_ocr_results)
        raw_conf = sum(c for _, _, c in raw_ocr_results) / len(raw_ocr_results) if raw_ocr_results else 0.0
        raw_cer = char_error_rate(raw_text, gt_text)
        raw_wer = word_error_rate(raw_text, gt_text)
        # Sorted CER: spatially sort both GT and OCR words
        raw_words_with_boxes = _ocr_words_with_boxes(raw_ocr_results)
        raw_text_sorted = _spatial_sort_text(raw_words_with_boxes)
        raw_cer_sorted = char_error_rate(raw_text_sorted, gt_text_sorted)

        _results["funsd_raw"].append({
            "file": stem, "gt_words": len(gt_words), "gt_fields": gt_fields,
            "detected_regions": len(raw_ocr_results), "cer": raw_cer, "wer": raw_wer,
            "avg_confidence": raw_conf, "cer_sorted": raw_cer_sorted,
        })

        # --- Preprocessed OCR (deskew + binarise) ---
        preprocessed, steps = preprocess_for_ocr(img)
        pp_ocr_results = self.reader.readtext(preprocessed)
        pp_text = " ".join(t for _, t, _ in pp_ocr_results)
        pp_conf = sum(c for _, _, c in pp_ocr_results) / len(pp_ocr_results) if pp_ocr_results else 0.0
        pp_cer = char_error_rate(pp_text, gt_text)
        pp_wer = word_error_rate(pp_text, gt_text)
        pp_words_with_boxes = _ocr_words_with_boxes(pp_ocr_results)
        pp_text_sorted = _spatial_sort_text(pp_words_with_boxes)
        pp_cer_sorted = char_error_rate(pp_text_sorted, gt_text_sorted)

        _results["funsd_preprocessed"].append({
            "file": stem, "gt_words": len(gt_words), "gt_fields": gt_fields,
            "detected_regions": len(pp_ocr_results), "cer": pp_cer, "wer": pp_wer,
            "avg_confidence": pp_conf, "steps": steps, "cer_sorted": pp_cer_sorted,
        })

        logger.info(
            "FUNSD %s: raw CER=%.3f→%.3f  sorted CER=%.3f→%.3f  steps=%s",
            stem, raw_cer, pp_cer, raw_cer_sorted, pp_cer_sorted, "+".join(steps),
        )
        assert True


# ---------------------------------------------------------------------------
# IAM — handwriting OCR accuracy
# ---------------------------------------------------------------------------


class TestIAM:
    """OCR accuracy on IAM handwriting line images."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.reader = get_paddle_reader(lang="eng")

    @pytest.mark.parametrize(
        "stem",
        ["short_1602", "median_15", "long_4", "wide_1739", "narrow_1163"],
    )
    def test_iam_line(self, stem: str) -> None:
        img_path = IAM_DIR / f"{stem}.png"
        gt_path = IAM_DIR / f"{stem}.gt.txt"
        if not img_path.exists():
            pytest.skip(f"Fixture missing: {img_path}")

        img = cv2.imread(str(img_path))
        assert img is not None, f"Failed to load {img_path}"

        gt_text = gt_path.read_text().strip()

        ocr_results = self.reader.readtext(img)
        pred_text = " ".join(text for _, text, _ in ocr_results)

        cer = char_error_rate(pred_text, gt_text)
        wer = word_error_rate(pred_text, gt_text)
        avg_conf = (
            sum(c for _, _, c in ocr_results) / len(ocr_results) if ocr_results else 0.0
        )

        _results["iam"].append(
            {
                "file": stem,
                "gt_text": gt_text,
                "pred_text": pred_text,
                "gt_words": len(gt_text.split()),
                "cer": cer,
                "wer": wer,
                "avg_confidence": avg_conf,
            }
        )

        logger.info(
            "IAM %s: CER=%.3f WER=%.3f conf=%.2f | gt=%r pred=%r",
            stem, cer, wer, avg_conf, gt_text[:60], pred_text[:60],
        )
        assert True


# ---------------------------------------------------------------------------
# DocLayNet — layout analysis accuracy
# ---------------------------------------------------------------------------


def _match_layout_regions(
    layout_regions: list[LayoutRegion],
    gt_blocks: list[tuple[tuple[float, float, float, float], str]],
    iou_threshold: float = 0.3,
) -> dict[str, dict[str, int]]:
    """Match predicted layout regions to GT blocks by IoU.

    Returns per-class counts keyed by womblex label::

        {"paragraph": {"tp": 2, "fp": 1, "fn": 3}, ...}

    A true positive requires both IoU >= threshold and correct label mapping.
    A false positive is a prediction that either didn't match any GT block or
    matched with the wrong label. A false negative is a GT block with no
    matching prediction.
    """
    used_gt: set[int] = set()
    # Collect all labels from both sides
    gt_womblex_labels = [
        DOCLAYNET_TO_WOMBLEX.get(lbl, "paragraph") for _, lbl in gt_blocks
    ]
    all_labels: set[str] = set(gt_womblex_labels)
    all_labels.update(r.block_type for r in layout_regions)

    counts: dict[str, dict[str, int]] = {
        lbl: {"tp": 0, "fp": 0, "fn": 0} for lbl in all_labels
    }

    for pred_region in layout_regions:
        best_iou = 0.0
        best_gt_idx = -1
        for gi, (gt_box, _) in enumerate(gt_blocks):
            if gi in used_gt:
                continue
            score = iou(pred_region.bbox, gt_box)
            if score > best_iou:
                best_iou = score
                best_gt_idx = gi

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            used_gt.add(best_gt_idx)
            expected_type = gt_womblex_labels[best_gt_idx]
            if pred_region.block_type == expected_type:
                counts[expected_type]["tp"] += 1
            else:
                # Matched spatially but wrong label
                counts[pred_region.block_type]["fp"] += 1
                counts[expected_type]["fn"] += 1
        else:
            # No spatial match
            counts[pred_region.block_type]["fp"] += 1

    # Unmatched GT blocks are false negatives
    for gi, (_, _) in enumerate(gt_blocks):
        if gi not in used_gt:
            counts[gt_womblex_labels[gi]]["fn"] += 1

    return counts


class TestDocLayNet:
    """Layout label accuracy on DocLayNet page images."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.analyzer = get_layout_analyzer()
        self.reader = get_paddle_reader(lang="eng")

    @pytest.mark.parametrize(
        "stem",
        ["dense_text_548", "diverse_layout_49", "sparse_text_344", "formula_29", "table_0"],
    )
    def test_doclaynet_layout(self, stem: str) -> None:
        img_path = DOCLAYNET_DIR / f"{stem}.png"
        ann_path = DOCLAYNET_DIR / f"{stem}.json"
        if not img_path.exists():
            pytest.skip(f"Fixture missing: {img_path}")

        img = cv2.imread(str(img_path))
        assert img is not None, f"Failed to load {img_path}"

        annotation = json.loads(ann_path.read_text())
        gt_labels_raw = [DOCLAYNET_LABELS.get(lbl, "Unknown") for lbl in annotation["labels"]]
        gt_bboxes_raw = annotation["bboxes"]
        gt_words = annotation["words"]
        gt_text = " ".join(gt_words)

        gt_blocks = _aggregate_doclaynet_blocks(gt_bboxes_raw, gt_labels_raw)

        gt_label_counts: dict[str, int] = {}
        for _, lbl in gt_blocks:
            gt_label_counts[lbl] = gt_label_counts.get(lbl, 0) + 1

        # Layout analysis (same for both passes — operates on raw image)
        layout_regions: list[LayoutRegion] = self.analyzer.analyze(img)

        pred_label_counts: dict[str, int] = {}
        for r in layout_regions:
            pred_label_counts[r.block_type] = pred_label_counts.get(r.block_type, 0) + 1

        per_class = _match_layout_regions(layout_regions, gt_blocks)
        total_tp = sum(c["tp"] for c in per_class.values())
        total_fp = sum(c["fp"] for c in per_class.values())
        total_fn = sum(c["fn"] for c in per_class.values())
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        # --- Raw OCR ---
        raw_text, _, _ = _ocr_image(self.reader, img)
        raw_cer = char_error_rate(raw_text, gt_text)
        raw_wer = word_error_rate(raw_text, gt_text)

        _results["doclaynet_raw"].append({
            "file": stem, "gt_blocks": len(gt_blocks), "gt_words": len(gt_words),
            "pred_layout_regions": len(layout_regions),
            "precision": precision, "recall": recall, "f1": f1,
            "per_class": per_class,
            "gt_label_counts": gt_label_counts, "pred_label_counts": pred_label_counts,
            "cer": raw_cer, "wer": raw_wer,
        })

        # --- Preprocessed OCR (deskew + binarise) ---
        preprocessed, steps = preprocess_for_ocr(img)
        pp_text, _, _ = _ocr_image(self.reader, preprocessed)
        pp_cer = char_error_rate(pp_text, gt_text)
        pp_wer = word_error_rate(pp_text, gt_text)

        _results["doclaynet_preprocessed"].append({
            "file": stem, "gt_blocks": len(gt_blocks), "gt_words": len(gt_words),
            "pred_layout_regions": len(layout_regions),
            "precision": precision, "recall": recall, "f1": f1,
            "per_class": per_class,
            "gt_label_counts": gt_label_counts, "pred_label_counts": pred_label_counts,
            "cer": pp_cer, "wer": pp_wer, "steps": steps,
        })

        logger.info(
            "DocLayNet %s: P=%.1f%% R=%.1f%% F1=%.1f%% CER raw=%.3f→pp=%.3f steps=%s",
            stem, precision * 100, recall * 100, f1 * 100, raw_cer, pp_cer,
            "+".join(steps),
        )
        assert True


# ---------------------------------------------------------------------------
# Womblex-collection — extraction fidelity against human transcripts
# ---------------------------------------------------------------------------


class TestWomblexExtraction:
    """Extraction accuracy on womblex-collection fixtures with transcripts."""

    @pytest.mark.parametrize(
        "fixture",
        _WOMBLEX_FIXTURES,
        ids=[f["name"] for f in _WOMBLEX_FIXTURES],
    )
    def test_womblex_extraction_cer(self, fixture: dict[str, str]) -> None:
        file_path = WOMBLEX_DIR / fixture["file"]
        transcript_path = WOMBLEX_DIR / fixture["transcript"]
        if not file_path.exists():
            pytest.skip(f"Fixture missing: {file_path}")
        if not transcript_path.exists():
            pytest.skip(f"Transcript missing: {transcript_path}")

        from womblex.ingest.detect import DetectionConfig, detect_file_type
        from womblex.ingest.extract import extract_text

        import fitz as _fitz

        profile = detect_file_type(file_path, DetectionConfig())
        _max_pages = 30

        # Determine total page count for truncation ratio
        if file_path.suffix.lower() == ".pdf":
            _doc = _fitz.open(str(file_path))
            total_pages = _doc.page_count
            _doc.close()
        else:
            total_pages = None

        results = extract_text(file_path, profile, max_pages=_max_pages)
        extracted = results[0].full_text

        gt_text = transcript_path.read_text(encoding="utf-8").strip()

        # Truncate GT proportionally when pages were limited
        if total_pages and total_pages > _max_pages:
            ratio = _max_pages / total_pages
            gt_text = gt_text[:int(len(gt_text) * ratio)]

        cer = char_error_rate(extracted, gt_text)
        wer = word_error_rate(extracted, gt_text)

        _results["womblex"].append({
            "name": fixture["name"],
            "file": file_path.name,
            "doc_type": str(profile.doc_type.value) if hasattr(profile.doc_type, "value") else str(profile.doc_type),
            "pages": len(results[0].pages),
            "total_pages": total_pages or len(results[0].pages),
            "cer": cer,
            "wer": wer,
            "extracted_chars": len(extracted),
            "gt_chars": len(gt_text),
        })

        logger.info(
            "Womblex %s: CER=%.3f WER=%.3f (extracted=%d chars, GT=%d chars)",
            fixture["name"], cer, wer, len(extracted), len(gt_text),
        )
        assert True


# ---------------------------------------------------------------------------
# Report generation — runs after all tests
# ---------------------------------------------------------------------------


def _generate_report() -> str:
    """Build the accuracy.md content from accumulated results."""
    lines: list[str] = []
    from datetime import date

    lines.append("# Extraction Accuracy Report")
    lines.append("")
    lines.append("Generated by `test_fixture_accuracy.py`.")
    lines.append("")
    lines.append(f"**Date:** {date.today().isoformat()}")
    lines.append("**Engine:** rapidocr-onnxruntime (PaddleOCR v4 ONNX)")
    lines.append("")

    # --- FUNSD ---
    lines.append("## FUNSD — Form OCR")
    lines.append("")

    funsd_raw = _results["funsd_raw"]
    funsd_pp = _results["funsd_preprocessed"]
    if funsd_raw:
        lines.append("| File | GT Words | CER (raw) | CER (pp) | CER-s (raw) | CER-s (pp) | WER (raw) | WER (pp) | Conf (raw) | Conf (pp) | Steps |")
        lines.append("|------|----------|-----------|----------|-------------|------------|-----------|----------|------------|-----------|-------|")
        for raw, pp in zip(funsd_raw, funsd_pp):
            lines.append(
                f"| `{raw['file']}` | {raw['gt_words']} "
                f"| {raw['cer']:.3f} | {pp['cer']:.3f} "
                f"| {raw['cer_sorted']:.3f} | {pp['cer_sorted']:.3f} "
                f"| {raw['wer']:.3f} | {pp['wer']:.3f} "
                f"| {raw['avg_confidence']:.2f} | {pp['avg_confidence']:.2f} "
                f"| {'+'.join(pp.get('steps', []))} |"
            )
        def _a(key: str, data: list) -> float:
            return sum(r[key] for r in data) / len(data)
        lines.append(
            f"| **Average** | "
            f"| **{_a('cer', funsd_raw):.3f}** | **{_a('cer', funsd_pp):.3f}** "
            f"| **{_a('cer_sorted', funsd_raw):.3f}** | **{_a('cer_sorted', funsd_pp):.3f}** "
            f"| **{_a('wer', funsd_raw):.3f}** | **{_a('wer', funsd_pp):.3f}** "
            f"| **{_a('avg_confidence', funsd_raw):.2f}** | **{_a('avg_confidence', funsd_pp):.2f}** | |"
        )
    else:
        lines.append("*No FUNSD results collected.*")
    lines.append("")

    # --- IAM ---
    lines.append("## IAM — Handwriting OCR")
    lines.append("")

    iam = _results["iam"]
    if iam:
        lines.append("| File | GT Words | CER | WER | Avg Confidence | Ground Truth | Prediction |")
        lines.append("|------|----------|-----|-----|----------------|-------------|------------|")
        for r in iam:
            gt_short = r["gt_text"][:40] + ("..." if len(r["gt_text"]) > 40 else "")
            pred_short = r["pred_text"][:40] + ("..." if len(r["pred_text"]) > 40 else "")
            lines.append(
                f"| `{r['file']}` | {r['gt_words']} "
                f"| {r['cer']:.3f} | {r['wer']:.3f} | {r['avg_confidence']:.2f} "
                f"| {gt_short} | {pred_short} |"
            )
        avg_cer = sum(r["cer"] for r in iam) / len(iam)
        avg_wer = sum(r["wer"] for r in iam) / len(iam)
        avg_conf = sum(r["avg_confidence"] for r in iam) / len(iam)
        lines.append(
            f"| **Average** | | **{avg_cer:.3f}** | **{avg_wer:.3f}** | **{avg_conf:.2f}** | | |"
        )
    else:
        lines.append("*No IAM results collected.*")
    lines.append("")

    # --- DocLayNet ---
    lines.append("## DocLayNet — Layout Analysis")
    lines.append("")

    dln_raw = _results["doclaynet_raw"]
    dln_pp = _results["doclaynet_preprocessed"]
    if dln_raw:
        lines.append("| File | GT Blocks | Pred | Precision | Recall | F1 | CER (raw) | CER (pp) | WER (raw) | WER (pp) | Steps |")
        lines.append("|------|----------|------|-----------|--------|-----|-----------|----------|-----------|----------|-------|")
        for raw, pp in zip(dln_raw, dln_pp):
            lines.append(
                f"| `{raw['file']}` | {raw['gt_blocks']} | {raw['pred_layout_regions']} "
                f"| {raw['precision']:.1%} | {raw['recall']:.1%} | {raw['f1']:.1%} "
                f"| {raw['cer']:.3f} | {pp['cer']:.3f} "
                f"| {raw['wer']:.3f} | {pp['wer']:.3f} "
                f"| {'+'.join(pp.get('steps', []))} |"
            )

        def _a(key: str, data: list) -> float:
            return sum(r[key] for r in data) / len(data)
        lines.append(
            f"| **Average** | | "
            f"| **{_a('precision', dln_raw):.1%}** | **{_a('recall', dln_raw):.1%}** | **{_a('f1', dln_raw):.1%}** "
            f"| **{_a('cer', dln_raw):.3f}** | **{_a('cer', dln_pp):.3f}** "
            f"| **{_a('wer', dln_raw):.3f}** | **{_a('wer', dln_pp):.3f}** | |"
        )

        # Per-class P/R/F1
        lines.append("")
        lines.append("### Per-Class Layout Detection")
        lines.append("")
        # Aggregate per-class counts across all fixtures
        agg_class: dict[str, dict[str, int]] = {}
        for r in dln_raw:
            for lbl, cnts in r["per_class"].items():
                if lbl not in agg_class:
                    agg_class[lbl] = {"tp": 0, "fp": 0, "fn": 0}
                agg_class[lbl]["tp"] += cnts["tp"]
                agg_class[lbl]["fp"] += cnts["fp"]
                agg_class[lbl]["fn"] += cnts["fn"]
        lines.append("| Label | TP | FP | FN | Precision | Recall | F1 |")
        lines.append("|-------|----|----|-----|-----------|--------|-----|")
        for lbl in sorted(agg_class, key=lambda x: -(agg_class[x]["tp"] + agg_class[x]["fn"])):
            c = agg_class[lbl]
            p = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
            r = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
            f = 2 * p * r / (p + r) if (p + r) else 0.0
            lines.append(f"| {lbl} | {c['tp']} | {c['fp']} | {c['fn']} | {p:.1%} | {r:.1%} | {f:.1%} |")

    else:
        lines.append("*No DocLayNet results collected.*")
    lines.append("")

    # --- Womblex-collection ---
    lines.append("## Womblex-Collection — Extraction Fidelity")
    lines.append("")

    womblex = _results["womblex"]
    if womblex:
        lines.append("| Fixture | Doc Type | Pages (eval/total) | CER | WER | Extracted Chars | GT Chars |")
        lines.append("|---------|----------|--------------------|-----|-----|-----------------|----------|")
        for r in womblex:
            total = r.get("total_pages", r["pages"])
            page_str = f"{r['pages']}/{total}" if total != r["pages"] else str(r["pages"])
            lines.append(
                f"| `{r['name']}` | {r['doc_type']} | {page_str} "
                f"| {r['cer']:.3f} | {r['wer']:.3f} "
                f"| {r['extracted_chars']:,} | {r['gt_chars']:,} |"
            )
        if len(womblex) > 1:
            def _a2(key: str) -> float:
                return sum(r[key] for r in womblex) / len(womblex)
            lines.append(
                f"| **Average** | | | **{_a2('cer'):.3f}** | **{_a2('wer'):.3f}** | | |"
            )
    else:
        lines.append("*No womblex-collection results collected.*")
    lines.append("")

    # --- Metric definitions ---
    lines.append("## Metric Definitions")
    lines.append("")
    lines.append("- **(raw)** = PaddleOCR on original image; **(pp)** = after preprocessing (deskew + binarise).")
    lines.append("- **CER**: Levenshtein distance / reference length. 0.0 = perfect.")
    lines.append("- **CER-s**: GT and OCR words spatially sorted by bounding-box centroid before CER. Isolates recognition from reading-order.")
    lines.append("- **WER**: Word-level Levenshtein / reference word count.")
    lines.append("- **Precision**: Predicted layout blocks matching a GT block with correct label (IoU >= 0.3).")
    lines.append("- **Recall**: GT layout blocks matched by a prediction with correct label (IoU >= 0.3).")
    lines.append("- **F1**: Harmonic mean of precision and recall.")
    lines.append("- **Avg Confidence**: Mean per-region OCR confidence from RapidOCR (0\u20131).")
    lines.append("")

    return "\n".join(lines)


@pytest.fixture(scope="session", autouse=True)
def write_report(request: pytest.FixtureRequest) -> None:
    """Write accuracy/EXTRACTION.md after all tests complete."""

    def _finalise() -> None:
        report = _generate_report()
        out = Path(__file__).resolve().parent.parent / "docs" / "accuracy" / "EXTRACTION.md"
        out.parent.mkdir(exist_ok=True)
        out.write_text(report)
        print(f"\n{'=' * 60}")
        print(f"Accuracy report written to: {out}")
        print(f"{'=' * 60}")

    request.addfinalizer(_finalise)
