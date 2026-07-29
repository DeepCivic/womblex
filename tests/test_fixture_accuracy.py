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

import functools
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

# Tier 2 (womblex-benchmark). Every test here runs the real OCR / layout /
# extraction pipeline over benchmark fixtures (the Auditor-General CER case
# alone OCRs a 30-page PDF, >5 min), and the suite regenerates docs/accuracy/*.
# Mark the whole module `benchmark` so the fast tier (`-m "not slow and not
# benchmark"`) excludes it; run it explicitly via `-m benchmark` or by naming
# the file (per CLAUDE.md). The per-class @pytest.mark.benchmark below are now
# redundant but harmless.
pytestmark = pytest.mark.benchmark

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
        "file": "_documents/Auditor-General_Report_2020-21_19-First-30-Pages.pdf",
        "transcript": "_documents/Auditor-General_Report_2020-21_19_transcript-First-30-Pages.txt",
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

# Map DocLayNet labels to womblex block_type equivalents.
# Must match `_YOLO_DOCLAYNET_LABEL_MAP` in `ingest/paddle_ocr.py` so the
# layout F1 measurement credits the analyzer for correct predictions.
DOCLAYNET_TO_WOMBLEX: dict[str, str] = {
    "Caption": "caption",
    "Footnote": "footnote",
    "Formula": "paragraph",
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
    from womblex.utils.metrics import cer
    return cer(reference, predicted)


def word_error_rate(predicted: str, reference: str) -> float:
    """Word error rate (WER)."""
    from womblex.utils.metrics import wer
    return wer(reference, predicted)


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
    from womblex.utils.metrics import spatial_sort_text
    return spatial_sort_text(words_with_boxes, line_tolerance)


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
    "act_eci": [],
    # table reconstruction. Populated by tests/test_table_benchmark.py
    # when that module runs in the same session; the write_report finaliser
    # below renders it into EXTRACTION.md's Table Reconstruction section.
    # See docs/evaluation.md §2b.
    "tables": [],
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


# Table-labelled runs shorter than this are annotation strays (footnote lines
# mislabelled Table in dense_text_548) — dropped from GT rather than FN-charged.
MIN_TABLE_GT_SPANS = 3


def _aggregate_doclaynet_blocks(
    bboxes: list[list[float]], labels: list[str],
) -> list[tuple[tuple[float, float, float, float], str]]:
    """Merge adjacent DocLayNet text spans with the same label into blocks.

    DocLayNet annotates per text-line; layout models predict page-level blocks.
    This groups consecutive spans sharing a label into a single bounding box.
    Table blocks with < ``MIN_TABLE_GT_SPANS`` spans are dropped as strays.
    """
    if not bboxes:
        return []

    blocks: list[tuple[tuple[float, float, float, float], str, int]] = []
    cur_label = labels[0]
    x0, y0, x1, y1 = bboxes[0]
    n_spans = 1

    for i in range(1, len(bboxes)):
        if labels[i] == cur_label:
            # Extend the current block
            bx0, by0, bx1, by1 = bboxes[i]
            x0 = min(x0, bx0)
            y0 = min(y0, by0)
            x1 = max(x1, bx1)
            y1 = max(y1, by1)
            n_spans += 1
        else:
            blocks.append(((x0, y0, x1, y1), cur_label, n_spans))
            cur_label = labels[i]
            x0, y0, x1, y1 = bboxes[i]
            n_spans = 1

    blocks.append(((x0, y0, x1, y1), cur_label, n_spans))
    return [(box, lbl) for box, lbl, n in blocks
            if not (lbl == "Table" and n < MIN_TABLE_GT_SPANS)]


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

        annotation = json.loads(ann_path.read_text(encoding="utf-8"))
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

        gt_text = gt_path.read_text(encoding="utf-8").strip()

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

        annotation = json.loads(ann_path.read_text(encoding="utf-8"))
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

        import fitz as _fitz

        from womblex.ingest.detect import DetectionConfig, detect_file_type
        from womblex.ingest.extract import extract_text

        profile = detect_file_type(file_path, DetectionConfig())

        # Determine total page count for reporting
        if file_path.suffix.lower() == ".pdf":
            _doc = _fitz.open(str(file_path))
            total_pages = _doc.page_count
            _doc.close()
        else:
            total_pages = None

        # Engine pinned: these numbers describe the region-based (paddleocr)
        # path. A config default change to an LLM engine would silently
        # measure a different pipeline. See docs/decisions.md “Table-cell
        # reconstruction on OCR pages — region-based engines only”.
        results = extract_text(file_path, profile, engine="paddleocr")
        extracted = results[0].full_text

        gt_text = transcript_path.read_text(encoding="utf-8").strip()

        # Strip page break markers from GT — extraction doesn't produce them
        gt_text = gt_text.replace("<Page Break>", "")
        # Collapse any resulting double-newlines from marker removal
        while "\n\n\n" in gt_text:
            gt_text = gt_text.replace("\n\n\n", "\n\n")
        gt_text = gt_text.strip()

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


# ---------------------------------------------------------------------------
# ACT-ECI labelled cohort: raw extraction vs normalise-stage output
# ---------------------------------------------------------------------------

_ACT_ECI_DIR = WOMBLEX_DIR / "_documents" / "act-eci-labelled-pages"
_ACT_ECI_MIN_GT = 20  # skip degenerate GT (fully-redacted / page-number stamps)

# Representative corpus OCR-confusion substitutions (i<->l, m<->rn families,
# with case variants). Demonstrates the normalise stage's effect on CER; the
# production map is derived data-driven + dictionary-gated, never hardcoded.
_ACT_ECI_OCR_SUBS: dict[str, str] = {
    "chlld": "child", "Chlld": "Child", "chlldren": "children",
    "servlce": "service", "servlces": "services", "Servlce": "Service",
    "provlder": "provider", "Provlder": "Provider", "incldent": "incident",
    "complalnt": "complaint", "educatlon": "education", "compllance": "compliance",
    "famlly": "family", "wlth": "with", "thls": "this", "emall": "email",
    "concem": "concern", "Concem": "Concern", "concems": "concerns",
    "concemed": "concerned", "govemment": "government", "Govemment": "Government",
    "retum": "return", "intemal": "internal", "extemal": "external",
    "leaming": "learning", "Leaming": "Learning",
}


def _act_eci_pages() -> list[dict]:
    """Discover labelled pages with non-degenerate plain GT (skip ⚠ pages)."""
    pages: list[dict] = []
    if not _ACT_ECI_DIR.is_dir():
        return pages
    for gt in sorted(_ACT_ECI_DIR.glob("*.gt.md")):
        stem = gt.name[:-len(".gt.md")]
        meta = _ACT_ECI_DIR / f"{stem}.meta.json"
        if not meta.exists():
            continue
        m = json.loads(meta.read_text(encoding="utf-8"))
        gt_text = gt.read_text(encoding="utf-8").strip()
        if len(gt_text) < _ACT_ECI_MIN_GT:
            continue
        pages.append({
            "stem": stem, "gt": gt_text, "pdf": m.get("source_pdf", ""),
            "page": int(m.get("page", 0)), "strategy": m.get("strategy", "?"),
        })
    return pages


@functools.cache
def _act_eci_extract(pdf_path: str):
    """Extract a fixture PDF once (cached — pages of one doc share extraction)."""
    from womblex.ingest.detect import DetectionConfig, detect_file_type
    from womblex.ingest.extract import extract_text
    p = Path(pdf_path)
    profile = detect_file_type(p, DetectionConfig())
    return extract_text(p, profile, engine="paddleocr")[0]


def _reassemble_page(elements: list, page: int, transforms=None) -> str:
    """Reassemble a page's narrative text (as `score` does); optionally normalise
    per element first so the result mirrors the normalise-stage output."""
    from womblex.process.normalise import normalise_text
    from womblex.score import DEFAULT_TEXT_KINDS
    selected = sorted(
        (e for e in elements if e.page == page and e.kind in DEFAULT_TEXT_KINDS),
        key=lambda e: e.order,
    )
    parts: list[str] = []
    for e in selected:
        text = e.text or ""
        if transforms is not None:
            text, _ = normalise_text(text, e.kind, transforms)
        text = text.strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


@pytest.mark.benchmark
class TestActEciLabelledPages:
    """Per-page CER on the ACT-ECI labelled cohort: raw extraction vs the
    normalise-stage (cleanup) output. Quantifies how normalisation moves CER."""

    @pytest.mark.parametrize("page", _act_eci_pages(), ids=lambda p: p["stem"][:36])
    def test_extraction_vs_normalised_cer(self, page: dict) -> None:
        from womblex.process.normalise import NormaliseTransforms

        pdf = _ACT_ECI_DIR / page["pdf"]
        if not pdf.exists():
            pytest.skip(f"source PDF missing: {pdf.name}")

        result = _act_eci_extract(str(pdf))
        transforms = NormaliseTransforms(
            unicode_hygiene=True, collapse_whitespace=True,
            despace_page_marker=True, substitutions=_ACT_ECI_OCR_SUBS,
        )
        raw = _reassemble_page(result.elements, page["page"])
        norm = _reassemble_page(result.elements, page["page"], transforms)
        raw_cer = char_error_rate(raw, page["gt"])
        norm_cer = char_error_rate(norm, page["gt"])

        _results["act_eci"].append({
            "stem": page["stem"], "strategy": page["strategy"],
            "raw_cer": raw_cer, "norm_cer": norm_cer, "delta": norm_cer - raw_cer,
            "gt_chars": len(page["gt"]),
        })
        logger.info(
            "ACT-ECI %s [%s]: CER raw=%.3f norm=%.3f (delta=%+.3f)",
            page["stem"][:36], page["strategy"], raw_cer, norm_cer, norm_cer - raw_cer,
        )
        # Regression guard: normalisation must not worsen fidelity.
        assert norm_cer <= raw_cer + 0.01


def _generate_report() -> str:
    """Delegate to accuracy_reports.generate_extraction_report."""
    from tests.accuracy_reports import generate_extraction_report
    return generate_extraction_report(_results)


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
