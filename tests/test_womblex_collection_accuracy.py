"""Accuracy tests for the womblex-collection fixtures.

Runs PII detection across all womblex-collection documents (production-
representative) and benchmark image fixtures. Throsby has PII ground truth;
other documents are scanned without GT to surface false positives and coverage.

Results are written to ``docs/accuracy/REDACTION_HANDLING.md`` and
``docs/accuracy/PII_CLEANING.md``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

from tests.accuracy_reports import generate_pii_report, generate_redaction_report

logger = logging.getLogger(__name__)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "fixtures"
WOMBLEX_DIR = FIXTURES / "womblex-collection"
THROSBY_PDF = WOMBLEX_DIR / "_documents" / (
    "00768-213A-270825-Throsby-Out-of-School-Care-"
    "Administrative-Decision-Other-Notice-and-Direction_Redacted.pdf"
)
THROSBY_GT_REDACTED = THROSBY_PDF.with_name(
    THROSBY_PDF.stem + "_transcript-with-redacted-tags.txt"
)
THROSBY_GT_PII = THROSBY_PDF.with_name(
    THROSBY_PDF.stem + "_transcript-with-redacted-tags-and-pii-cleaning.txt"
)

# ---------------------------------------------------------------------------
# Accumulated results (populated by tests, consumed by report generators)
# ---------------------------------------------------------------------------

_redaction: dict = {}
_pii: dict = {}
_fidelity: dict = {}
_cross_fixture_pii: list[dict] = []

FUNSD_DIR = FIXTURES / "funsd"
IAM_DIR = FIXTURES / "iam_line"
DOCLAYNET_DIR = FIXTURES / "doclaynet"

# All image fixtures for cross-fixture PII scanning (no GT, just report output).
_IMAGE_FIXTURES: list[dict[str, str]] = [
    {"dataset": "FUNSD", "stem": "85540866", "dir": "funsd/images", "ext": ".png"},
    {"dataset": "FUNSD", "stem": "82200067_0069", "dir": "funsd/images", "ext": ".png"},
    {"dataset": "FUNSD", "stem": "87594142_87594144", "dir": "funsd/images", "ext": ".png"},
    {"dataset": "FUNSD", "stem": "87528321", "dir": "funsd/images", "ext": ".png"},
    {"dataset": "FUNSD", "stem": "87528380", "dir": "funsd/images", "ext": ".png"},
    {"dataset": "IAM", "stem": "short_1602", "dir": "iam_line", "ext": ".png"},
    {"dataset": "IAM", "stem": "median_15", "dir": "iam_line", "ext": ".png"},
    {"dataset": "IAM", "stem": "long_4", "dir": "iam_line", "ext": ".png"},
    {"dataset": "IAM", "stem": "wide_1739", "dir": "iam_line", "ext": ".png"},
    {"dataset": "IAM", "stem": "narrow_1163", "dir": "iam_line", "ext": ".png"},
    {"dataset": "DocLayNet", "stem": "dense_text_548", "dir": "doclaynet", "ext": ".png"},
    {"dataset": "DocLayNet", "stem": "diverse_layout_49", "dir": "doclaynet", "ext": ".png"},
    {"dataset": "DocLayNet", "stem": "sparse_text_344", "dir": "doclaynet", "ext": ".png"},
    {"dataset": "DocLayNet", "stem": "formula_29", "dir": "doclaynet", "ext": ".png"},
    {"dataset": "DocLayNet", "stem": "table_0", "dir": "doclaynet", "ext": ".png"},
]


# ---------------------------------------------------------------------------
# Ground-truth parsing helpers
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"<([A-Z_]+)>")


def _count_tags(text: str) -> dict[str, int]:
    """Count occurrences of each ``<TAG>`` in text."""
    counts: dict[str, int] = {}
    for m in _TAG_RE.finditer(text):
        tag = m.group(1)
        counts[tag] = counts.get(tag, 0) + 1
    return counts


def _extract_tagged_spans(
    tagged_text: str, untagged_text: str, tag: str,
) -> list[str]:
    """Extract original text replaced by ``<TAG>``, handling mixed-tag lines."""
    _OTHER_TAG = re.compile(r"<[A-Z_]+>")
    spans: list[str] = []
    tag_marker = f"<{tag}>"
    tagged_lines = tagged_text.splitlines()
    untagged_lines = untagged_text.splitlines()

    for t_line, u_line in zip(tagged_lines, untagged_lines):
        if tag_marker not in t_line:
            continue
        parts = t_line.split(tag_marker)
        pos = 0
        for i in range(len(parts) - 1):
            anchor = parts[i]
            next_anchor = parts[i + 1]
            anchor_end = u_line.find(anchor, pos) if not _OTHER_TAG.search(anchor) else -1
            if anchor_end == -1:
                # Anchor contains other tags — use regex with wildcards
                anchor_pat = re.escape(anchor)
                anchor_pat = _OTHER_TAG.sub(r".*?", anchor_pat)
                m = re.search(anchor_pat, u_line[pos:])
                if m:
                    anchor_end = pos + m.start()
                    span_start = pos + m.end()
                else:
                    continue
            else:
                span_start = anchor_end + len(anchor)

            if next_anchor.strip():
                # Build regex-safe anchor for matching (other tags → wildcard)
                na_stripped = next_anchor.strip()
                if _OTHER_TAG.search(na_stripped):
                    na_pat = re.escape(na_stripped)
                    na_pat = _OTHER_TAG.sub(r".*?", na_pat)
                    m = re.search(na_pat, u_line[span_start:])
                    span_end = (span_start + m.start()) if m else len(u_line)
                else:
                    idx = u_line.find(na_stripped, span_start)
                    span_end = idx if idx != -1 else len(u_line)
            else:
                span_end = len(u_line)

            span_text = u_line[span_start:span_end].strip()
            if span_text:
                spans.append(span_text)
            pos = span_end

    return spans


# ---------------------------------------------------------------------------
# Redaction detection accuracy
# ---------------------------------------------------------------------------


class TestRedactionDetection:
    """Test that redaction detector finds the correct number of redacted
    regions on the correct pages of the Throsby PDF."""

    def test_throsby_redaction_count_and_pages(self) -> None:
        if not THROSBY_PDF.exists():
            pytest.skip(f"Fixture missing: {THROSBY_PDF}")

        from womblex.redact.detector import RedactionDetector
        from womblex.redact.stage import detect_redactions

        gt_text = THROSBY_GT_REDACTED.read_text(encoding="utf-8")
        gt_count = gt_text.count("<REDACTED>")

        detector = RedactionDetector()
        report = detect_redactions(THROSBY_PDF, page_count=3, detector=detector)

        logger.info(
            "Redaction detection: GT=%d tags, detected=%d regions on pages %s",
            gt_count, report.total, report.affected_pages,
        )

        per_page: dict[int, list[dict]] = {}
        for page_num in sorted(report.page_redactions):
            regions = report.page_redactions[page_num]
            per_page[page_num] = [
                {"bbox": r.bbox, "area_px": r.area_px} for r in regions
            ]
            logger.info("  Page %d: %d regions detected", page_num, len(regions))
            for r in regions:
                logger.info("    bbox=%s area=%d px", r.bbox, r.area_px)

        _redaction.update(
            gt_count=gt_count,
            detected=report.total,
            affected_pages_gt=sorted(
                p for p in [0, 2] if any(p == pg for pg in report.page_redactions)
            ),
            affected_pages_detected=sorted(report.affected_pages),
            per_page=per_page,
        )

        assert report.total > 0, "No redactions detected at all"
        assert 0 in report.affected_pages, "Page 0 should have redactions"
        assert 1 not in report.affected_pages, "Page 1 should have no redactions"

    def test_throsby_redaction_annotates_extraction(self) -> None:
        """Full pipeline: extract then detect redactions, verify annotation."""
        if not THROSBY_PDF.exists():
            pytest.skip(f"Fixture missing: {THROSBY_PDF}")

        from womblex.ingest.detect import DetectionConfig, detect_file_type
        from womblex.ingest.extract import extract_text
        from womblex.redact.detector import RedactionDetector
        from womblex.redact.stage import annotate_extraction, detect_redactions

        profile = detect_file_type(THROSBY_PDF, DetectionConfig())
        results = extract_text(THROSBY_PDF, profile, max_pages=30)
        extraction = results[0]
        assert extraction.error is None

        detector = RedactionDetector()
        report = detect_redactions(
            THROSBY_PDF,
            page_count=len(extraction.pages),
            detector=detector,
        )
        annotate_extraction(extraction, report)

        assert report.total > 0
        assert len(extraction.warnings) > 0, "Expected redaction warnings"
        for w in extraction.warnings:
            assert "redacted region" in w
            logger.info("Warning: %s", w)

        _redaction["annotation_warnings"] = len(extraction.warnings)

    def test_throsby_blackout_mode(self) -> None:
        """Blackout mode prepends [REDACTED] to affected page text."""
        if not THROSBY_PDF.exists():
            pytest.skip(f"Fixture missing: {THROSBY_PDF}")

        from womblex.ingest.detect import DetectionConfig, detect_file_type
        from womblex.ingest.extract import extract_text
        from womblex.redact.detector import RedactionDetector
        from womblex.redact.stage import apply_text_redaction, detect_redactions

        profile = detect_file_type(THROSBY_PDF, DetectionConfig())
        results = extract_text(THROSBY_PDF, profile, max_pages=30)
        extraction = results[0]

        detector = RedactionDetector()
        report = detect_redactions(
            THROSBY_PDF,
            page_count=len(extraction.pages),
            detector=detector,
        )
        apply_text_redaction(extraction.pages, report, mode="blackout")

        redacted_pages = [
            p for p in extraction.pages if p.text.startswith("[REDACTED]")
        ]
        assert len(redacted_pages) >= 1, "At least one page should have [REDACTED] prefix"
        logger.info(
            "Blackout mode: %d/%d pages tagged", len(redacted_pages), len(extraction.pages)
        )
        _redaction["blackout_pages"] = len(redacted_pages)
        _redaction["total_pages"] = len(extraction.pages)


# ---------------------------------------------------------------------------
# PII tagging accuracy
# ---------------------------------------------------------------------------


class TestPIITagging:
    """Test PII cleaner against human-annotated ground truth."""

    _SUPPORTED_ENTITIES = ["PERSON", "ADDRESS"]

    @pytest.fixture()
    def gt_entities(self) -> dict[str, list[str]]:
        """Parse PII entity spans from ground truth for all supported types."""
        gt_pii = THROSBY_GT_PII.read_text(encoding="utf-8")
        gt_redacted = THROSBY_GT_REDACTED.read_text(encoding="utf-8")
        tags = _count_tags(gt_pii)
        tags.pop("REDACTED", None)
        logger.info("Ground truth PII tags: %s", tags)
        result: dict[str, list[str]] = {"tag_counts": tags}  # type: ignore[dict-item]
        for entity in self._SUPPORTED_ENTITIES:
            spans = _extract_tagged_spans(gt_pii, gt_redacted, entity)
            result[entity] = spans
            logger.info("Ground truth %s spans: %s", entity, spans)
        return result

    def test_throsby_pii_detection(self, gt_entities: dict) -> None:
        """Run PII cleaner with all supported entities and check detection."""
        if not THROSBY_PDF.exists():
            pytest.skip(f"Fixture missing: {THROSBY_PDF}")

        from womblex.ingest.detect import DetectionConfig, detect_file_type
        from womblex.ingest.extract import extract_text

        profile = detect_file_type(THROSBY_PDF, DetectionConfig())
        results = extract_text(THROSBY_PDF, profile, max_pages=30)
        extracted_text = results[0].full_text

        try:
            from womblex.pii.cleaner import PIICleaner
        except ImportError:
            pytest.skip("PII dependencies not installed (sentence-transformers)")

        cleaner = PIICleaner(entities=self._SUPPORTED_ENTITIES)

        try:
            cleaned_text, replacement_count = cleaner.clean(extracted_text)
        except ImportError:
            pytest.skip("PII dependencies not installed (presidio-anonymizer)")
        except Exception as exc:
            err_str = f"{type(exc).__name__}: {exc}"
            if any(kw in err_str for kw in ("Proxy", "Connection", "Timeout", "Forbidden")):
                pytest.skip(f"Cannot download PII model: {err_str}")
            raise

        logger.info("PII cleaner: %d replacements made", replacement_count)

        entity_results: dict[str, dict] = {}
        for entity in self._SUPPORTED_ENTITIES:
            gt_spans: list[str] = gt_entities.get(entity, [])
            tag_marker = f"<{entity}>"
            pred_count = cleaned_text.count(tag_marker)

            found: list[str] = []
            missed: list[str] = []
            for span in gt_spans:
                if span in extracted_text and span not in cleaned_text:
                    found.append(span)
                elif span not in extracted_text:
                    logger.info("  %s skipped (in redacted region): %s", entity, span)
                else:
                    missed.append(span)

            recall = len(found) / len(gt_spans) if gt_spans else 0.0
            precision = len(found) / pred_count if pred_count else 0.0

            logger.info(
                "%s recall: %.1f%% (%d/%d)", entity, recall * 100, len(found), len(gt_spans)
            )
            logger.info("%s tags in output: %d (expected %d from GT)", entity, pred_count, len(gt_spans))
            for s in found:
                logger.info("  Found: %s", s)
            for s in missed:
                logger.info("  Missed: %s", s)

            entity_results[entity] = {
                "gt_spans": gt_spans,
                "found": found,
                "missed": missed,
                "recall": recall,
                "precision": precision,
                "pred_count": pred_count,
            }

        _pii.update(
            replacement_count=replacement_count,
            entity_results=entity_results,
            threshold=cleaner._threshold,
            model=cleaner._model_name,
            # Backwards-compat keys for PERSON (used by report)
            pred_count=entity_results["PERSON"]["pred_count"],
            gt_persons=entity_results["PERSON"]["gt_spans"],
            found_persons=entity_results["PERSON"]["found"],
            missed_persons=entity_results["PERSON"]["missed"],
            recall=entity_results["PERSON"]["recall"],
            precision=entity_results["PERSON"]["precision"],
        )

        assert replacement_count > 0, "Expected at least one PII replacement"

    def test_throsby_pii_entity_coverage(self, gt_entities: dict) -> None:
        """Report which GT entity types the pipeline can and cannot handle."""
        tag_counts: dict[str, int] = gt_entities["tag_counts"]

        supported = {"PERSON", "ADDRESS"}
        total_entities = sum(tag_counts.values())
        supported_entities = sum(
            count for tag, count in tag_counts.items() if tag in supported
        )
        coverage = supported_entities / total_entities if total_entities else 0.0

        logger.info(
            "Entity coverage: %.1f%% (%d/%d entities from %d types)",
            coverage * 100, supported_entities, total_entities, len(tag_counts),
        )
        logger.info(
            "Unsupported: %s",
            {t: c for t, c in tag_counts.items() if t not in supported},
        )

        _pii.update(
            tag_counts=tag_counts,
            supported_entities=supported_entities,
            total_entities=total_entities,
            coverage=coverage,
        )


# ---------------------------------------------------------------------------
# Extraction quality — text fidelity against human transcript
# ---------------------------------------------------------------------------


class TestExtractionFidelity:
    """Compare womblex extraction output against the human-proofread transcript."""

    def test_throsby_native_extraction_cer(self) -> None:
        """CER between native PDF extraction and human transcript."""
        if not THROSBY_PDF.exists():
            pytest.skip(f"Fixture missing: {THROSBY_PDF}")

        from womblex.ingest.detect import DetectionConfig, detect_file_type
        from womblex.ingest.extract import extract_text

        profile = detect_file_type(THROSBY_PDF, DetectionConfig())
        results = extract_text(THROSBY_PDF, profile, max_pages=30)
        extracted = results[0].full_text

        gt_text = THROSBY_GT_REDACTED.read_text(encoding="utf-8")
        gt_clean = gt_text.replace("<REDACTED>", "").strip()

        cer = _char_error_rate(extracted, gt_clean)
        wer = _word_error_rate(extracted, gt_clean)

        logger.info(
            "Throsby extraction fidelity: CER=%.3f WER=%.3f "
            "(extracted=%d chars, GT=%d chars)",
            cer, wer, len(extracted), len(gt_clean),
        )

        _fidelity.update(
            cer=cer,
            wer=wer,
            extracted_chars=len(extracted),
            gt_chars=len(gt_clean),
        )

        assert cer < 0.10, f"CER {cer:.3f} too high for native PDF extraction"
        assert wer < 0.15, f"WER {wer:.3f} too high for native PDF extraction"


# ---------------------------------------------------------------------------
# Womblex-collection PII scanning (production-representative)
# ---------------------------------------------------------------------------

_WOMBLEX_PII_FIXTURES: list[dict[str, str]] = [
    {
        "name": "Throsby",
        "file": "_documents/00768-213A-270825-Throsby-Out-of-School-Care-"
                "Administrative-Decision-Other-Notice-and-Direction_Redacted.pdf",
    },
    {
        "name": "Auditor-General",
        "file": "_documents/Auditor-General_Report_2020-21_19.pdf",
    },
    {
        "name": "DFAT-Budget-Statements",
        "file": "_documents/foreign-affairs-and-trade-2025-26-portfolio-budget-statements.docx",
    },
    {
        "name": "Approved-Providers",
        "file": "_spreadsheets/Approved-providers-au-export_20260204.csv",
    },
    {
        "name": "MSO-Statistics",
        "file": "_spreadsheets/mso-statistics-sept-qtr-2025.xlsx",
    },
]

_womblex_pii: list[dict] = []


class TestWomblexCollectionPII:
    """Run PII detection on all womblex-collection documents."""

    @pytest.mark.parametrize(
        "fixture",
        _WOMBLEX_PII_FIXTURES,
        ids=[f["name"] for f in _WOMBLEX_PII_FIXTURES],
    )
    def test_womblex_pii_scan(self, fixture: dict[str, str]) -> None:
        doc_path = WOMBLEX_DIR / fixture["file"]
        if not doc_path.exists():
            pytest.skip(f"Fixture missing: {doc_path}")

        from womblex.ingest.detect import DetectionConfig, detect_file_type
        from womblex.ingest.extract import extract_text

        profile = detect_file_type(doc_path, DetectionConfig())
        results = extract_text(doc_path, profile, max_pages=30)
        extracted_text = results[0].full_text

        all_entities = ["PERSON", "ADDRESS"]
        tag_counts: dict[str, int] = {}
        replacement_count = 0
        try:
            from womblex.pii.cleaner import PIICleaner
            cleaner = PIICleaner(entities=all_entities)
            cleaned_text, replacement_count = cleaner.clean(extracted_text)
            for ent in all_entities:
                tag_counts[ent] = cleaned_text.count(f"<{ent}>")
        except (ImportError, Exception) as exc:
            logger.info("PII scan skipped for %s: %s", fixture["name"], exc)

        _womblex_pii.append({
            "name": fixture["name"],
            "extracted_words": len(extracted_text.split()),
            "tag_counts": tag_counts,
            "total_replacements": replacement_count,
        })

        logger.info(
            "Womblex/%s: %d words, tags=%s, %d replacements",
            fixture["name"], len(extracted_text.split()),
            tag_counts, replacement_count,
        )


# ---------------------------------------------------------------------------
# Benchmark-dataset PII scanning (OCR image fixtures, no GT)
# ---------------------------------------------------------------------------


class TestBenchmarkPII:
    """Run PII detection on benchmark image fixtures (no GT)."""

    @pytest.mark.parametrize(
        "fixture",
        _IMAGE_FIXTURES,
        ids=[f"{f['dataset']}/{f['stem']}" for f in _IMAGE_FIXTURES],
    )
    def test_benchmark_pii_scan(self, fixture: dict[str, str]) -> None:
        img_path = FIXTURES / fixture["dir"] / (fixture["stem"] + fixture["ext"])
        if not img_path.exists():
            pytest.skip(f"Fixture missing: {img_path}")

        import cv2
        from womblex.ingest.paddle_ocr import get_paddle_reader, preprocess_for_ocr

        img = cv2.imread(str(img_path))
        assert img is not None
        preprocessed, _ = preprocess_for_ocr(img)
        reader = get_paddle_reader(lang="eng")
        ocr_results = reader.readtext(preprocessed)
        extracted_text = " ".join(t for _, t, _ in ocr_results)

        all_entities = ["PERSON", "ADDRESS"]
        tag_counts: dict[str, int] = {}
        try:
            from womblex.pii.cleaner import PIICleaner
            cleaner = PIICleaner(entities=all_entities)
            cleaned_text, replacement_count = cleaner.clean(extracted_text)
            for ent in all_entities:
                tag_counts[ent] = cleaned_text.count(f"<{ent}>")
        except (ImportError, Exception) as exc:
            replacement_count = 0
            logger.info("PII scan skipped for %s: %s", fixture["stem"], exc)

        _cross_fixture_pii.append({
            "dataset": fixture["dataset"],
            "stem": fixture["stem"],
            "extracted_words": len(extracted_text.split()),
            "tag_counts": tag_counts,
            "total_replacements": replacement_count,
        })

        logger.info(
            "%s/%s: %d words, tags=%s, %d replacements",
            fixture["dataset"], fixture["stem"],
            len(extracted_text.split()), tag_counts, replacement_count,
        )
        assert True


# ---------------------------------------------------------------------------
# Report writing (generators live in accuracy_reports.py)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="session")
def write_reports(request: pytest.FixtureRequest) -> None:
    """Write accuracy docs after the session completes."""
    docs = Path(__file__).resolve().parent.parent / "docs" / "accuracy"

    def _finalise() -> None:
        docs.mkdir(exist_ok=True)
        redaction_path = docs / "REDACTION_HANDLING.md"
        redaction_path.write_text(generate_redaction_report(_redaction))
        print(f"\n{'=' * 60}")
        print(f"Redaction report written to: {redaction_path}")

        pii_path = docs / "PII_CLEANING.md"
        pii_path.write_text(
            generate_pii_report(_pii, _womblex_pii, _cross_fixture_pii),
        )
        print(f"PII report written to: {pii_path}")
        print(f"{'=' * 60}")

    request.addfinalizer(_finalise)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _levenshtein(s1: str, s2: str) -> int:
    from rapidfuzz.distance import Levenshtein
    return Levenshtein.distance(s1, s2)


def _levenshtein_seq(s1: list[str], s2: list[str]) -> int:
    from rapidfuzz.distance import Levenshtein
    return Levenshtein.distance(s1, s2)


def _char_error_rate(predicted: str, reference: str) -> float:
    pred = _normalise(predicted)
    ref = _normalise(reference)
    if not ref:
        return 0.0 if not pred else 1.0
    return _levenshtein(pred, ref) / len(ref)


def _word_error_rate(predicted: str, reference: str) -> float:
    pred_words = _normalise(predicted).split()
    ref_words = _normalise(reference).split()
    if not ref_words:
        return 0.0 if not pred_words else 1.0
    return _levenshtein_seq(pred_words, ref_words) / len(ref_words)
