"""Form field extraction.

Three sources, in priority order:
1. PDF AcroForm widgets (`_extract_form_fields`) — interactive form
   fields native to the PDF, highest confidence.
2. Spatial text-pair extraction (`_extract_form_pairs_from_text`) —
   walks `page.get_text("dict")` looking for label/value pairs by span
   alignment. Catches text-only forms in native PDFs (NQA notifications,
   provider/service field blocks).
3. Line-based extraction (`_extract_form_pairs_from_lines`) — scans
   assembled plain text (OCR output) for label/value pairs. Used by
   scanned extractors where the native text layer is empty.

Label heuristics: 1–6 words, must start with uppercase, no sentence
punctuation, no list-marker shapes (`A)`, `(i)`), no URL-prefix labels
(`http`, `https`).
"""
from __future__ import annotations

import re

import fitz

from womblex.ingest.extract import (
    FormField,
    Position,
    _normalise_bbox,
    _normalise_rect,
)


# Label heuristics: 1–6 words, must start with uppercase, no sentence
# punctuation inside, no list-marker shapes ("A)", "(i)").
_FORM_LABEL_RE = re.compile(r"^[A-Z][A-Za-z0-9 /'\-()&]{0,49}:?$")
_LIST_MARKER_RE = re.compile(r"^([A-Za-z]\)|\(.+\))$")
_URL_LABEL_LOWER = {"http", "https", "ftp", "file", "ssh", "mailto"}
_FORM_PAIR_GAP_RE = re.compile(r"^([A-Z]\S?(?:.{0,48}?\S)?)\s{2,}(\S.+?)\s*$")

# Labels that recur in Australian regulatory letter prose and aren't real
# form fields — see STATUS.md K3 for the audit data.
_LABEL_DENYLIST: frozenset[str] = frozenset({
    "Penalty",      # Regulation citation: "Penalty: $10 000, in the case of an individual"
    "OFFICIAL",     # Document classification banner: "OFFICIAL: Sensitive - Legislative Secrecy"
    "Note",         # Aside: "Note: ..."
    "Caution",      # Email warning banner
})


def _looks_like_form_label(text: str) -> bool:
    text = text.strip().rstrip(":")
    if not text or len(text) > 50:
        return False
    if not text[0].isupper():
        return False
    if text in _LABEL_DENYLIST:
        return False
    if text.lower() in _URL_LABEL_LOWER:
        return False
    words = text.split()
    if not (1 <= len(words) <= 6):
        return False
    if any(c in text for c in ".?!"):
        return False
    if _LIST_MARKER_RE.match(text):
        return False
    return bool(_FORM_LABEL_RE.match(text + ":"))


def _extract_form_fields(page: fitz.Page) -> list[FormField]:
    """Extract interactive AcroForm widgets from a page."""
    fields: list[FormField] = []
    pw, ph = page.rect.width, page.rect.height

    for widget in page.widgets():
        name = widget.field_name or ""
        value = widget.field_value or ""
        pos = _normalise_rect(widget.rect, pw, ph)
        fields.append(FormField(field_name=name, value=value, position=pos, confidence=0.9))

    return fields


def _extract_form_pairs_from_text(page: fitz.Page) -> list[FormField]:
    """Extract label/value pairs from text-only forms (no AcroForm widgets).

    Catches NQA-style notification forms where labels and values are
    separate spans on the same line (`Provider Name    Wonderschool ...`)
    or single spans split by colon (`Provider Name: Wonderschool ...`).
    """
    fields: list[FormField] = []
    pw, ph = page.rect.width, page.rect.height

    raw = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
            if not spans:
                continue

            # Multi-span line — first span = label candidate, rest = value
            if len(spans) >= 2:
                first = spans[0]
                label = first.get("text", "").strip().rstrip(":")
                value_parts = [s.get("text", "").strip() for s in spans[1:]]
                value = " ".join(p for p in value_parts if p).strip()
                if not value or not _looks_like_form_label(label):
                    continue
                # Require horizontal gap between label end and value start
                first_bbox = first.get("bbox", (0, 0, 0, 0))
                second_bbox = spans[1].get("bbox", (0, 0, 0, 0))
                gap = second_bbox[0] - first_bbox[2]
                if gap < 5:
                    continue
                pos = _normalise_bbox(first_bbox, pw, ph)
                fields.append(
                    FormField(field_name=label, value=value, position=pos, confidence=0.65)
                )
                continue

            # Single span — only catch explicit "Label: value" pattern
            text = spans[0].get("text", "").strip()
            if ":" not in text or len(text) > 200:
                continue
            label, _, value = text.partition(":")
            label = label.strip()
            value = value.strip()
            if not value or not _looks_like_form_label(label):
                continue
            bbox = spans[0].get("bbox", (0, 0, 0, 0))
            pos = _normalise_bbox(bbox, pw, ph)
            fields.append(
                FormField(field_name=label, value=value, position=pos, confidence=0.7)
            )

    return fields


def _extract_forms(page: fitz.Page) -> list[FormField]:
    """Extract form fields, preferring AcroForm widgets, falling back to text."""
    fields = _extract_form_fields(page)
    if fields:
        return fields
    return _extract_form_pairs_from_text(page)


def _extract_form_pairs_from_lines(text: str) -> list[FormField]:
    """Scan plain text (e.g. assembled OCR output) for label/value pairs.

    Used by scanned extractors where `page.get_text("dict")` is empty —
    NQA notification forms in the corpus carry their structure only in the
    OCR'd text. Position is zero (no bbox available from line-only scan).
    Detects same-line pairs separated by ≥2-space gap or by a colon.
    """
    fields: list[FormField] = []
    zero_pos = Position(x=0.0, y=0.0, width=0.0, height=0.0)

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line or len(line) < 4:
            continue

        m = _FORM_PAIR_GAP_RE.match(line)
        if m:
            label = m.group(1).strip().rstrip(":")
            value = m.group(2).strip()
            if value and _looks_like_form_label(label):
                fields.append(
                    FormField(field_name=label, value=value, position=zero_pos, confidence=0.6)
                )
                continue

        if ":" in line:
            label, _, value = line.partition(":")
            label = label.strip()
            value = value.strip()
            if (
                label
                and value
                and not value.startswith("//")
                and _looks_like_form_label(label)
            ):
                fields.append(
                    FormField(field_name=label, value=value, position=zero_pos, confidence=0.65)
                )

    return fields
