"""Block grouping under the pdfium backend matches MuPDF's on the synthetic PDFs.

MuPDF starts a block where the baseline pitch exceeds about 1.5 font sizes, keeps a
row's fragments (table cells) together, and does not break on a size or weight
change alone. The spacing fixture varies leading and paragraph gap so the rule is
measured at more than one pitch.
"""

from __future__ import annotations

import collections
from pathlib import Path

import pytest

from tests._synthetic import AUDIT_PDF, NOTICE_PDF, SCHEDULE_PDF, SPACING_PDF
from womblex.ingest import pdf
from womblex.ingest.detect import detect_file_type
from womblex.ingest.extract import extract_text
from womblex.ingest.pdf import open_document

PARITY = [AUDIT_PDF, SPACING_PDF, NOTICE_PDF, SCHEDULE_PDF]


def _lines_per_block(path: Path, backend: str) -> list[list[int]]:
    return [[len(b.lines) for b in page.text_dict()] for page in open_document(path, backend=backend)]


def _kinds(path: Path, backend: str, monkeypatch: pytest.MonkeyPatch) -> collections.Counter[str]:
    monkeypatch.setattr(pdf, "DEFAULT_BACKEND", backend)
    return collections.Counter(
        e.kind for r in extract_text(path, detect_file_type(path)) for e in r.elements
    )


@pytest.mark.parametrize("path", PARITY, ids=lambda p: p.name)
def test_blocks_hold_the_same_lines_as_mupdf(path: Path) -> None:
    assert _lines_per_block(path, "pdfium") == _lines_per_block(path, "fitz")


@pytest.mark.parametrize("path", [AUDIT_PDF, SPACING_PDF], ids=lambda p: p.name)
def test_element_kinds_and_counts_match_mupdf(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert _kinds(path, "pdfium", monkeypatch) == _kinds(path, "fitz", monkeypatch)
