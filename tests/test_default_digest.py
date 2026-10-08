"""With no plugin configured, ``content_digest`` on the synthetic fixtures is pinned.

The set covers every default extraction path: text, DOCX, CSV, XLSX, native
PDFs (vector redactions, a ruled table) and an image-only PDF, which runs the
layout step and OCR. The values were recorded when the synthetic set replaced
the vendored documents, on unchanged extraction code.

A change here means the default model group, or the extraction it drives, has
changed output: regenerate the digests deliberately, never to get green.

PDF digests also depend on the PyMuPDF version, so the PDF cases run only under
the version they were recorded with (the one ``uv.lock`` pins).
"""

from __future__ import annotations

import fitz
import pytest

from tests._synthetic import SYNTHETIC_DIR
from womblex.config import DatasetConfig, PathsConfig, WomblexConfig
from womblex.ingest.detect import detect_file_type
from womblex.ingest.extract import extract_text
from womblex.ingest.layout_step import LayoutSettings
from womblex.store.content_digest import content_digest

_PDF_PINNED_UNDER = "1.27.2.2"

_PINNED = {
    "documents/koala-habitat-audit_transcript.txt":
        "4e36c52f80a58d095c8c5cda3036332c1480778c0460c444b9554bb50c59fd66",
    "documents/wombat-portfolio-budget-statements.docx":
        "20edacc66c1f433618fd27e3d3877ff175c9eb8765547a0eb08cd1ce54930a1b",
    "spreadsheets/platypus-sightings-register.csv":
        "78f54d548b957076cdd2a360bd2862b32a32f01df6e4775cb2afcc3283707631",
    "spreadsheets/echidna-population-statistics.xlsx":
        "e8dcda866921d1b7144db7e5e5d7bcc7f84e06eef172af4c87195c75785cd4a1",
    "documents/quokka-care-decision-notice_redacted.pdf":
        "d16e3945836c8ec22c2c9ab25a5c350d3cd25fe4a86b3e85cdc77928f51cda08",
    "documents/koala-habitat-audit.pdf":
        "be8091e48559f7c91fedef45c83809780297c6c12299db61d74d465da593bc9b",
    "documents/numbat-scanned-survey-page.pdf":
        "54b024a1f64cfaab4653a55655a728957de6047becd65c8749113c298e01268f",
}


@pytest.mark.parametrize(
    "rel",
    [
        pytest.param(rel, marks=pytest.mark.slow) if rel.endswith(".pdf") else rel
        for rel in _PINNED
    ],
)
def test_default_content_digest_is_pinned(rel: str) -> None:
    path = SYNTHETIC_DIR / rel
    if path.suffix == ".pdf" and fitz.VersionBind != _PDF_PINNED_UNDER:
        pytest.skip(f"PDF digests pinned under PyMuPDF {_PDF_PINNED_UNDER}, found {fitz.VersionBind}")
    # The layout step is part of default extraction: OCR pages read its regions.
    config = WomblexConfig(
        dataset=DatasetConfig(name="digest"),
        paths=PathsConfig(input_root=path.parent, output_root=path.parent, checkpoint_dir=path.parent),
    )
    results = extract_text(path, detect_file_type(path), layout=LayoutSettings.from_config(config))
    assert content_digest(results[0].elements) == _PINNED[rel]
