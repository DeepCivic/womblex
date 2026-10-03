"""With no plugin configured, ``content_digest`` on vendored fixtures is pinned.

The pinned values were verified identical on the commit before the model
registry landed, so they record that the registry changed no default output.

A change here means the default model group, or the extraction it drives, has
changed output: regenerate the digests deliberately, never to get green.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from womblex.ingest.detect import detect_file_type
from womblex.ingest.extract import extract_text
from womblex.store.content_digest import content_digest

_COLLECTION = Path(__file__).resolve().parent.parent / "fixtures" / "fixtures" / "womblex-collection"

_PINNED = {
    "_documents/Auditor-General_Report_2020-21_19_transcript-First-30-Pages.txt":
        "b77c08bce6b08e4138e259cb4ca5ff5eaae2aae7db3ae41c1c27ca44b91c2e01",
    "_documents/foreign-affairs-and-trade-2025-26-portfolio-budget-statements.docx":
        "19b1aa947fb7f0d540eed941ae54c703300df71bfd8bd419e68d14d913273b00",
    "_spreadsheets/Approved-providers-au-export_20260204.csv":
        "068a1e6cc72e6b25558163c8cf3a2ee9640ee128575303f508b78dbd19c22215",
    "_spreadsheets/mso-statistics-sept-qtr-2025.xlsx":
        "75c55d96b43f33967745ca61c911eec767a7d43659c982e8d5fa1e3065b16f0b",
    "_documents/00768-213A-270825-Throsby-Out-of-School-Care-Administrative-Decision-Other-Notice-and-Direction_Redacted.pdf":
        "0753b31f3c251e00bc900fb3c505d05f41e1c7e81b5d6b2e7d039f9b1c6b099f",
    "_documents/Auditor-General_Report_2020-21_19-First-30-Pages.pdf":
        "45089a415a82caf3ba534a90d5727f6d6a79013be48d2333d0795dd4dce101ce",
}


@pytest.mark.parametrize(
    "rel",
    [
        pytest.param(rel, marks=pytest.mark.slow) if rel.endswith(".pdf") else rel
        for rel in _PINNED
    ],
)
def test_default_content_digest_is_pinned(rel: str) -> None:
    path = _COLLECTION / rel
    if not path.exists():
        pytest.skip(f"fixture not present: {path}")
    results = extract_text(path, detect_file_type(path))
    assert content_digest(results[0].elements) == _PINNED[rel]
