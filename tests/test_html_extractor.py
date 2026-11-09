"""Tests for HTML file extraction."""

from pathlib import Path

import pytest

from womblex.cli._shared import SUPPORTED_EXTENSIONS
from womblex.ingest.detect import DetectionConfig, DocumentType, detect_file_type
from womblex.ingest.extract import extract_text

PAGE = """<!doctype html><html><head><title>Ignored</title>
<style>p { color: red }</style><script>var x = "not text";</script></head>
<body><h1>Koala  Audit</h1>
<p>First <b>bold</b>
   paragraph &amp; more.</p>
<ul><li>one</li><li>two</li></ul>
<table><tr><th>Site</th><th>Count</th></tr><tr><td>North</td><td>12</td></tr></table>
<pre>keep
  this</pre>
<p>Last.</p></body></html>"""


def _extract(tmp_path: Path, body: str, name: str = "doc.html") -> list:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    profile = detect_file_type(path, DetectionConfig())
    return extract_text(path, profile)[0].elements


class TestHtml:
    @pytest.mark.parametrize("name", ["a.html", "a.htm", "A.HTML"])
    def test_detected_and_supported(self, tmp_path: Path, name: str) -> None:
        path = tmp_path / name
        path.write_text(PAGE, encoding="utf-8")
        profile = detect_file_type(path, DetectionConfig())
        assert profile.doc_type == DocumentType.HTML
        assert profile.has_tables is True
        assert {".html", ".htm"} <= SUPPORTED_EXTENSIONS

    def test_elements_in_document_order(self, tmp_path: Path) -> None:
        got = [(e.kind, e.text) for e in _extract(tmp_path, PAGE)]
        assert got == [
            ("heading", "Koala Audit"),
            ("paragraph", "First bold paragraph & more."),
            ("list_item", "one"),
            ("list_item", "two"),
            ("table", None),
            ("paragraph", "keep\n  this"),
            ("paragraph", "Last."),
        ]

    def test_table_cells_and_header(self, tmp_path: Path) -> None:
        table = next(e for e in _extract(tmp_path, PAGE) if e.kind == "table")
        assert table.header_rows == [0]
        assert [(c.row, c.col, c.value) for c in table.cells] == [
            (0, 0, "Site"), (0, 1, "Count"), (1, 0, "North"), (1, 1, "12"),
        ]

    def test_script_style_and_head_are_not_text(self, tmp_path: Path) -> None:
        text = " ".join(e.text or "" for e in _extract(tmp_path, PAGE))
        assert "not text" not in text and "color" not in text and "Ignored" not in text

    def test_unclosed_tags_and_bare_text_still_extract(self, tmp_path: Path) -> None:
        got = [(e.kind, e.text) for e in _extract(tmp_path, "bare<p>open<p>two")]
        assert got == [("paragraph", "bare"), ("paragraph", "open"), ("paragraph", "two")]

    def test_empty_document_has_no_elements(self, tmp_path: Path) -> None:
        assert _extract(tmp_path, "<html><body></body></html>") == []

    def test_latin1_fallback(self, tmp_path: Path) -> None:
        path = tmp_path / "l.html"
        path.write_bytes("<p>caf\xe9</p>".encode("latin-1"))
        profile = detect_file_type(path, DetectionConfig())
        assert extract_text(path, profile)[0].elements[0].text == "café"
