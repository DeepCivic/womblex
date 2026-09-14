"""Reviewer-facing rendering of a ground-truth segment: narrative, tables, forms."""

from __future__ import annotations

from womblex.ingest.elements import Cell, Element, FieldEntry
from womblex.process.chunker import reassemble_narrative, table_to_markdown
from womblex.process.renderer import render_elements, rendered_order


def para(order: int, text: str, page: int | None = 0) -> Element:
    return Element(order=order, kind="paragraph", extractor="test", page=page, text=text)


def table(order: int, page: int | None = 0) -> Element:
    cells = [Cell(row=r, col=c, value=f"r{r}c{c}") for r in range(2) for c in range(2)]
    return Element(
        order=order, kind="table", extractor="test", page=page,
        cells=cells, header_rows=[0],
    )


def form(order: int, fields: list[tuple[str, str]], page: int | None = 0) -> Element:
    return Element(
        order=order, kind="form", extractor="test", page=page,
        fields=[FieldEntry(name=n, value=v) for n, v in fields],
    )


def _is_subsequence(needle: str, haystack: str) -> bool:
    it = iter(haystack)
    return all(ch in it for ch in needle)


# The closed structural set the renderer may introduce: a blank line between
# elements, GFM table pipes + separator dashes + cell padding, and the `: `
# of a form line. Stripping these characters from the rendered output must
# leave only content that already appears, in order, in the element text.
_STRUCTURAL_CHARS = set("|-: \t\n")


def _strip_structural(text: str) -> str:
    return "".join(c for c in text if c not in _STRUCTURAL_CHARS)


def _element_content(e: Element) -> str:
    """The raw content bytes an element carries, no structural delimiters."""
    if e.text:
        return e.text
    if e.kind == "table":
        return "".join(c.value for c in sorted(e.cells or [], key=lambda c: (c.row, c.col)))
    if e.kind == "form":
        return "".join(f.name + f.value for f in e.fields or [])
    return ""


# ---------------------------------------------------------------------------
# Narrative (merge 1, unchanged)
# ---------------------------------------------------------------------------


def test_narrative_renders_verbatim_blank_line_separated() -> None:
    elements = [para(0, "First paragraph."), para(1, "Second paragraph.")]
    assert render_elements(elements) == "First paragraph.\n\nSecond paragraph."


def test_render_reuses_reassemble_narrative() -> None:
    # The reviewer corrects the bytes the chunker sees — one coordinate space.
    elements = [para(0, "alpha"), para(1, "beta"), para(2, "gamma")]
    assert render_elements(elements) == reassemble_narrative(elements)[0]


def test_redaction_markers_survive_unchanged() -> None:
    elements = [para(0, "Signed by <REDACTED> on the date shown.")]
    assert "<REDACTED>" in render_elements(elements)


def test_rendered_order_follows_element_order() -> None:
    elements = [para(2, "c"), para(0, "a"), para(1, "b")]
    ordered = sorted(elements, key=lambda e: e.order)
    assert rendered_order(ordered) == [0, 1, 2]


def test_empty_text_element_omitted() -> None:
    elements = [para(0, "kept"), para(1, ""), para(2, "also kept")]
    assert render_elements(elements) == "kept\n\nalso kept"
    assert rendered_order(elements) == [0, 2]


def test_empty_stream_renders_empty() -> None:
    assert render_elements([]) == ""
    assert rendered_order([]) == []


def test_render_is_deterministic_across_calls() -> None:
    elements = [para(0, "one"), para(1, "two"), para(2, "three")]
    assert render_elements(elements) == render_elements(list(elements))


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def test_table_renders_as_gfm_markdown() -> None:
    out = render_elements([table(0)])
    # Header row, one separator row, one data row — the GFM shape.
    assert out == "| r0c0 | r0c1 |\n| --- | --- |\n| r1c0 | r1c1 |"


def test_table_reuses_chunker_table_to_markdown() -> None:
    # Same projection the segmenter budgets against — no second markdown code.
    t = table(0)
    assert render_elements([t]) == table_to_markdown(["r0c0", "r0c1"], [["r1c0", "r1c1"]])


def test_table_interleaved_at_document_position_not_appended() -> None:
    elements = [para(0, "before"), table(1), para(2, "after")]
    out = render_elements(elements)
    assert out == (
        "before\n\n"
        "| r0c0 | r0c1 |\n| --- | --- |\n| r1c0 | r1c1 |\n\n"
        "after"
    )
    assert rendered_order(elements) == [0, 1, 2]


def test_empty_table_contributes_nothing() -> None:
    empty = Element(order=1, kind="table", extractor="test", page=0, cells=[])
    elements = [para(0, "before"), empty, para(2, "after")]
    assert render_elements(elements) == "before\n\nafter"
    assert rendered_order(elements) == [0, 2]


def test_redaction_marker_in_table_cell_survives() -> None:
    cell = Element(
        order=0, kind="table", extractor="test", page=0,
        cells=[Cell(row=0, col=0, value="Name"), Cell(row=1, col=0, value="<REDACTED>")],
        header_rows=[0],
    )
    assert "<REDACTED>" in render_elements([cell])


# ---------------------------------------------------------------------------
# Forms
# ---------------------------------------------------------------------------


def test_form_renders_label_value_lines() -> None:
    out = render_elements([form(0, [("Name", "Jane Doe"), ("Date", "2026-01-01")])])
    assert out == "Name: Jane Doe\nDate: 2026-01-01"


def test_form_interleaved_between_narrative_and_table() -> None:
    elements = [para(0, "intro"), form(1, [("Ref", "A1")]), table(2), para(3, "outro")]
    out = render_elements(elements)
    assert out == (
        "intro\n\n"
        "Ref: A1\n\n"
        "| r0c0 | r0c1 |\n| --- | --- |\n| r1c0 | r1c1 |\n\n"
        "outro"
    )
    assert rendered_order(elements) == [0, 1, 2, 3]


def test_empty_form_contributes_nothing() -> None:
    empty = Element(order=1, kind="form", extractor="test", page=0, fields=[])
    elements = [para(0, "before"), empty, para(2, "after")]
    assert render_elements(elements) == "before\n\nafter"
    assert rendered_order(elements) == [0, 2]


def test_redaction_marker_in_form_field_survives() -> None:
    out = render_elements([form(0, [("Signatory", "<REDACTED>")])])
    assert "<REDACTED>" in out


# ---------------------------------------------------------------------------
# Order and invent-nothing across the whole closed set
# ---------------------------------------------------------------------------


def test_consecutive_tables_and_forms_keep_document_order() -> None:
    elements = [table(0), form(1, [("K", "V")]), table(2)]
    assert rendered_order(elements) == [0, 1, 2]
    out = render_elements(elements)
    assert out.index("K: V") < out.index("| r0c0 | r0c1 |", out.index("K: V"))


def test_rendered_order_omits_non_contributing_kinds() -> None:
    elements = [
        para(0, "text"),
        Element(order=1, kind="page_break", extractor="test", page=1),
        Element(order=2, kind="image", extractor="test", page=1, alt_text="a logo"),
        table(3),
    ]
    assert rendered_order(elements) == [0, 3]


def test_rendered_output_minus_structural_set_is_subsequence_of_element_text() -> None:
    # The invent-nothing guarantee: strip the closed structural delimiters
    # from the render and the remainder must appear, in order, in the raw
    # element/cell/field content — so any injected placeholder or label fails.
    elements = [
        para(0, "A cover note about the schedule."),
        form(1, [("Reference", "FOI-2026-001"), ("Officer", "<REDACTED>")]),
        table(2),
        para(3, "A closing remark."),
    ]
    rendered = _strip_structural(render_elements(elements))
    content = _strip_structural("".join(_element_content(e) for e in elements))
    assert _is_subsequence(rendered, content)
