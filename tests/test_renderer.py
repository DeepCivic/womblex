"""Reviewer-facing rendering of a ground-truth segment (narrative slice)."""

from __future__ import annotations

from womblex.ingest.elements import Cell, Element
from womblex.process.chunker import reassemble_narrative
from womblex.process.renderer import render_elements, rendered_order


def para(order: int, text: str, page: int | None = 0) -> Element:
    return Element(order=order, kind="paragraph", extractor="test", page=page, text=text)


def table(order: int, page: int | None = 0) -> Element:
    cells = [Cell(row=r, col=c, value=f"r{r}c{c}") for r in range(2) for c in range(2)]
    return Element(
        order=order, kind="table", extractor="test", page=page,
        cells=cells, header_rows=[0],
    )


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


def test_reordering_elements_reorders_output() -> None:
    # Order is presentation: the same element identities in a new order
    # render their same texts in that new order, nothing reassigned.
    a, b = para(0, "alpha"), para(1, "beta")
    assert render_elements([a, b]) == "alpha\n\nbeta"
    forward = render_elements([Element(order=0, kind="paragraph", extractor="t", text="beta"),
                               Element(order=1, kind="paragraph", extractor="t", text="alpha")])
    assert forward == "beta\n\nalpha"


def test_non_text_kinds_contribute_nothing_yet() -> None:
    # Tables/forms are the next merge; a non-text element between two
    # paragraphs neither appears nor perturbs the narrative order.
    elements = [para(0, "before"), table(1), para(2, "after")]
    assert render_elements(elements) == "before\n\nafter"
    assert rendered_order(elements) == [0, 2]


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
