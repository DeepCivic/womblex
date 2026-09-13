"""Reviewer-facing rendering of a ground-truth segment.

A *segment* (:mod:`womblex.process.segmenter`) is a contiguous run of
elements from one source document; the renderer turns that run into the
markdown a human reads and corrects. Ground truth is keyed to element
identity, not to this rendering — reading order and formatting are
presentation and are regenerable, the corrected text is not — so the
renderer is a pure, order-preserving projection of the element stream,
inventing nothing beyond a closed set of structural delimiters.

**This is the narrative slice of that renderer.** Narrative-bearing
elements (:data:`~womblex.ingest.elements.TEXT_KINDS`) render verbatim,
one blank line between elements, via the same reassembly the chunker
uses (:func:`~womblex.process.chunker.reassemble_narrative`) so a
reviewed unit and a chunked one cannot drift into two coordinate spaces.
Table and form rendering — the other members of the closed structural
set — arrive in the next merge; until then a non-text element
contributes nothing to the output rather than being dropped from a place
it never occupied. On a narrative-only source (the per-page cohort the
first proof runs against) the two states are identical.

Guarantees held here, each pinned by a test:

- **Order.** Rendered narrative follows element order exactly.
  :func:`rendered_order` returns the ``order`` sequence as it appears in
  the output, which a test compares against the input sequence.
- **Verbatim.** The only bytes emitted are element text plus the
  blank-line join between elements. Redaction markers present in the
  element text survive unchanged because nothing here rewrites text.
- **Determinism** is inherent to a pure projection over an ordered list;
  the formal cross-process guarantee, and the extension of it over table
  numeric formatting, is U4 and lands with it.
"""

from __future__ import annotations

from collections.abc import Sequence

from womblex.ingest.elements import Element
from womblex.process.chunker import element_spans, reassemble_narrative

__all__ = ["render_elements", "rendered_order"]


def render_elements(elements: Sequence[Element]) -> str:
    """Render an ordered element run to reviewer-facing markdown.

    ``elements`` is a segment's element slice, assumed already sorted by
    ``order`` (a segment's range is contiguous and ordered by
    construction). Narrative-bearing elements render verbatim, separated
    by one blank line; non-text kinds contribute nothing in this merge.

    The narrative projection is :func:`reassemble_narrative`'s text —
    reused rather than reimplemented, so the bytes a reviewer corrects
    are the bytes the chunker sees.
    """
    text, _page_breaks = reassemble_narrative(list(elements))
    return text


def rendered_order(elements: Sequence[Element]) -> list[int]:
    """The element ``order`` values in the sequence they appear in the output.

    A pure narrative projection follows element order, so this is the
    ``order`` of each narrative-bearing element in output order — the
    handle a reading-order check compares against the source element
    sequence without parsing the rendered markdown back apart. Elements
    that contribute no text (empty text kinds, and every non-text kind in
    this merge) do not appear, exactly as they do not appear in the output.
    """
    return [order for order, _start, _end in element_spans(list(elements))]
