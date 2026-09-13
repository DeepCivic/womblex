"""Ground-truth segmentation: element ranges under a token budget.

A *segment* is a contiguous run of elements from one source document,
small enough for a human to review in one sitting. It is the unit
ground truth is keyed to, so its boundaries must be a function of the
element stream and the configuration alone — never of a stage's output.
Deriving them from chunk boundaries would make the ground-truth unit a
function of the chunker, and the chunker is one of the things being
measured against it.

Two constraints, applied in that order:

- a **page ceiling** partitions the stream into runs no wider than N
  pages (a reviewer holds a handful of pages in view, not forty), and
- a **token budget** packs each run into segments via
  :func:`womblex.utils.token_packer.pack_by_tokens` — the same greedy
  grouping the Isaacus request packer uses, including its solo path for
  an item that does not fit on its own.

Splits happen only between elements, so a table and its cells, or a
form and its fields, are always wholly inside one segment: they nest
within a single :class:`~womblex.ingest.elements.Element`.

Segments of one document tile it — gapless and non-overlapping — which
is what lets a future re-cut of boundaries be arithmetic over existing
corrections rather than a re-review.

The budget is in tokens because that is what the packer counts. The
human-facing word count is recorded alongside on each segment rather
than converted at the boundary, so no rounding sits between the two.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from itertools import pairwise

from womblex.config import SegmentationConfig
from womblex.ingest.elements import TEXT_KINDS, Element
from womblex.ingest.views import _element_to_table_data
from womblex.process.chunker import table_to_markdown
from womblex.utils.token_packer import pack_by_tokens

CountFn = Callable[[list[str]], list[int]]

# One `label: value` line per form field — the same closed set of
# structural delimiters the reviewer-facing renderer is allowed to emit,
# so a segment's measured size matches what a reviewer will be handed.
FIELD_JOIN = "\n"


@dataclass(frozen=True, slots=True)
class Segment:
    """One reviewable run of elements, and what it measured.

    ``element_range`` and ``page_range`` are half-open, over ``elem_order``
    and zero-based page indices respectively. ``page_range`` is ``None``
    when no element in the segment carries a page — a source with no page
    concept (DOCX, spreadsheet) segments on the token budget alone.

    ``oversize`` marks the one segment shape the budget cannot honour: a
    single element whose own token count exceeds the budget, emitted alone
    because there is no boundary inside an element to split at.
    """

    element_range: tuple[int, int]
    page_range: tuple[int, int] | None
    tokens: int
    words: int
    oversize: bool


def element_text(element: Element) -> str:
    """The text an element contributes to its segment's measured size.

    Narrative kinds contribute their verbatim text; a table contributes its
    markdown projection and a form its ``label: value`` lines, so a segment
    is budgeted against what a reviewer actually reads. Structural elements
    with no text (page breaks, sheet metadata) contribute nothing and are
    carried by whichever segment they fall in — dropping them would leave a
    gap in the tiling.
    """
    if element.kind in TEXT_KINDS:
        return element.text or ""
    if element.kind == "table":
        td = _element_to_table_data(element)
        return table_to_markdown(td.headers, td.rows)
    if element.kind == "form":
        return FIELD_JOIN.join(f"{f.name}: {f.value}" for f in element.fields or [])
    if element.kind in ("image", "figure"):
        return element.alt_text or ""
    if element.kind == "sheet_cell":
        return element.value or ""
    return ""


def segment_elements(
    elements: Sequence[Element],
    count_fn: CountFn,
    config: SegmentationConfig,
) -> list[Segment]:
    """Cut an ordered element stream into budgeted, page-bounded segments.

    ``count_fn`` maps a list of texts to their exact token counts — e.g.
    ``TokenCounter().count_batch``. It is passed in rather than constructed
    here so segmentation stays offline-testable and so the caller chooses
    the tokeniser its budget is expressed in.

    Raises ``ValueError`` if the stream is not strictly increasing by
    ``order`` (segment ranges would not tile), or if an element exceeds the
    budget on its own under ``oversize='error'``.
    """
    ordered = list(elements)
    _check_monotonic(ordered)
    if not ordered:
        return []

    by_order = {e.order: e for e in ordered}
    segments: list[Segment] = []

    for run in _page_runs(ordered, config.page_ceiling):
        items = [(str(e.order), element_text(e)) for e in run]
        for group in pack_by_tokens(
            items, count_fn, max_items=len(items), token_budget=config.token_budget
        ):
            members = [by_order[int(item.key)] for item in group.items]
            oversize = group.total_tokens > config.token_budget
            if oversize and config.oversize == "error":
                raise ValueError(
                    f"element {members[0].order} ({members[0].kind}) is "
                    f"{group.total_tokens} tokens, over the {config.token_budget}-token "
                    "budget, and has no internal boundary to split at"
                )
            segments.append(
                Segment(
                    element_range=(members[0].order, members[-1].order + 1),
                    page_range=_page_range(members),
                    tokens=group.total_tokens,
                    words=sum(len(item.text.split()) for item in group.items),
                    oversize=oversize,
                )
            )

    return segments


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_monotonic(elements: Sequence[Element]) -> None:
    for prev, curr in pairwise(elements):
        if curr.order <= prev.order:
            raise ValueError(
                "element stream must be sorted by strictly increasing order; "
                f"saw {prev.order} then {curr.order}"
            )


def _page_runs(elements: Sequence[Element], page_ceiling: int) -> Iterator[list[Element]]:
    """Partition into runs spanning at most ``page_ceiling`` pages.

    Elements with no page (spreadsheet cells, DOCX paragraphs) never widen
    the span, so a source with no page concept yields exactly one run and
    is bounded by the token budget alone.
    """
    run: list[Element] = []
    low: int | None = None
    high: int | None = None

    for e in elements:
        if e.page is not None:
            new_low = e.page if low is None else min(low, e.page)
            new_high = e.page if high is None else max(high, e.page)
            if run and new_high - new_low + 1 > page_ceiling:
                yield run
                run, low, high = [], e.page, e.page
            else:
                low, high = new_low, new_high
        run.append(e)

    if run:
        yield run


def _page_range(members: Sequence[Element]) -> tuple[int, int] | None:
    pages = [e.page for e in members if e.page is not None]
    if not pages:
        return None
    return (min(pages), max(pages) + 1)


__all__ = ["Segment", "element_text", "segment_elements"]
