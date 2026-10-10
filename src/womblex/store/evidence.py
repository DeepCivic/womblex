"""The evidence reference: one shape for "where in the source this came from".

Every annotation sidecar that marks a span of source text (money, entity links,
PII) carries these columns in place of an anchor of its own, so a consumer
resolves any of them the same way: ``elem_order`` names the element the span
lies in (its start, for a span that crosses elements), ``page`` / ``bbox``
locate that element on the page, and ``char_start`` / ``char_end`` index the
text named by ``text_layer``:

==================  ======================================================
``text_layer``      what ``char_start`` / ``char_end`` index
==================  ======================================================
elements |          the document narrative reassembled from that
normalised |        element-text layer (the space chunks and enrichment
spellfix            mentions use)
``table_markdown``  the markdown of the table element ``elem_order`` (or
                    of the sheet ``sheet``) — the space table chunks and
                    table mentions use
``cell``            the value of one table cell (``elem_order`` +
                    ``cell_row`` / ``cell_col``) or one spreadsheet cell
                    (``elem_order`` of the ``sheet_cell`` element)
==================  ======================================================

``source_hash`` is the sidecar's own column and completes the reference. A span
that cannot be anchored carries nulls rather than a guess; the span check
(:mod:`womblex.process.evidence`) refuses a row whose anchor does not reproduce
its text.
"""

from __future__ import annotations

import pyarrow as pa

from womblex.store.output import _BBOX_TYPE

TABLE_LAYER = "table_markdown"
CELL_LAYER = "cell"

EVIDENCE_FIELDS: tuple[pa.Field, ...] = (
    pa.field("elem_order", pa.int32()),
    pa.field("page", pa.int32()),
    pa.field("bbox", _BBOX_TYPE),
    pa.field("sheet", pa.string()),
    pa.field("cell_row", pa.int32()),
    pa.field("cell_col", pa.int32()),
    pa.field("char_start", pa.int32()),
    pa.field("char_end", pa.int32()),
    pa.field("text_layer", pa.string()),
)
EVIDENCE_COLUMNS: tuple[str, ...] = tuple(f.name for f in EVIDENCE_FIELDS)


def no_evidence() -> dict[str, object]:
    """An unanchored reference: every evidence column null."""
    return dict.fromkeys(EVIDENCE_COLUMNS)


class EvidenceError(ValueError):
    """A span's evidence does not reproduce its text; the batch is not published."""


def backfill_evidence(
    raw: pa.Table, columns: dict[str, str] | None = None,
) -> pa.Table:
    """Add the evidence columns a pre-evidence file lacks.

    ``columns`` maps an evidence column to the legacy column that carried the
    same value unchanged (a cell's row and column, a sheet name); everything
    else is null. Narrative and chunk-relative offsets of the old shape cannot
    be re-expressed without the elements, so they are not carried over: re-run
    the stage for a file that needs them.
    """
    columns = columns or {}
    for field in EVIDENCE_FIELDS:
        if field.name in raw.schema.names:
            continue
        legacy = columns.get(field.name)
        if legacy is not None and legacy in raw.schema.names:
            raw = raw.append_column(field.name, raw.column(legacy).cast(field.type))
        else:
            raw = raw.append_column(field.name, pa.nulls(raw.num_rows, field.type))
    return raw
