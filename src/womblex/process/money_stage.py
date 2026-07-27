"""Per-stage monetary amount annotation over an existing shard directory.

Consumes ``*.elements.parquet`` + ``*.table_cells.parquet`` and writes
``*.money_spans.parquet`` and ``*.money_columns.parquet`` siblings per batch.

An **annotation op** in the mould of ``quality``: offline, API-free, no
ordering dependency on enrich, and it never rewrites element or chunk text.
Its input is the extraction parquet, never the source files — a second reader
would be a parallel extraction path, which ``docs/money.md`` rejects.

Three loci, two coordinate spaces, never mixed:

- ``narrative`` — character offsets into the reassembled narrative, in the
  element-text layer named by ``text_source`` (stamped on every row), so the
  spans share enrichment's coordinate space;
- ``table_cell`` — ``(parent_elem_order, row, col)`` on the table_cells sidecar;
- ``sheet_cell`` — ``(sheet, row, col)``.

Mirrors :mod:`womblex.process.normalise_stage`: per-stage ``CheckpointManager``,
skip-existing on resume, batch-level isolation.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from womblex.config import MoneyConfig
from womblex.ingest.elements import Element
from womblex.process.chunk_stage import _batch_bases, _load_elements
from womblex.process.chunker import reassemble_narrative
from womblex.process.money import MoneyOptions, MoneySpan, context_for, find_money
from womblex.process.money_columns import (
    ColumnOptions,
    ColumnVerdict,
    classify_column,
    extract_column,
)
from womblex.process.text_overlay import apply_overlay, load_overlay
from womblex.store.checkpoint import CheckpointManager
from womblex.store.money_output import (
    money_spans_path_for,
    quantise,
    write_money_columns,
    write_money_spans,
)
from womblex.store.output import read_manifest

logger = logging.getLogger(__name__)


@dataclass
class MoneyStageResult:
    batches_written: int
    spans_written: int
    columns_classified: int
    money_columns: int


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def money_shards(
    shard_dir: Path,
    config: MoneyConfig,
    *,
    text_source: str = "elements",
    checkpoint_mgr: CheckpointManager | None = None,
) -> MoneyStageResult:
    """Annotate monetary amounts for every batch; write the two siblings."""
    if not shard_dir.is_dir():
        raise FileNotFoundError(f"shard directory not found: {shard_dir}")

    bases = _batch_bases(shard_dir)
    if not bases:
        logger.warning("money_shards: no batches found in %s", shard_dir)
        return MoneyStageResult(0, 0, 0, 0)

    layer = config.text_source or text_source
    opts = MoneyOptions(
        default_currency=config.default_currency,
        international_numbers=config.international_numbers,
        implicit_context=config.implicit_context,
        min_confidence=config.min_confidence,
        context_chars=config.context_chars,
    )
    col_opts = ColumnOptions(
        default_currency=config.default_currency,
        numeric_fraction_min=config.columns.numeric_fraction_min,
        min_cells=config.columns.min_cells,
        extra_header_terms=frozenset(t.lower() for t in config.columns.extra_header_terms),
        extra_veto_terms=frozenset(t.lower() for t in config.columns.extra_veto_terms),
        international_numbers=config.international_numbers,
    )

    batches_written = spans_written = columns_classified = money_columns = 0

    for base in bases:
        if checkpoint_mgr is not None and _all_docs_checkpointed(base, checkpoint_mgr):
            logger.info("money_shards: skipping %s (all docs checkpointed)", base.stem)
            continue

        span_rows, column_rows = _annotate_batch(base, config, opts, col_opts, layer)

        write_money_spans(span_rows, base)
        write_money_columns(column_rows, base)
        batches_written += 1
        spans_written += len(span_rows)
        columns_classified += len(column_rows)
        money_columns += sum(1 for r in column_rows if r["verdict"] == "money")

        if checkpoint_mgr is not None:
            doc_ids = _doc_ids(base)
            if doc_ids:
                checkpoint_mgr.update(
                    doc_ids=doc_ids,
                    succeeded=len(doc_ids),
                    failed=0,
                    batch_num=int(base.stem.replace("batch-", "") or 0),
                )

        logger.info(
            "money_shards: %s wrote %d spans over %d columns",
            base.stem, len(span_rows), len(column_rows),
        )

    return MoneyStageResult(
        batches_written=batches_written,
        spans_written=spans_written,
        columns_classified=columns_classified,
        money_columns=money_columns,
    )


def _annotate_batch(
    base_path: Path, config: MoneyConfig, opts: MoneyOptions,
    col_opts: ColumnOptions, text_source: str,
) -> tuple[list[dict], list[dict]]:
    """Return ``(span_rows, column_rows)`` for one batch."""
    elements_by_hash = _load_elements(base_path)
    overrides = load_overlay(base_path, text_source)

    span_rows: list[dict] = []
    column_rows: list[dict] = []
    for source_hash, elements in elements_by_hash.items():
        apply_overlay(source_hash, elements, overrides)
        try:
            if config.narrative:
                span_rows.extend(_narrative_rows(source_hash, elements, opts, text_source))
            if config.columns.enabled:
                s, c = _table_rows(source_hash, elements, opts, col_opts)
                span_rows.extend(s)
                column_rows.extend(c)
                s, c = _sheet_rows(source_hash, elements, opts, col_opts)
                span_rows.extend(s)
                column_rows.extend(c)
        except Exception as e:  # one document must not stop the batch
            logger.warning("money_shards: %s failed (%s)", source_hash, e)
    return span_rows, column_rows


# ---------------------------------------------------------------------------
# Narrative locus
# ---------------------------------------------------------------------------


def _narrative_rows(
    source_hash: str, elements: list[Element], opts: MoneyOptions, text_source: str,
) -> list[dict]:
    text, page_breaks = reassemble_narrative(elements)
    if not text:
        return []
    rows = []
    for span in find_money(text, opts):
        value = quantise(span.value)
        if value is None:
            logger.warning(
                "money_shards: %s dropped un-storable value %r at %d",
                source_hash, span.text, span.start,
            )
            continue
        rows.append(_span_row(
            source_hash, span, value, locus="narrative", text_source=text_source,
            start_char=span.start, end_char=span.end,
            page=_page_for(span.start, page_breaks),
            context=context_for(text, span, opts.context_chars),
        ))
    return rows


def _page_for(offset: int, page_breaks: list[tuple[int, int]]) -> int | None:
    """Resolve a narrative offset to its page via ``reassemble_narrative`` spans."""
    for end, page in page_breaks:
        if offset < end:
            return page
    return None


# ---------------------------------------------------------------------------
# Cell loci
# ---------------------------------------------------------------------------


def _table_rows(
    source_hash: str, elements: list[Element], opts: MoneyOptions, col_opts: ColumnOptions,
) -> tuple[list[dict], list[dict]]:
    """Column-evidenced + self-evidencing amounts in ``kind='table'`` elements."""
    span_rows: list[dict] = []
    column_rows: list[dict] = []
    for elem in elements:
        if elem.kind != "table" or not elem.cells:
            continue
        by_col: dict[int, list[tuple[int, str]]] = defaultdict(list)
        for cell in elem.cells:
            by_col[cell.col].append((cell.row, cell.value or ""))

        # No declared header row means no recoverable unit or vocabulary: bare
        # cells are left alone rather than guessed at (docs/money.md).
        header_rows = sorted(elem.header_rows or [])
        header_by_col = _header_texts(by_col, header_rows)

        for col, entries in sorted(by_col.items()):
            entries.sort()
            body = [(r, v) for r, v in entries if r not in set(header_rows)]
            header = header_by_col.get(col, "")
            verdict = (
                classify_column(header, [v for _, v in body], options=col_opts)
                if header_rows else
                ColumnVerdict(verdict="insufficient", evidence="no_header",
                              header_text="", cells_total=len(body))
            )
            column_id = f"elem{elem.order}:col{col}"
            extracted = _cell_spans(
                source_hash, body, verdict, opts, col_opts, column_id,
                locus="table_cell", parent_elem_order=elem.order, col=col,
            )
            span_rows.extend(extracted)
            if header_rows:
                column_rows.append(_column_row(
                    source_hash, column_id, "table_cell", verdict, col,
                    parent_elem_order=elem.order, sheet=None,
                    cells_extracted=len(extracted),
                ))
    return span_rows, column_rows


def _sheet_rows(
    source_hash: str, elements: list[Element], opts: MoneyOptions, col_opts: ColumnOptions,
) -> tuple[list[dict], list[dict]]:
    """Column-evidenced + self-evidencing amounts in spreadsheet ``sheet_cell``s.

    Row 0 is the header, matching the established sheet projection. The cells'
    ``number_format`` is the strongest evidence available anywhere in the
    corpus — for a register whose money column is bare digits, it is the only
    currency marker in the file.
    """
    span_rows: list[dict] = []
    column_rows: list[dict] = []
    sheets: dict[str, dict[int, list[tuple[int, str, int, str | None]]]] = defaultdict(
        lambda: defaultdict(list))
    for e in elements:
        if e.kind != "sheet_cell" or e.sheet is None or e.row is None or e.col is None:
            continue
        sheets[e.sheet][e.col].append((e.row, e.value or "", e.order, e.number_format))

    for sheet, cols in sheets.items():
        for col, entries in sorted(cols.items()):
            entries.sort()
            header = next((v for r, v, _, _ in entries if r == 0), "")
            body = [(r, v, order) for r, v, order, _ in entries if r != 0]
            fmt = _dominant_format([f for r, _, _, f in entries if r != 0])
            verdict = classify_column(
                header, [v for _, v, _ in body], number_format=fmt, options=col_opts)
            column_id = f"sheet:{sheet}:col{col}"
            extracted = _cell_spans(
                source_hash, [(r, v) for r, v, _ in body], verdict, opts, col_opts,
                column_id, locus="sheet_cell", sheet=sheet, col=col,
                orders={r: o for r, _, o in body},
            )
            span_rows.extend(extracted)
            column_rows.append(_column_row(
                source_hash, column_id, "sheet_cell", verdict, col,
                parent_elem_order=None, sheet=sheet, cells_extracted=len(extracted),
            ))
    return span_rows, column_rows


def _cell_spans(
    source_hash: str,
    body: list[tuple[int, str]],
    verdict: ColumnVerdict,
    opts: MoneyOptions,
    col_opts: ColumnOptions,
    column_id: str,
    *,
    locus: str,
    col: int,
    parent_elem_order: int | None = None,
    sheet: str | None = None,
    orders: dict[int, int] | None = None,
) -> list[dict]:
    """Amounts for one column's cells.

    A classified money column owns its cells — the column verdict supplies
    currency, scale and the accounting-negative gate, and per-cell rescanning
    would double-count. Cells in every other column are still scanned for
    *self-evidencing* amounts: a ``$1,200.50`` cell carries its own evidence
    whatever its header says.
    """
    rows: list[dict] = []
    if verdict.is_money:
        values = [v for _, v in body]
        for idx, value, negative in extract_column(values, verdict, options=col_opts):
            stored = quantise(value)
            if stored is None:
                continue
            row_idx = body[idx][0]
            rows.append(_cell_row(
                source_hash, locus=locus, text=body[idx][1], value=stored,
                currency=verdict.currency, currency_source=(
                    "number_format" if verdict.evidence == "number_format" else "column_header"),
                evidence=verdict.evidence, confidence=verdict.confidence,
                negative=negative, multiplier=verdict.scale, column_id=column_id,
                row=row_idx, col=col, parent_elem_order=parent_elem_order, sheet=sheet,
                elem_order=(orders or {}).get(row_idx),
            ))
        return rows

    for row_idx, text in body:
        if not any(ch.isdigit() for ch in text):
            continue  # every pattern needs a digit; skip the scan on prose cells
        for span in find_money(text, opts):
            stored = quantise(span.value)
            if stored is None:
                continue
            rows.append(_cell_row(
                source_hash, locus=locus, text=span.text, value=stored,
                currency=span.currency, currency_source=span.currency_source,
                evidence=span.evidence, confidence=span.confidence,
                negative=span.negative, multiplier=span.multiplier, column_id=None,
                row=row_idx, col=col, parent_elem_order=parent_elem_order, sheet=sheet,
                elem_order=(orders or {}).get(row_idx), modifier=span.modifier,
                context=text[:200],
            ))
    return rows


def _header_texts(
    by_col: dict[int, list[tuple[int, str]]], header_rows: list[int],
) -> dict[int, str]:
    """Header text per column — every declared header row, joined."""
    if not header_rows:
        return {}
    wanted = set(header_rows)
    return {
        col: " ".join(v for r, v in sorted(entries) if r in wanted and v).strip()
        for col, entries in by_col.items()
    }


def _dominant_format(formats: list[str | None]) -> str | None:
    present = [f for f in formats if f]
    if not present:
        return None
    return Counter(present).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------


_EMPTY_ROW: dict[str, object] = {f: None for f in (
    "source_hash", "locus", "text_source", "start_char", "end_char", "page",
    "elem_order", "parent_elem_order", "sheet", "row", "col", "text", "value",
    "currency", "currency_source", "evidence", "modifier", "multiplier",
    "negative", "confidence", "range_group", "range_role", "column_id", "context",
)}


def _span_row(
    source_hash: str, span: MoneySpan, value: Decimal, *, locus: str,
    text_source: str, start_char: int, end_char: int, page: int | None, context: str,
) -> dict:
    row = dict(_EMPTY_ROW)
    row.update({
        "source_hash": source_hash, "locus": locus, "text_source": text_source,
        "start_char": start_char, "end_char": end_char, "page": page,
        "text": span.text, "value": value, "currency": span.currency,
        "currency_source": span.currency_source, "evidence": span.evidence,
        "modifier": span.modifier, "multiplier": span.multiplier,
        "negative": span.negative, "confidence": span.confidence,
        "range_group": span.range_group, "range_role": span.range_role,
        "context": context,
    })
    return row


def _cell_row(
    source_hash: str, *, locus: str, text: str, value: Decimal,
    currency: str | None, currency_source: str, evidence: str, confidence: float,
    negative: bool, multiplier: str | None, column_id: str | None,
    row: int, col: int, parent_elem_order: int | None, sheet: str | None,
    elem_order: int | None, modifier: str | None = None, context: str | None = None,
) -> dict:
    out = dict(_EMPTY_ROW)
    out.update({
        "source_hash": source_hash, "locus": locus, "text": text, "value": value,
        "currency": currency, "currency_source": currency_source,
        "evidence": evidence, "confidence": confidence, "negative": negative,
        "multiplier": multiplier, "column_id": column_id, "row": row, "col": col,
        "parent_elem_order": parent_elem_order, "sheet": sheet,
        "elem_order": elem_order, "modifier": modifier, "context": context,
    })
    return out


def _column_row(
    source_hash: str, column_id: str, locus: str, verdict: ColumnVerdict, col: int,
    *, parent_elem_order: int | None, sheet: str | None, cells_extracted: int,
) -> dict:
    return {
        "source_hash": source_hash, "column_id": column_id, "locus": locus,
        "parent_elem_order": parent_elem_order, "sheet": sheet, "col": col,
        "header_text": verdict.header_text, "number_format": verdict.number_format,
        "verdict": verdict.verdict, "evidence": verdict.evidence,
        "veto_term": verdict.veto_term, "currency": verdict.currency,
        "scale": verdict.scale, "numeric_fraction": verdict.numeric_fraction,
        "null_fraction": verdict.null_fraction, "confidence": verdict.confidence,
        "cells_total": verdict.cells_total, "cells_extracted": cells_extracted,
    }


def _doc_ids(base_path: Path) -> list[str]:
    try:
        return list(read_manifest(base_path).column("doc_id").to_pylist())
    except Exception:
        return []


def _all_docs_checkpointed(base_path: Path, mgr: CheckpointManager) -> bool:
    if not money_spans_path_for(base_path).exists():
        return False
    doc_ids = _doc_ids(base_path)
    return bool(doc_ids) and all(d in mgr.state.processed_ids for d in doc_ids)


__all__ = ["MoneyStageResult", "money_shards"]
