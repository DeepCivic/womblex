"""Per-stage statistical figure annotation over an existing shard directory.

Consumes ``*.elements.parquet`` + ``*.table_cells.parquet`` and writes a
``*.stat_spans.parquet`` sibling per batch. An annotation op beside ``money``:
offline, API-free, no ordering dependency on any other stage, and it never
rewrites element or chunk text. Its input is the extraction parquet, never the
source files.

**Headings carry the context.** A document states the place and period its
figures refer to in a heading far more often than in the sentence, so every row
carries the headings it sits under (``heading_path``), and a figure whose
sentence names no period takes the one its headings state. Elements record no
heading level, so the path is built from order alone: the headings that open
the document (its title), then the most recent run of consecutive headings.

Two loci, each located by the shared evidence reference and checked against the
source before the sidecar is written:

- ``narrative`` — character offsets into the reassembled narrative, in the
  element-text layer named by ``text_source``;
- ``table_cell`` — a table read both ways: a row whose label cell names a
  measure, or a column whose header does. The other axis gives the period.

Page headers, footers and signatures are not scanned, and neither are
spreadsheet cells.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from womblex.config import StatsConfig
from womblex.ingest.elements import TEXT_KINDS, Element
from womblex.process.chunk_stage import _batch_bases, _load_elements
from womblex.process.chunker import element_spans
from womblex.process.evidence import EvidenceIndex, stamp_evidence
from womblex.process.stats import StatSpan, find_stats, read_cell_figure
from womblex.process.stats_periods import Period, find_periods
from womblex.process.stats_vocab import Measure, Vocabulary, load_vocabulary
from womblex.process.text_overlay import apply_overlay, load_overlay, require_overlays
from womblex.store.checkpoint import CheckpointManager
from womblex.store.money_output import quantise
from womblex.store.output import read_manifest
from womblex.store.stats_output import STAT_SPANS_SCHEMA, stat_spans_path_for, write_stat_spans

logger = logging.getLogger(__name__)

#: Page furniture: it repeats on every page and states nothing of its own.
_UNSCANNED = frozenset({"header", "footer", "signature"})


@dataclass
class StatsStageResult:
    batches_written: int
    spans_written: int
    figures_written: int


@dataclass
class _Headings:
    """The headings in force at one point of a document, from element order alone."""

    title: list[Element] = field(default_factory=list)
    section: list[Element] = field(default_factory=list)
    _in_run: bool = False
    _title_open: bool = True

    def see(self, elem: Element) -> None:
        if elem.kind == "heading":
            if self._title_open:
                self.title.append(elem)
            elif self._in_run:
                self.section.append(elem)
            else:
                self.section = [elem]
            self._in_run = True
        else:
            self._in_run = self._title_open = False

    @property
    def path(self) -> list[Element]:
        return [*self.title, *self.section]

    def period(self, *, above: Element | None = None) -> Period | None:
        """The period the innermost heading states; ``above`` excludes a heading and those after it."""
        path = self.path
        if above is not None:
            path = path[:next((i for i, h in enumerate(path) if h is above), len(path))]
        for heading in reversed(path):
            periods = find_periods(heading.text or "", bare_years=True)
            if periods:
                return periods[0]
        return None


def stats_shards(
    shard_dir: Path,
    config: StatsConfig,
    *,
    text_source: str = "elements",
    checkpoint_mgr: CheckpointManager | None = None,
) -> StatsStageResult:
    """Annotate statistical figures for every batch; write the sidecar."""
    if not shard_dir.is_dir():
        raise FileNotFoundError(f"shard directory not found: {shard_dir}")

    bases = _batch_bases(shard_dir)
    if not bases:
        logger.warning("stats_shards: no batches found in %s", shard_dir)
        return StatsStageResult(0, 0, 0)

    layer = config.text_source or text_source
    require_overlays(bases, layer)
    vocab = Vocabulary(load_vocabulary(config))
    if not vocab.measures:
        logger.warning(
            "stats_shards: no measures declared (stats.measures / stats.vocabulary); "
            "periods and cues are written, figures need a measure",
        )

    batches = spans = figures = 0
    for base in bases:
        if checkpoint_mgr is not None and _all_docs_checkpointed(base, checkpoint_mgr):
            logger.info("stats_shards: skipping %s (all docs checkpointed)", base.stem)
            continue
        rows = _annotate_batch(base, config, vocab, layer)
        write_stat_spans(rows, base, vocabulary_digest=vocab.digest)
        batches += 1
        spans += len(rows)
        figures += sum(1 for r in rows if r["kind"] == "figure")
        if checkpoint_mgr is not None and (doc_ids := _doc_ids(base)):
            checkpoint_mgr.update(
                doc_ids=doc_ids, succeeded=len(doc_ids), failed=0,
                batch_num=int(base.stem.replace("batch-", "") or 0),
            )
        logger.info("stats_shards: %s wrote %d spans", base.stem, len(rows))
    return StatsStageResult(batches, spans, figures)


def _annotate_batch(
    base_path: Path, config: StatsConfig, vocab: Vocabulary, text_source: str,
) -> list[dict]:
    elements_by_hash = _load_elements(base_path)
    overrides = load_overlay(base_path, text_source, required=True)
    rows: list[dict] = []
    indexes: dict[str, EvidenceIndex] = {}
    for source_hash, elements in elements_by_hash.items():
        apply_overlay(source_hash, elements, overrides)
        try:
            index = indexes[source_hash] = EvidenceIndex(elements, text_source)
            rows.extend(_document_rows(source_hash, elements, index, config, vocab))
        except Exception as e:  # one document must not stop the batch
            logger.warning("stats_shards: %s failed (%s)", source_hash, e)
    stamp_evidence(
        rows, lambda h, _layer: indexes.get(h), text_key="text",
        label="stat_spans", base=base_path,
    )
    return rows


def _document_rows(
    source_hash: str, elements: list[Element], index: EvidenceIndex,
    config: StatsConfig, vocab: Vocabulary,
) -> list[dict]:
    """One document's rows, walking its elements in order so headings stay current."""
    rows: list[dict] = []
    offsets = {order: end for order, _start, end in element_spans(elements)}
    headings = _Headings()
    for elem in elements:
        if elem.kind == "table" and elem.cells and config.tables:
            for span, where, period_source in _table_spans(elem, index, config, vocab, headings):
                rows.append(_row(source_hash, span, "table_cell", where, headings,
                                 context=None, period_source=period_source))
            continue
        if elem.kind not in TEXT_KINDS or not elem.text or elem.kind in _UNSCANNED:
            continue
        headings.see(elem)
        if not config.narrative or elem.order not in offsets:
            continue
        is_heading = elem.kind == "heading"
        base = offsets[elem.order] - len(elem.text)
        inherited = headings.period(above=elem if is_heading else None)
        for span in find_stats(elem.text, vocab, inherited=inherited, bare_years=is_heading,
                               default_currency=config.default_currency):
            if span.value is not None and quantise(span.value) is None:
                logger.warning("stats_shards: %s dropped un-storable value %r", source_hash, span.text)
                continue
            where = (index.narrative_ref(base + span.start, base + span.end)
                     or index.document_ref())
            rows.append(_row(
                source_hash, span, "narrative", where, headings,
                context=_context(elem.text, span, config.context_chars),
                period_source=span.period_source,
            ))
    return rows


def _table_spans(
    elem: Element, index: EvidenceIndex, config: StatsConfig, vocab: Vocabulary,
    headings: _Headings,
) -> list[tuple[StatSpan, dict, str | None]]:
    """Figures and gap cues in one table, each with its evidence and period source.

    A cell is a figure when its row's label cell, or its column's header, names
    a measure. Whichever axis does not name the measure states the period, and
    the headings do where neither axis does.
    """
    header_rows = set(elem.header_rows or [])
    by_row: dict[int, dict[int, str]] = defaultdict(dict)
    for cell in elem.cells or []:
        by_row[cell.row][cell.col] = cell.value or ""
    if not by_row:
        return []
    label_col = min(min(cols) for cols in by_row.values())
    headers: dict[int, str] = defaultdict(str)
    for r in sorted(header_rows):
        for col, value in by_row.get(r, {}).items():
            headers[col] = f"{headers[col]} {value}".strip()
    col_measure = {c: _first_measure(t, vocab) for c, t in headers.items()}
    col_period = {c: _first_period(t) for c, t in headers.items()}
    heading_period = headings.period()

    out: list[tuple[StatSpan, dict, str | None]] = []
    for r in sorted(set(by_row) - header_rows):
        label = by_row[r].get(label_col, "")
        row_measure, row_period = _first_measure(label, vocab), _first_period(label)
        for col, value in sorted(by_row[r].items()):
            measure = row_measure or col_measure.get(col)
            if col == label_col or measure is None:
                continue
            span = read_cell_figure(value, config.default_currency) or _gap_cue(value, vocab)
            if span is None:
                continue
            span.measure = measure
            period_source: str | None = None
            if span.kind == "figure":
                span.unit = span.unit or measure.unit
                # The axis that names the measure cannot also be the one stating the period.
                choices = [("column", col_period.get(col)), ("row", row_period)]
                if row_measure is None:
                    choices.reverse()
                choices.append(("heading", heading_period))
                period_source, span.period = next(
                    ((s, p) for s, p in choices if p is not None), (None, None))
            where = (index.cell_ref(elem.order, span.start, span.end, cell=(r, col))
                     or index.element_ref(elem.order, cell=(r, col)) or index.document_ref())
            out.append((span, where, period_source))
    return out


def _first_measure(text: str, vocab: Vocabulary) -> Measure | None:
    found = vocab.find_measures(text) if text else []
    return vocab.measures[found[0].kind] if found else None


def _first_period(text: str) -> Period | None:
    periods = find_periods(text, bare_years=True) if text else []
    return periods[0] if periods else None


def _gap_cue(value: str, vocab: Vocabulary) -> StatSpan | None:
    """A cell that is a gap marker and nothing else (``n.a.``, ``not available``)."""
    stripped = value.strip()
    offset = value.index(stripped) if stripped else 0
    for cue in vocab.find_cues(stripped):
        if cue.cue_type == "gap" and (cue.start, cue.end) == (0, len(stripped)):
            return StatSpan("cue", stripped, offset, offset + len(stripped),
                            cue_type="gap", cue=cue.kind)
    return None


def _context(text: str, span: StatSpan, width: int) -> str:
    lo, hi = max(0, span.start - width // 2), min(len(text), span.end + width // 2)
    return text[lo:hi]


_EMPTY_ROW: dict[str, object] = {f.name: None for f in STAT_SPANS_SCHEMA}


def _row(
    source_hash: str, span: StatSpan, locus: str, where: dict, headings: _Headings, *,
    context: str | None, period_source: str | None,
) -> dict:
    row = dict(_EMPTY_ROW)
    row.update(where)
    period, measure = span.period, span.measure
    value = quantise(span.value) if span.value is not None else None
    path = headings.path
    row.update({
        "source_hash": source_hash, "kind": span.kind, "locus": locus, "text": span.text,
        "measure_id": measure.id if measure else None,
        "measure_label": measure.label if measure else None,
        "value": value, "unit": span.unit, "rate_per": span.rate_per,
        "multiplier": span.multiplier,
        "period": period.label if period else None,
        "period_kind": period.kind if period else None,
        "period_start": period.period_start if period else None,
        "period_end": period.period_end if period else None,
        "period_source": period_source if period else None,
        "cue_type": span.cue_type, "cue": span.cue,
        "calculation": span.calculation, "source": span.source, "status": span.status,
        "heading_path": [h.text or "" for h in path],
        "heading_elem_orders": [h.order for h in path],
        "context": context,
    })
    return row


def _doc_ids(base_path: Path) -> list[str]:
    try:
        return list(read_manifest(base_path).column("doc_id").to_pylist())
    except Exception:
        return []


def _all_docs_checkpointed(base_path: Path, mgr: CheckpointManager) -> bool:
    if not stat_spans_path_for(base_path).exists():
        return False
    doc_ids = _doc_ids(base_path)
    return bool(doc_ids) and all(d in mgr.state.processed_ids for d in doc_ids)


__all__ = ["StatsStageResult", "stats_shards"]
