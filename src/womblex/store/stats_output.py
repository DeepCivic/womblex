"""``*.stat_spans.parquet`` schema + IO — the ``stats`` annotation op's sidecar.

One row per span, joinable on ``source_hash`` and discriminated by ``kind``:

==========  ==============================================================
``period``  a reference period: ``period`` (canonical label), ``period_kind``
            and, where the text fixes them, ``period_start`` / ``period_end``
``measure`` a phrase the corpus vocabulary names: ``measure_id``
``cue``     a ``calculation`` / ``source`` / ``gap`` / ``forecast`` cue, with
            the measure it sits beside, if any
``figure``  a number stated beside a measure: ``value`` and ``unit``, with the
            measure, period, calculation, source and status its sentence (or,
            for the period, its headings) gives it
==========  ==============================================================

Two loci share the file. A ``narrative`` row's offsets index the reassembled
narrative under ``text_layer``, the space enrichment mentions and money spans
use; a ``table_cell`` row names the table element and ``cell_row`` /
``cell_col``, with offsets into that cell's value. Both are located by the
shared evidence reference (:mod:`womblex.store.evidence`).

``heading_path`` is the text of the headings a row sits under, outermost first,
and ``heading_elem_orders`` their elements: the place and period a figure refers
to are usually stated there and not in its sentence.

``value`` is ``decimal128(38, 4)``, exact, as for money. The footer names the
vocabulary that produced the file (``womblex.stats_vocabulary``).

Annotation only: neither element nor chunk text is ever mutated.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from womblex.store.evidence import EVIDENCE_FIELDS
from womblex.store.money_output import VALUE_TYPE
from womblex.store.output import _write_rows
from womblex.store.run_stamp import sidecar_footer

logger = logging.getLogger(__name__)

STAT_SPANS_SUFFIX = ".stat_spans.parquet"
VOCABULARY_KEY = b"womblex.stats_vocabulary"

STAT_SPANS_SCHEMA = pa.schema([
    ("source_hash", pa.string()),
    ("kind", pa.string()),             # period | measure | cue | figure
    ("locus", pa.string()),            # narrative | table_cell
    *EVIDENCE_FIELDS,                  # where the span is, and how precisely
    ("text", pa.string()),             # original — never lost
    ("measure_id", pa.string()),
    ("measure_label", pa.string()),
    ("value", VALUE_TYPE),             # figures only
    ("unit", pa.string()),             # ISO currency | percent | percentage_points | the measure's
    ("rate_per", pa.string()),         # week | year | … for `$1,242/week`
    ("multiplier", pa.string()),       # thousand | million | … — already applied to value
    ("period", pa.string()),           # canonical label: 2019-20 | 2025-Q1 | CY2025 | …
    ("period_kind", pa.string()),
    ("period_start", pa.date32()),     # null where the text does not fix the bounds
    ("period_end", pa.date32()),
    ("period_source", pa.string()),    # stated | sentence | column | row | heading
    ("cue_type", pa.string()),         # calculation | source | gap | forecast
    ("cue", pa.string()),              # canonical cue name
    ("calculation", pa.string()),      # a figure's calculation cues, `;`-joined
    ("source", pa.string()),           # a figure's source cue, as written
    ("status", pa.string()),           # figures: stated | forecast
    ("heading_path", pa.list_(pa.string())),
    ("heading_elem_orders", pa.list_(pa.int32())),
    ("context", pa.string()),
])


def stat_spans_path_for(base_path: Path) -> Path:
    return base_path.parent / f"{base_path.stem}{STAT_SPANS_SUFFIX}"


def write_stat_spans(
    rows: list[dict], output_path: Path, *, vocabulary_digest: str | None = None,
) -> Path:
    """Write a batch's stat spans (rows match :data:`STAT_SPANS_SCHEMA`)."""
    target = stat_spans_path_for(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    footer = dict(sidecar_footer(output_path, "stats") or {})
    if vocabulary_digest:
        footer[VOCABULARY_KEY] = vocabulary_digest.encode()
    _write_rows(rows, target, STAT_SPANS_SCHEMA, metadata=footer or None)
    logger.info("Wrote stat_spans shard %s: rows=%d", target.name, len(rows))
    return target


def read_stat_spans(path: Path) -> pa.Table:
    """Read stat spans from a single shard file or a shard-directory glob."""
    p = Path(path)
    if p.is_dir():
        shards = sorted(p.glob(f"*{STAT_SPANS_SUFFIX}"))
        if not shards:
            return pa.table(
                {f.name: pa.array([], type=f.type) for f in STAT_SPANS_SCHEMA},
                schema=STAT_SPANS_SCHEMA,
            )
        return pa.concat_tables([_read_shard(s) for s in shards])
    return _read_shard(p if p.name.endswith(STAT_SPANS_SUFFIX) else stat_spans_path_for(p))


def _read_shard(path: Path) -> pa.Table:
    raw = pq.read_table(str(path))
    missing = [f.name for f in STAT_SPANS_SCHEMA if f.name not in raw.schema.names]
    if missing:
        raise ValueError(
            f"stat_spans shard {path} missing columns {missing}; schema bump without compat shim?"
        )
    return raw.select([f.name for f in STAT_SPANS_SCHEMA]).cast(STAT_SPANS_SCHEMA)


__all__ = [
    "STAT_SPANS_SCHEMA",
    "STAT_SPANS_SUFFIX",
    "VOCABULARY_KEY",
    "read_stat_spans",
    "stat_spans_path_for",
    "write_stat_spans",
]
