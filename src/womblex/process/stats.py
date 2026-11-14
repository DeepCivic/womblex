"""Statistical figure recognition (pure core of the ``stats`` annotation op).

Over one block of text, finds four kinds of span:

- ``period`` — a reference period (:mod:`womblex.process.stats_periods`);
- ``measure`` — a phrase the corpus vocabulary names (:mod:`womblex.process.stats_vocab`);
- ``cue`` — a calculation, source, gap or forecast cue;
- ``figure`` — a number stated in the same sentence as a measure.

**A number is a figure only beside a measure.** A sentence that names no measure
yields no figure, however many numbers it holds: the vocabulary is the positive
evidence, as a currency marker is for the money op. How close "beside" is
depends on the number. One that states its own unit (``$1,242``, ``4.5%``) is a
quantity on its face and takes the nearest measure in its sentence. A unit-less
one (``43,516``) could as easily be a footnote marker, a paragraph number or a
page number, so it must sit next to the measure: ``the population was 43,516``,
``24 million hectares``. Each figure carries what its
sentence says about it — the nearest measure, the nearest period (else the one
its headings state, handed in as ``inherited``), the calculation and source
cues, and ``forecast`` in place of ``stated`` when a forecast cue is present.

A bare four-digit number that reads as a year (``2021``) and carries no unit,
separator or scale is neither a figure nor a period in running text: it could be
either, and a wrong value is worse than a missing one.

Nothing here reads or writes parquet; offsets index the string handed in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from itertools import pairwise

from womblex.process.money import MoneyOptions, find_money
from womblex.process.money_numbers import apply_scale, blocked_spans, parse_number
from womblex.process.money_vocab import NUM_AU
from womblex.process.stats_periods import Period, find_periods
from womblex.process.stats_vocab import Found, Measure, Vocabulary

_OWS = r"[^\S\n]*"
_NEG = "[-−]"
_SCALE = r"(?P<scale>thousand|million|billion|trillion|mn|bn)"
_PERCENT_RE = re.compile(
    rf"(?<![\w.,$/])(?P<neg>{_NEG})?(?P<num>{NUM_AU}){_OWS}"
    rf"(?P<unit>%|per{_OWS}cent\b|percentage{_OWS}points?\b)",
    re.IGNORECASE,
)
# What follows a number that is a day of the month or a contents-page number.
_NOT_A_COUNT = (
    r"(?![^\S\n]+(?:(?-i:Figure|Table|Chapter|Appendix|Box)\b|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b))"
)
# A sign is read only where it opens the number: `10-20` is a range, not -20. A
# number run on to a colon, quote or bracket is a footnote marker.
_NUMBER_RE = re.compile(
    rf"(?<![\w.,$/:\-–−’”\"')\]])(?P<neg>{_NEG}(?=[\d.]))?(?P<num>{NUM_AU})"
    rf"(?:{_OWS}{_SCALE}\b)?(?![\w–]|[.,/\-]\d){_NOT_A_COUNT}",
    re.IGNORECASE,
)
_RATE_RE = re.compile(
    rf"{_OWS}(?:/|per{_OWS}|a{_OWS}|each{_OWS})"
    r"(?P<per>hour|day|week|fortnight|month|quarter|year|annum)\b",
    re.IGNORECASE,
)
_BARE_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
# A number that opens a line and is followed by a capitalised word numbers the
# paragraph (`4.21 We also heard`); it does not count anything.
_ENUMERATOR_RE = re.compile(r"(?:(?<=\n)|\A)[\d.]+[^\S\n]+[A-Z]")
_YEAR_IN_GAP_RE = re.compile(r"(?<![\w.,])(?:19|20)\d{2}(?![\w]|[.,]\d)")
# Between a measure and its unit-less number: a few words, and nothing that ends
# a clause or is itself a number.
_GAP_BREAK_RE = re.compile(r"[\d.,;!?()\n]")
_GAP_AFTER_MEASURE, _GAP_BEFORE_MEASURE = 48, 24
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])[^\S\n]+(?=[A-Z(\"$‘“])|\n+")
# Money's false-positive classes that are not false here: the stats grammar reads
# periods and percentages itself, and a measurement (`271,000 hectares`) is a figure.
_OWN_CLASSES = frozenset({
    "percent", "financial_year", "date_worded", "date_worded_leading",
    "measurement", "temperature",
})


@dataclass(slots=True)
class StatSpan:
    """One span. ``text`` is the original; fields not relevant to ``kind`` are null."""

    kind: str                       # figure | measure | period | cue
    text: str
    start: int
    end: int
    measure: Measure | None = None
    value: Decimal | None = None
    unit: str | None = None         # ISO currency | percent | percentage_points | the measure's
    rate_per: str | None = None     # week | year | … for `$1,242/week`
    multiplier: str | None = None
    period: Period | None = None
    period_source: str | None = None  # stated (a period span) | sentence | heading
    cue_type: str | None = None
    cue: str | None = None
    calculation: str | None = None  # the sentence's calculation cues, `;`-joined
    source: str | None = None       # the sentence's source cue, as written
    status: str | None = None       # stated | forecast


@dataclass(slots=True)
class _Number:
    start: int
    end: int
    value: Decimal
    unit: str | None
    multiplier: str | None = None
    plain: bool = False             # states no unit of its own


def _overlaps(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(start < e and s < end for s, e in spans)


def _signed(value: Decimal, neg: str | None) -> Decimal:
    return -value if neg else value


def _numbers(text: str, taken: list[tuple[int, int]], default_currency: str) -> list[_Number]:
    """Every number in *text* that states a quantity, in order, none overlapping."""
    found: list[_Number] = []
    claimed = list(taken)

    def claim(n: _Number) -> None:
        if not _overlaps(n.start, n.end, claimed):
            claimed.append((n.start, n.end))
            found.append(n)

    for span in find_money(text, MoneyOptions(default_currency=default_currency)):
        claim(_Number(span.start, span.end, span.value, span.currency, span.multiplier))
    for m in _PERCENT_RE.finditer(text):
        value = parse_number(m["num"])
        if value is not None:
            unit = "percentage_points" if m["unit"].lower().startswith("percentage") else "percent"
            claim(_Number(m.start(), m.end(), _signed(value, m["neg"]), unit))
    for m in _NUMBER_RE.finditer(text):
        value = parse_number(m["num"])
        if value is None or (not m["scale"] and _BARE_YEAR_RE.fullmatch(m["num"])):
            continue
        if m["num"].startswith(".") or _ENUMERATOR_RE.match(text, m.start()):
            continue  # a footnote marker after a full stop; a paragraph number
        value, multiplier = apply_scale(value, m["scale"])
        claim(_Number(m.start(), m.end(), _signed(value, m["neg"]), None, multiplier, plain=True))
    return sorted(found, key=lambda n: n.start)


def _sentence_bounds(text: str, keep_whole: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """``[start, end)`` per sentence; a boundary inside a matched phrase is not one."""
    cuts = [m.end() for m in _SENTENCE_END_RE.finditer(text)
            if not _overlaps(m.start(), m.end(), keep_whole)]
    edges = [0, *cuts, len(text)]
    return [(a, b) for a, b in pairwise(edges) if b > a]


def _gap(start: int, end: int, item: Found | Period) -> tuple[int, int]:
    """Distance from ``[start, end)`` to *item*; an item before it wins a tie."""
    if item.end <= start:
        return start - item.end, 0
    return max(item.start - end, 0), 1


def _nearest[T: (Found, Period)](start: int, end: int, items: list[T]) -> T | None:
    return min(items, key=lambda i: _gap(start, end, i), default=None)


def _adjacent(text: str, start: int, end: int, measure: Found, periods: list[Period]) -> bool:
    """Whether the unit-less number at ``[start, end)`` sits next to *measure*.

    A period between the two is transparent (``the population in 2021 was
    43,516``), as is a bare year; any other number, or clause punctuation, is not.
    """
    if measure.end <= start:
        lo, hi, limit = measure.end, start, _GAP_AFTER_MEASURE
    elif end <= measure.start:
        lo, hi, limit = end, measure.start, _GAP_BEFORE_MEASURE
    else:
        return False
    gap = text[lo:hi]
    for p in periods:
        if lo <= p.start and p.end <= hi:
            gap = gap[:p.start - lo] + " " * (p.end - p.start) + gap[p.end - lo:]
    gap = _YEAR_IN_GAP_RE.sub(" ", gap)
    if len(gap.strip()) > limit or _GAP_BREAK_RE.search(gap):
        return False
    return ":" not in gap or measure.end <= start


def _within[T: (Found, Period)](bounds: tuple[int, int], items: list[T]) -> list[T]:
    return [i for i in items if bounds[0] <= i.start < bounds[1]]


def find_stats(
    text: str,
    vocab: Vocabulary,
    *,
    inherited: Period | None = None,
    bare_years: bool = False,
    default_currency: str = "AUD",
) -> list[StatSpan]:
    """Every period, measure, cue and figure in *text*, sorted by position.

    ``inherited`` is the period the text's headings state, given to a figure
    whose own sentence names none. ``bare_years`` reads a lone ``2021`` as a
    period, for a heading.
    """
    if not text:
        return []
    periods = find_periods(text, bare_years=bare_years)
    measures = vocab.find_measures(text)
    cues = vocab.find_cues(text)
    phrases = [(p.start, p.end) for p in periods] + [(f.start, f.end) for f in (*measures, *cues)]

    out: list[StatSpan] = [
        StatSpan("period", p.text, p.start, p.end, period=p, period_source="stated")
        for p in periods
    ]
    out += [
        StatSpan("measure", text[m.start:m.end], m.start, m.end, measure=vocab.measures[m.kind])
        for m in measures
    ]

    blocked = [(s, e) for s, e, name in blocked_spans(text) if name not in _OWN_CLASSES]
    for bounds in _sentence_bounds(text, phrases):
        here_measures = _within(bounds, measures)
        here_cues = _within(bounds, cues)
        here_periods = _within(bounds, periods)
        for cue in here_cues:
            near = _nearest(cue.start, cue.end, here_measures)
            out.append(StatSpan(
                "cue", text[cue.start:cue.end], cue.start, cue.end,
                measure=vocab.measures[near.kind] if near else None,
                cue_type=cue.cue_type, cue=cue.kind,
            ))
        if not here_measures:
            continue
        calculation = ";".join(sorted({c.kind for c in here_cues if c.cue_type == "calculation"}))
        source = next((text[c.start:c.end] for c in here_cues if c.cue_type == "source"), None)
        status = "forecast" if any(c.cue_type == "forecast" for c in here_cues) else "stated"
        sentence = text[bounds[0]:bounds[1]]
        taken = [(s - bounds[0], e - bounds[0]) for s, e in (*phrases, *blocked)
                 if s < bounds[1] and bounds[0] < e]
        for n in _numbers(sentence, taken, default_currency):
            start, end = n.start + bounds[0], n.end + bounds[0]
            candidates = [
                m for m in here_measures
                if not n.plain or _adjacent(text, start, end, m, here_periods)
            ]
            if not candidates:
                continue
            measure = vocab.measures[min(candidates, key=lambda m: _gap(start, end, m)).kind]
            period = _nearest(start, end, here_periods)
            rate = _RATE_RE.match(text, end)
            out.append(StatSpan(
                "figure", text[start:end], start, end, measure=measure,
                value=n.value, unit=n.unit or measure.unit, multiplier=n.multiplier,
                rate_per=_rate(rate), period=period or inherited,
                period_source="sentence" if period else ("heading" if inherited else None),
                calculation=calculation or None, source=source, status=status,
            ))
    return sorted(out, key=lambda s: (s.start, s.end))


def _rate(m: re.Match[str] | None) -> str | None:
    if m is None:
        return None
    per = m["per"].lower()
    return "year" if per == "annum" else per


def read_cell_figure(text: str, default_currency: str = "AUD") -> StatSpan | None:
    """A table cell that is one number, as a figure; ``None`` for any other cell.

    In a row whose label names a measure the cell *is* the figure, so a plain
    ``2021`` here is a value: the column header, not the cell, states the period.
    """
    stripped = text.strip()
    if not stripped or "\n" in stripped:
        return None
    offset = text.index(stripped)
    numbers = _numbers(stripped, [], default_currency)
    if not numbers and (m := _NUMBER_RE.fullmatch(stripped)):
        value = parse_number(m["num"])
        numbers = [_Number(0, len(stripped), _signed(value, m["neg"]), None)] if value is not None else []
    if len(numbers) != 1:
        return None
    n = numbers[0]
    rate = _RATE_RE.fullmatch(stripped, n.end)
    if (n.start, n.end) != (0, len(stripped)) and not (n.start == 0 and rate):
        return None
    return StatSpan(
        "figure", stripped[n.start:n.end], offset + n.start, offset + n.end,
        value=n.value, unit=n.unit, multiplier=n.multiplier, rate_per=_rate(rate), status="stated",
    )


__all__ = ["StatSpan", "find_stats", "read_cell_figure"]
