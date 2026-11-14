"""Reference-period grammar for the ``stats`` annotation op (pure core).

Reads the ways Australian statistical publications write the period a figure
refers to, and resolves each to a canonical label and, where the text fixes
them, calendar bounds:

==================  ======================================  ===========
``kind``            written as                              ``label``
==================  ======================================  ===========
``financial_year``  ``2019–20``, ``2019/20``, ``FY20``,     ``2019-20``
                    ``FY2019-20``, ``2019-20 financial
                    year``
``quarter``         ``2025-Q1``, ``Q1 2025``, ``March       ``2025-Q1``
                    quarter 2025``, ``March quarter``
``half``            ``2026-H1``, ``first half of 2026``     ``2026-H1``
``calendar_year``   ``CY2025``, ``calendar year 2025``      ``CY2025``
``year``            ``in 2021``, ``2021 Census``            ``2021``
``month``           ``March 2025``, ``2025-03``             ``2025-03``
``date``            ``30 June 2021``                        ``2021-06-30``
``year_range``      ``2019–21``, ``2019–2021``              ``2019-21``
==================  ======================================  ===========

A financial year is the Australian one (1 July to 30 June), and ``FY20`` names
the year it ends in. A quarter is a calendar quarter, named by its last month as
the ABS does. Bounds are left null where the text does not fix them: a quarter
written without its year, and a ``year_range``, which may be calendar or
financial years.

Offsets index the string handed in. A bare four-digit number is a year only
beside a temporal cue, or anywhere when ``bare_years`` is set (headings and
column headers, where a lone ``2021`` is the period).
"""

from __future__ import annotations

import re
from calendar import monthrange
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

_MONTHS: dict[str, int] = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
_MONTH = "(?P<month>" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?"
_YEAR = r"(?:19|20)\d{2}"
# Hyphen, non-breaking hyphen, figure dash, en dash, em dash, minus, slash.
_DASH = "[-‑‒–—−/]"
# Whitespace that stays on one line: a period never spans two elements.
_WS = r"[^\S\n]+"
_OWS = r"[^\S\n]*"
# A year sitting inside a longer number, code or date is not a period.
_NOT_BEFORE = r"(?<![\w.,$\-–/])"
_NOT_AFTER = r"(?![\d\-–/]|[.,]\d)"
# Beside a month name the year is not in doubt, so a footnote marker run on to it
# (`31 March 2020.56`) is left where it is.
_YEAR_END = r"(?!\d)"
_ORDINAL_HALF = {"first": 1, "1st": 1, "second": 2, "2nd": 2}


_Resolved = tuple[str, str, date | None, date | None]


@dataclass(slots=True)
class Period:
    """One reference period. ``text`` is the original; ``label`` the canonical form."""

    text: str
    start: int
    end: int
    kind: str
    label: str
    period_start: date | None = None
    period_end: date | None = None


def _span(year: int, first_month: int, months: int) -> tuple[date, date]:
    last_month = first_month + months - 1
    end_year, end_month = year + (last_month - 1) // 12, (last_month - 1) % 12 + 1
    return date(year, first_month, 1), date(end_year, end_month, monthrange(end_year, end_month)[1])


def _financial_year(start_year: int) -> tuple[str, date, date]:
    return (f"{start_year}-{(start_year + 1) % 100:02d}", *_span(start_year, 7, 12))


def _fy_pair(first: str, second: str, *, cued: bool) -> _Resolved | None:
    """``(kind, label, start, end)`` for ``first<dash>second``; ``None`` when it is no period."""
    a = int(first)
    b = int(second) if len(second) == 4 else a - a % 100 + int(second)
    if len(second) == 2 and a % 100 == 99:
        b += 100  # 1999-00
    if b == a + 1 and (len(second) == 2 or cued):
        return ("financial_year", *_financial_year(a))
    if b > a:
        return "year_range", f"{a}-{b % 100:02d}" if len(second) == 2 else f"{a}-{b}", None, None
    return None


def _quarter(year: int | None, q: int) -> _Resolved:
    if year is None:
        return "quarter", f"Q{q}", None, None
    return ("quarter", f"{year}-Q{q}", *_span(year, 3 * q - 2, 3))


def _read_fy(m: re.Match[str]) -> _Resolved | None:
    return _fy_pair(m["a"], m["b"], cued=bool(m.groupdict().get("cue") or m.groupdict().get("tail")))


def _read_fy_short(m: re.Match[str]) -> _Resolved | None:
    raw = m["y"]
    # Two digits pivot as POSIX `%y` does: FY99 ends in 1999, FY20 in 2020.
    end = int(raw) if len(raw) == 4 else (1900 if int(raw) >= 69 else 2000) + int(raw)
    return ("financial_year", *_financial_year(end - 1))


def _read_quarter_iso(m: re.Match[str]) -> _Resolved | None:
    return _quarter(int(m["y"]), int(m["q"]))


def _read_quarter_named(m: re.Match[str]) -> _Resolved | None:
    month = _MONTHS[m["month"].lower()]
    if month % 3:  # the ABS names a quarter by its last month
        return None
    return _quarter(int(m["y"]) if m["y"] else None, month // 3)


def _read_half(m: re.Match[str]) -> _Resolved | None:
    h = int(m["h"]) if m["h"].isdigit() else _ORDINAL_HALF[m["h"].lower()]
    return ("half", f"{m['y']}-H{h}", *_span(int(m["y"]), 6 * h - 5, 6))


def _read_calendar_year(m: re.Match[str]) -> _Resolved | None:
    return ("calendar_year", f"CY{m['y']}", *_span(int(m["y"]), 1, 12))


def _read_year(m: re.Match[str]) -> _Resolved | None:
    return ("year", m["y"], *_span(int(m["y"]), 1, 12))


def _read_date(m: re.Match[str]) -> _Resolved | None:
    try:
        day = date(int(m["y"]), _MONTHS[m["month"].lower()], int(m["d"]))
    except ValueError:
        return None
    return "date", day.isoformat(), day, day


def _read_month(m: re.Match[str]) -> _Resolved | None:
    month = _MONTHS[m["month"].lower()]
    return ("month", f"{m['y']}-{month:02d}", *_span(int(m["y"]), month, 1))


def _read_month_iso(m: re.Match[str]) -> _Resolved | None:
    year, month = int(m["y"]), int(m["mm"])
    if not 1 <= month <= 12 or month == (year + 1) % 100:  # 2011-12 is a financial year
        return None
    return ("month", f"{year}-{month:02d}", *_span(year, month, 1))


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


# In priority order: an earlier pattern keeps a span a later one also matches.
_Reader = Callable[[re.Match[str]], _Resolved | None]
_PATTERNS: tuple[tuple[re.Pattern[str], _Reader, str | None], ...] = (
    (_rx(rf"{_NOT_BEFORE}(?P<y>{_YEAR})(?:-|[^\S\n])?Q(?P<q>[1-4])\b"), _read_quarter_iso, None),
    (_rx(rf"\bQ(?P<q>[1-4])(?:-|[^\S\n])?(?P<y>{_YEAR}){_NOT_AFTER}"), _read_quarter_iso, None),
    (_rx(rf"\b{_MONTH}{_WS}(?P<y>{_YEAR}){_WS}(?:quarter|qtr)\b"), _read_quarter_named, None),
    (_rx(rf"\b{_MONTH}{_WS}(?:quarter|qtr)\b(?:{_WS}(?:of{_WS})?(?P<y>{_YEAR}){_YEAR_END})?"),
     _read_quarter_named, None),
    (_rx(rf"{_NOT_BEFORE}(?P<y>{_YEAR})(?:-|[^\S\n])?H(?P<h>[12])\b"), _read_half, None),
    (_rx(rf"\bH(?P<h>[12])(?:-|[^\S\n])?(?P<y>{_YEAR}){_NOT_AFTER}"), _read_half, None),
    (_rx(rf"\b(?P<h>first|second|1st|2nd){_WS}half{_WS}of{_WS}(?P<y>{_YEAR}){_NOT_AFTER}"),
     _read_half, None),
    (_rx(rf"\b(?P<cue>FY|(?:financial|fiscal){_WS}year){_OWS}(?P<a>{_YEAR}){_OWS}{_DASH}{_OWS}"
         rf"(?P<b>{_YEAR}|\d{{2}}){_NOT_AFTER}"), _read_fy, None),
    (_rx(rf"{_NOT_BEFORE}(?P<a>{_YEAR}){_OWS}{_DASH}{_OWS}(?P<b>{_YEAR}|\d{{2}}){_NOT_AFTER}"
         rf"(?P<tail>{_WS}(?:financial|fiscal){_WS}year\b)?"), _read_fy, None),
    (_rx(rf"\b(?:FY{_OWS}|(?:financial|fiscal){_WS}year{_WS})(?P<y>{_YEAR}|\d{{2}}){_NOT_AFTER}"),
     _read_fy_short, None),
    (_rx(rf"\b(?:CY{_OWS}|calendar{_WS}year{_WS})(?P<y>{_YEAR}){_NOT_AFTER}"),
     _read_calendar_year, None),
    (_rx(rf"\b(?P<d>\d{{1,2}}){_WS}{_MONTH}{_WS}(?P<y>{_YEAR}){_YEAR_END}"), _read_date, None),
    (_rx(rf"\b{_MONTH}{_WS}(?P<y>{_YEAR}){_YEAR_END}"), _read_month, None),
    (_rx(rf"{_NOT_BEFORE}(?P<y>{_YEAR})-(?P<mm>\d{{2}}){_NOT_AFTER}"), _read_month_iso, None),
    # The cue is context, not part of the period: the span is the year alone.
    (_rx(rf"\b(?:in|during|since|until|by|year|census){_WS}(?P<y>{_YEAR}){_NOT_AFTER}"),
     _read_year, "y"),
    (_rx(rf"{_NOT_BEFORE}(?P<y>{_YEAR})(?={_WS}census\b)"), _read_year, "y"),
)
_BARE_YEAR = _rx(rf"{_NOT_BEFORE}(?P<y>{_YEAR})\b{_NOT_AFTER}(?!{_OWS}(?:%|per{_WS}cent))")


def find_periods(text: str, *, bare_years: bool = False) -> list[Period]:
    """Every reference period in *text*, in order, none overlapping."""
    found: list[Period] = []
    taken: list[tuple[int, int]] = []
    patterns = (*_PATTERNS, (_BARE_YEAR, _read_year, "y")) if bare_years else _PATTERNS
    for pattern, read, group in patterns:
        for m in pattern.finditer(text):
            start, end = m.span(group) if group else m.span()
            if any(start < e and s < end for s, e in taken):
                continue
            # `may` is a verb far more often than a month: only the capitalised form counts.
            if "month" in m.groupdict() and m["month"] == "may":
                continue
            resolved = read(m)
            if resolved is None:
                continue
            kind, label, period_start, period_end = resolved
            taken.append((start, end))
            found.append(Period(text[start:end], start, end, kind, label, period_start, period_end))
    return sorted(found, key=lambda p: p.start)


__all__ = ["Period", "find_periods"]
