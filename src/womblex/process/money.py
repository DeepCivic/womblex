"""Self-evidencing monetary amount recognition (pure core).

Recovers amounts that carry their own evidence — a currency symbol, an ISO
4217 code or a currency word sitting with the number — from a text string, and
normalises them to exact ``Decimal`` values. The column-evidenced path (a bare
number whose money-ness comes from its column) lives in
:mod:`womblex.process.money_columns`; the per-stage driver that applies both
over a shard directory is :mod:`womblex.process.money_stage`.

Nothing here reads or writes parquet, and nothing rewrites text: offsets index
the string handed in, so callers own the coordinate space.

**Every extraction must have positive evidence that the number represents
money** (``docs/money.md``). A bare number with no marker is not an extraction
— except under ``implicit_context``, which is off by default because it is
measurably low precision on this corpus.

Patterns are applied in the priority order of ``docs/money.md``; overlap is
resolved by priority, then span length, then confidence.
"""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from decimal import Decimal

from womblex.process.money_vocab import (
    _SCALE_ALT,
    _SUFFIX_SYMBOL_ALT,
    _SYMBOL_ALT,
    _WORD_ALT,
    ACCOUNTING_RE,
    AMBIGUOUS_SCALES,
    CODE_ALIASES,
    CONTEXT_RE,
    CURRENCY_WORDS,
    FALSE_POSITIVE_PATTERNS,
    ISO_4217,
    MODIFIER_RE,
    NUM_AU,
    NUM_INTL,
    POSTCODE_RE,
    SCALES,
    STATE_RE,
    SUBUNIT_WORDS,
    SYMBOL_TO_CODE,
    currency_tier,
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MoneySpan:
    """One extracted amount. ``text`` is the original — never lost."""

    text: str
    start: int
    end: int
    value: Decimal
    currency: str | None
    currency_source: str
    evidence: str
    confidence: float
    negative: bool = False
    multiplier: str | None = None
    modifier: str | None = None
    range_group: int | None = None
    range_role: str | None = None


@dataclass(slots=True)
class MoneyOptions:
    """Detector knobs. Mirrors the narrative half of ``MoneyConfig``."""

    default_currency: str = "AUD"
    international_numbers: bool = False
    implicit_context: bool = False
    min_confidence: float = 0.5
    context_chars: int = 160


# Priority drives overlap resolution (lower wins). Ranges are resolved first
# and claim their whole span, so a range endpoint is never re-claimed.
# Accounting negatives outrank the symbol patterns because they *enclose* one:
# `($100)` must resolve to -100, not to the `$100` sitting inside it.
_PRIORITY = {"p7": 0, "p9": 0, "p1": 1, "p6": 1, "p2": 2, "p4": 4, "p3": 5,
             "p5": 6, "p10": 9}

_CONFIDENCE = {"p1": 0.99, "p6": 0.99, "p2": 0.99, "p3": 0.90, "p4": 0.90,
               "p5": 0.75, "p9": 0.90, "p10": 0.35}

# Whitespace inside a pattern may span at most one line break, so no match can
# cross the ``\n\n`` element join `reassemble_narrative` writes — a pattern that
# did would bind two unrelated paragraphs together. Same discipline as the PII
# regexes (CLAUDE.md). Range separators are stricter still (`_HGAP`): a range
# fabricates a *relationship* between two numbers, so it stays on one line.
_GAP = r"[^\S\n]*\n?[^\S\n]*"
_HGAP = r"[^\S\n]*"

_SCALE_TAIL = rf"(?:{_GAP}(?P<scale>{_SCALE_ALT})(?![A-Za-z]))?"
_NEG = rf"(?P<neg>-{_HGAP})?"


def _num_alt(international: bool) -> str:
    return f"(?:{NUM_INTL}|{NUM_AU})" if international else f"(?:{NUM_AU})"


def _compile(international: bool) -> dict[str, re.Pattern[str]]:
    """Build the pattern set for a locale mode (cached by :func:`_patterns`)."""
    num = _num_alt(international)
    scale_tail = rf"(?:{_GAP}(?P<scale_a>{_SCALE_ALT})(?![A-Za-z]))?"
    left = rf"(?:(?P<sym_a>{_SYMBOL_ALT}){_HGAP})?(?P<neg_a>-{_HGAP})?(?P<num_a>{num})" \
           rf"{scale_tail}"
    right = rf"(?:(?P<sym_b>{_SYMBOL_ALT}){_HGAP})?(?P<num_b>{num})" \
            rf"(?:{_HGAP}(?P<scale_b>{_SCALE_ALT})(?![A-Za-z]))?"
    return {
        # 7 — ranges. Scanned first so an endpoint is never claimed alone.
        "p7": re.compile(
            rf"(?<![A-Za-z0-9]){left}{_HGAP}(?:[\u2013\u2014\u2012-]|to){_HGAP}{right}",
            re.IGNORECASE),
        "p7_between": re.compile(
            rf"\bbetween{_HGAP}\s?{left}{_HGAP}\s?and{_HGAP}\s?{right}", re.IGNORECASE),
        # 1 / 6 — symbol prefix, optional magnitude.
        "p1": re.compile(
            rf"(?<![A-Za-z0-9]){_NEG}(?P<sym>{_SYMBOL_ALT}){_HGAP}(?P<num>{num})"
            rf"{_SCALE_TAIL}", re.IGNORECASE),
        # 2 — ISO prefix (`AUD 100`, `AUD$21.9 million`). Case-sensitive.
        "p2": re.compile(
            rf"(?<![A-Za-z])(?P<iso>[A-Z]{{3}}){_HGAP}(?P<sym>\$)?{_HGAP}{_NEG}"
            rf"(?P<num>{num}){_SCALE_TAIL}"),
        # 3 — ISO suffix (`100 AUD`).
        "p3": re.compile(
            rf"(?<![A-Za-z0-9]){_NEG}(?P<num>{num})"
            rf"{_SCALE_TAIL}{_HGAP}"
            rf"(?P<iso>[A-Z]{{3}})(?![A-Za-z])"),
        # 4 — currency word (`100 dollars`, `250 Australian dollars`).
        "p4": re.compile(
            rf"(?<![A-Za-z0-9]){_NEG}(?P<num>{num}){_SCALE_TAIL}{_GAP}"
            rf"(?P<word>{_WORD_ALT})\b", re.IGNORECASE),
        # 5 — symbol suffix (`100$`, `50€`).
        "p5": re.compile(
            rf"(?<![A-Za-z0-9]){_NEG}(?P<num>{num}){_HGAP}"
            rf"(?P<sym>{_SUFFIX_SYMBOL_ALT})(?![0-9])"),
        # 9 — accounting negative (`($100)`, `(6,550.1)` under context).
        "p9": re.compile(
            rf"\({_HGAP}(?P<sym>{_SYMBOL_ALT})?{_HGAP}(?P<num>{num})"
            rf"{_SCALE_TAIL}{_HGAP}\)", re.IGNORECASE),
        # 10 — bare number, implicit financial context. Off by default.
        "p10": re.compile(rf"(?<![A-Za-z0-9.,]){_NEG}(?P<num>{num})"
                          rf"{_SCALE_TAIL}(?![A-Za-z0-9])", re.IGNORECASE),
    }


_PATTERN_CACHE: dict[bool, dict[str, re.Pattern[str]]] = {}


def _patterns(international: bool) -> dict[str, re.Pattern[str]]:
    if international not in _PATTERN_CACHE:
        _PATTERN_CACHE[international] = _compile(international)
    return _PATTERN_CACHE[international]


# ---------------------------------------------------------------------------
# Number / currency normalisation
# ---------------------------------------------------------------------------


def parse_number(raw: str, *, international: bool = False) -> Decimal | None:
    """Parse an Australian (or, opt-in, international) formatted number."""
    s = raw.strip()
    if not s:
        return None
    if international and ("." in s and "," in s and s.rindex(",") > s.rindex(".")):
        s = s.replace(".", "").replace(",", ".")
    elif international and "," in s and "." not in s and re.fullmatch(r"\d+,\d+", s):
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        return Decimal(s)
    except Exception:
        return None


def apply_scale(value: Decimal, scale: str | None) -> tuple[Decimal, str | None]:
    """Multiply by a magnitude suffix. Returns ``(value, canonical_suffix)``."""
    if not scale:
        return value, None
    key = scale.lower()
    factor = SCALES.get(key)
    if factor is None:
        return value, None
    return value * factor, key


def resolve_symbol(sym: str) -> str:
    return SYMBOL_TO_CODE.get(sym, SYMBOL_TO_CODE.get(sym.upper(), "AUD"))


def resolve_iso(code: str) -> str | None:
    """Return the ISO code if *code* is a real currency code, else ``None``."""
    upper = code.upper()
    if upper in CODE_ALIASES:
        return CODE_ALIASES[upper]
    return upper if upper in ISO_4217 else None


# ---------------------------------------------------------------------------
# False-positive blocking
# ---------------------------------------------------------------------------


def blocked_spans(text: str) -> list[tuple[int, int, str]]:
    """Spans covering Australian false-positive classes (dates, ABNs, …).

    A candidate overlapping one of these is discarded. Measurement and
    percentage matches are skipped where a currency marker immediately
    precedes the number, so ``$100 m`` stays a magnitude expression while
    ``100m road`` does not.
    """
    out: list[tuple[int, int, str]] = []
    for name, pattern in FALSE_POSITIVE_PATTERNS.items():
        for m in pattern.finditer(text):
            if name in ("measurement", "temperature") and _preceded_by_currency(text, m.start()):
                continue
            if name == "incident_ref" and _iso_prefixed(m.group(0)):
                continue  # `USD100` is an amount, not a reference number
            out.append((m.start(), m.end(), name))
    for m in POSTCODE_RE.finditer(text):
        window = text[max(0, m.start() - 40):m.end() + 40]
        if STATE_RE.search(window):
            out.append((m.start(), m.end(), "postcode"))
    return out


def _iso_prefixed(token: str) -> bool:
    """True for `USD100`-shaped tokens whose letters are a real currency code."""
    m = re.match(r"([A-Z]{2,4})", token)
    return m is not None and resolve_iso(m.group(1)) is not None


def _preceded_by_currency(text: str, pos: int) -> bool:
    """True when a currency symbol or ISO code sits just before *pos*."""
    prefix = text[max(0, pos - 8):pos]
    stripped = prefix.rstrip()
    if not stripped:
        return False
    if any(stripped.endswith(sym) for sym in SYMBOL_TO_CODE):
        return True
    tail = stripped[-3:]
    return bool(re.fullmatch(r"[A-Z]{3}", tail)) and resolve_iso(tail) is not None


class _IntervalIndex:
    """Sorted interval set answering "does anything overlap [start, end)?".

    A linear scan per candidate is quadratic in document length — measured at
    3s on a 300 KB narrative, which is an ordinary FOI bundle. Sorting by start
    and carrying a running maximum end answers each query with one bisect: an
    overlap exists iff some interval starting before ``end`` reaches past
    ``start``, and the prefix maximum is exactly that reach.
    """

    __slots__ = ("_starts", "_max_end")

    def __init__(self, spans: list[tuple[int, int]]) -> None:
        items = sorted(spans)
        self._starts = [s for s, _ in items]
        self._max_end: list[int] = []
        reach = -1
        for _, end in items:
            reach = max(reach, end)
            self._max_end.append(reach)

    def overlaps(self, start: int, end: int) -> bool:
        i = bisect_left(self._starts, end)
        return i > 0 and self._max_end[i - 1] > start


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


_MALFORMED_GROUP_RE = re.compile(r",\d{1,2}(?!\d)")


def _ambiguous_continuation(m: re.Match[str], num_group: str, international: bool) -> bool:
    """True when the number runs on in a way this locale can't account for.

    Two cases, both silently wrong if extracted rather than declined:

    - ``€1.000,50`` in Australian mode reads as ``€1.000`` — one euro, wrong by
      10³. ``international_numbers`` is the deliberate opt-in for that format.
    - ``$1,23`` has a malformed thousands group; the pattern consumes ``$1`` and
      drops the rest, reporting one dollar instead of a hundred-odd.
    """
    tail = m.string[m.end(num_group):m.end(num_group) + 4]
    if _MALFORMED_GROUP_RE.match(tail):
        return True
    if international:
        return False
    raw = m.group(num_group)
    return bool(re.fullmatch(r"\d{1,3}\.\d{3}", raw)) and bool(re.match(r",\d", tail))


def _candidate(
    m: re.Match[str], evidence: str, *,
    currency: str | None, currency_source: str,
    num_group: str = "num", scale_group: str = "scale",
    confidence: float | None = None, negative: bool = False,
    international: bool = False, subunit: bool = False,
) -> MoneySpan | None:
    if _ambiguous_continuation(m, num_group, international):
        return None
    value = parse_number(m.group(num_group), international=international)
    if value is None:
        return None
    scale_raw = m.groupdict().get(scale_group)
    value, multiplier = apply_scale(value, scale_raw)
    if subunit:
        value = value / 100
        multiplier = multiplier or "cents"
    neg = negative or bool(m.groupdict().get("neg"))
    if neg:
        value = -value
    conf = _CONFIDENCE[evidence] if confidence is None else confidence
    if currency is not None and currency_tier(currency) == 3:
        conf = max(0.1, conf - 0.10)
    return MoneySpan(
        text=m.group(0).strip(),
        start=m.start(),
        end=m.end(),
        value=value,
        currency=currency,
        currency_source=currency_source,
        evidence=evidence,
        confidence=round(conf, 4),
        negative=neg,
        multiplier=multiplier,
    )


def _scan_symbol_prefix(text: str, pats, opts: MoneyOptions) -> list[MoneySpan]:
    out = []
    for m in pats["p1"].finditer(text):
        evidence = "p6" if m.group("scale") else "p1"
        span = _candidate(
            m, evidence, currency=resolve_symbol(m.group("sym")),
            currency_source="symbol", international=opts.international_numbers,
        )
        if span:
            out.append(span)
    return out


def _scan_iso_prefix(text: str, pats, opts: MoneyOptions) -> list[MoneySpan]:
    out = []
    for m in pats["p2"].finditer(text):
        code = resolve_iso(m.group("iso"))
        if code is None:
            continue
        evidence = "p6" if m.group("scale") else "p2"
        span = _candidate(
            m, evidence, currency=code, currency_source="iso",
            international=opts.international_numbers,
        )
        if span:
            out.append(span)
    return out


def _scan_iso_suffix(text: str, pats, opts: MoneyOptions) -> list[MoneySpan]:
    out = []
    for m in pats["p3"].finditer(text):
        code = resolve_iso(m.group("iso"))
        if code is None:
            continue
        span = _candidate(m, "p3", currency=code, currency_source="iso",
                          international=opts.international_numbers)
        if span:
            out.append(span)
    return out


def _scan_word(text: str, pats, opts: MoneyOptions) -> list[MoneySpan]:
    out = []
    for m in pats["p4"].finditer(text):
        word = m.group("word").lower()
        code = CURRENCY_WORDS.get(word, opts.default_currency)
        conf = _CONFIDENCE["p4"] if code is not None else 0.6
        span = _candidate(
            m, "p4", currency=code, currency_source="word",
            confidence=conf, international=opts.international_numbers,
            subunit=word in SUBUNIT_WORDS,
        )
        if span:
            out.append(span)
    return out


def _scan_symbol_suffix(text: str, pats, opts: MoneyOptions) -> list[MoneySpan]:
    out = []
    for m in pats["p5"].finditer(text):
        span = _candidate(m, "p5", currency=resolve_symbol(m.group("sym")),
                          currency_source="symbol",
                          international=opts.international_numbers)
        if span:
            out.append(span)
    return out


def _scan_accounting(text: str, pats, opts: MoneyOptions) -> list[MoneySpan]:
    """Bracketed negatives — gated, never scanned bare.

    Ungated this is the corpus's worst false-positive source (`s167(1)`,
    `(02) 6203 7300`, `(2018)`), so a bracketed number is an amount only when
    a currency marker sits inside or immediately before the brackets, or
    accounting context surrounds it *and* the number is formatted like a
    financial value (a decimal or a thousands group).
    """
    out = []
    for m in pats["p9"].finditer(text):
        sym = m.group("sym")
        currency = resolve_symbol(sym) if sym else None
        source = "symbol"
        confidence = _CONFIDENCE["p9"]
        if currency is None:
            before = text[max(0, m.start() - 6):m.start()].strip()
            code = resolve_iso(before[-3:]) if len(before) >= 3 else None
            if code:
                currency, source = code, "iso"
            else:
                raw = m.group("num")
                if "," not in raw and "." not in raw:
                    continue
                window = text[max(0, m.start() - 120):m.end() + 120]
                if not ACCOUNTING_RE.search(window):
                    continue
                currency, source, confidence = opts.default_currency, "document_default", 0.60
        span = _candidate(m, "p9", currency=currency, currency_source=source,
                          confidence=confidence, negative=True,
                          international=opts.international_numbers)
        if span:
            out.append(span)
    return out


def _scan_implicit(text: str, pats, opts: MoneyOptions) -> list[MoneySpan]:
    """Pattern 10 — bare numbers near financial trigger vocabulary."""
    windows = [(m.start(), m.end() + 60) for m in CONTEXT_RE.finditer(text)]
    if not windows:
        return []
    out = []
    for m in pats["p10"].finditer(text):
        if not any(s <= m.start() < e for s, e in windows):
            continue
        scale = m.group("scale")
        if scale and scale.lower() in AMBIGUOUS_SCALES:
            continue  # no currency marker to license a bare `m` / `k` / `b`
        span = _candidate(
            m, "p10", currency=opts.default_currency,
            currency_source="document_default",
            international=opts.international_numbers,
        )
        if span:
            out.append(span)
    return out


# ---------------------------------------------------------------------------
# Ranges (pattern 7)
# ---------------------------------------------------------------------------


def _scan_ranges(
    text: str, pats, opts: MoneyOptions, blocked: _IntervalIndex,
) -> tuple[list[MoneySpan], list[tuple[int, int]]]:
    """Both endpoints, linked by ``range_group``; the whole span is claimed.

    Australian documents use ranges frequently and one endpoint often carries
    the evidence for both (``$10–20 million``): a missing symbol or magnitude
    on either side is inherited from the other rather than collapsing the
    range to a single value.
    """
    spans: list[MoneySpan] = []
    claimed: list[tuple[int, int]] = []
    group = 0
    for key in ("p7_between", "p7"):
        for m in pats[key].finditer(text):
            if blocked.overlaps(m.start(), m.end()):
                continue
            if any(m.start() < e and s < m.end() for s, e in claimed):
                continue  # claimed ranges are few; a linear check is cheap here
            sym_a, sym_b = m.group("sym_a"), m.group("sym_b")
            if not sym_a and not sym_b:
                continue  # no evidence on either endpoint
            if any(_ambiguous_continuation(m, g, opts.international_numbers)
                   for g in ("num_a", "num_b")):
                continue  # malformed / continental endpoint — decline the pair
            lo = parse_number(m.group("num_a"), international=opts.international_numbers)
            hi = parse_number(m.group("num_b"), international=opts.international_numbers)
            if lo is None or hi is None:
                continue
            scale = m.group("scale_a") or m.group("scale_b")
            lo, mult = apply_scale(lo, scale)
            hi, _ = apply_scale(hi, scale)
            currency = resolve_symbol(sym_a or sym_b)
            neg = bool(m.group("neg_a"))
            if neg:
                lo = -lo
            conf = 0.95 if (sym_a and sym_b) else 0.90
            group += 1
            for value, role, (s, e) in (
                (lo, "lower", (m.start("num_a"), m.end("num_a"))),
                (hi, "upper", (m.start("num_b"), m.end("num_b"))),
            ):
                spans.append(MoneySpan(
                    text=text[s:e], start=s, end=e, value=value,
                    currency=currency, currency_source="symbol", evidence="p7",
                    confidence=conf, negative=neg and role == "lower",
                    multiplier=mult, range_group=group, range_role=role,
                ))
            claimed.append((m.start(), m.end()))
    return spans, claimed


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _resolve_overlaps(candidates: list[MoneySpan]) -> list[MoneySpan]:
    """Keep the highest-quality non-overlapping matches.

    Ranked by pattern priority, then span length, then confidence — a longer
    match at equal priority carries more evidence (`AUD$21.9 million` over
    `$21.9`), so length breaks priority ties rather than the reverse.
    """
    ordered = sorted(
        candidates,
        key=lambda c: (_PRIORITY[c.evidence], -(c.end - c.start), -c.confidence, c.start),
    )
    # `kept` is maintained sorted and disjoint, so only the neighbours either
    # side of the insertion point can overlap — no full rescan per candidate.
    kept: list[MoneySpan] = []
    starts: list[int] = []
    for cand in ordered:
        i = bisect_right(starts, cand.start)
        if i > 0 and kept[i - 1].end > cand.start:
            continue
        if i < len(kept) and kept[i].start < cand.end:
            continue
        kept.insert(i, cand)
        starts.insert(i, cand.start)
    return kept


def _attach_modifier(text: str, span: MoneySpan) -> None:
    """Record `approximately` / `up to` / `~` separately — never in the value."""
    prefix = text[max(0, span.start - 32):span.start]
    m = MODIFIER_RE.search(prefix)
    if m:
        span.modifier = m.group(0).strip().lower()


def context_for(text: str, span: MoneySpan, width: int) -> str:
    """Surrounding sentence-ish window, capped at *width* characters."""
    if width <= 0:
        return ""
    half = width // 2
    start = max(0, span.start - half)
    end = min(len(text), span.end + half)
    return " ".join(text[start:end].split())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def find_money(text: str, options: MoneyOptions | None = None) -> list[MoneySpan]:
    """Extract self-evidencing monetary amounts from *text*.

    Offsets index *text* exactly as handed in — no normalisation, no rewriting.
    Returns spans sorted by position, filtered to ``min_confidence``.
    """
    if not text:
        return []
    opts = options or MoneyOptions()
    pats = _patterns(opts.international_numbers)
    blocked = _IntervalIndex([(s, e) for s, e, _ in blocked_spans(text)])

    ranges, claimed = _scan_ranges(text, pats, opts, blocked)
    claimed_index = _IntervalIndex(claimed)

    candidates: list[MoneySpan] = []
    for scan in (_scan_symbol_prefix, _scan_iso_prefix, _scan_iso_suffix,
                 _scan_word, _scan_symbol_suffix, _scan_accounting):
        candidates.extend(scan(text, pats, opts))
    if opts.implicit_context:
        candidates.extend(_scan_implicit(text, pats, opts))

    candidates = [
        c for c in candidates
        if not blocked.overlaps(c.start, c.end)
        and not claimed_index.overlaps(c.start, c.end)
    ]

    resolved = _resolve_overlaps(candidates) + ranges
    resolved.sort(key=lambda c: (c.start, c.end))

    out = []
    for span in resolved:
        if span.confidence < opts.min_confidence:
            continue
        _attach_modifier(text, span)
        out.append(span)
    return out


__all__ = [
    "MoneyOptions",
    "MoneySpan",
    "apply_scale",
    "blocked_spans",
    "context_for",
    "find_money",
    "parse_number",
    "resolve_iso",
    "resolve_symbol",
]
