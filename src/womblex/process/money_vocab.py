"""Currency, scale and false-positive vocabulary for the money op. Data only.

Split out of :mod:`womblex.process.money` so the detector stays readable and
the tables stay auditable. Nothing here executes over documents; it is
lookups and compiled regexes.

The vocabulary is *language*-level (Australian English money conventions), not
dataset-level, so it ships as a default rather than an empty corpus-supplied
list. Corpus-specific additions come through ``MoneyConfig`` extension points.

Currency confidence is tiered (``docs/money.md``): tier 1 is Australian, tier 2
the internationals that recur in procurement / treasury / trade reporting, tier
3 is the rest of ISO 4217, admitted but never on three-uppercase-letters alone
without being a member of the list.
"""

from __future__ import annotations

import re
from decimal import Decimal

# ---------------------------------------------------------------------------
# Currency tiers
# ---------------------------------------------------------------------------

TIER1: frozenset[str] = frozenset({"AUD"})

TIER2: frozenset[str] = frozenset({
    "USD", "NZD", "GBP", "EUR", "JPY", "CAD", "SGD", "CHF", "HKD", "CNY",
})

# Full ISO 4217 active alphabetic codes. Membership is what makes three
# uppercase letters a currency — `ABC` and `XYZ` are not currencies.
ISO_4217: frozenset[str] = frozenset("""
AED AFN ALL AMD ANG AOA ARS AUD AWG AZN BAM BBD BDT BGN BHD BIF BMD BND BOB
BOV BRL BSD BTN BWP BYN BZD CAD CDF CHE CHF CHW CLF CLP CNY COP COU CRC CUP
CVE CZK DJF DKK DOP DZD EGP ERN ETB EUR FJD FKP GBP GEL GHS GIP GMD GNF GTQ
GYD HKD HNL HTG HUF IDR ILS INR IQD IRR ISK JMD JOD JPY KES KGS KHR KMF KPW
KRW KWD KYD KZT LAK LBP LKR LRD LSL LYD MAD MDL MGA MKD MMK MNT MOP MRU MUR
MVR MWK MXN MXV MYR MZN NAD NGN NIO NOK NPR NZD OMR PAB PEN PGK PHP PKR PLN
PYG QAR RON RSD RUB RWF SAR SBD SCR SDG SEK SGD SHP SLE SOS SRD SSP STN SVC
SYP SZL THB TJS TMT TND TOP TRY TTD TWD TZS UAH UGX USD USN UYI UYU UYW UZS
VED VES VND VUV WST XAF XCD XCG XDR XOF XPF YER ZAR ZMW ZWG
""".split())

# `RMB` is not an ISO code but is used interchangeably with CNY in trade
# reporting; `XBT` is the common non-ISO alias for bitcoin.
CODE_ALIASES: dict[str, str] = {"RMB": "CNY", "XBT": "XBT"}


def currency_tier(code: str | None) -> int:
    """Return 1/2/3 for a resolved code; 3 for anything outside tiers 1-2."""
    if code in TIER1:
        return 1
    if code in TIER2:
        return 2
    return 3


# ---------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------

# Longest-first: `AU$` must win over `$`, `US$` over `$`.
SYMBOL_TO_CODE: dict[str, str] = {
    "AU$": "AUD", "A$": "AUD", "US$": "USD", "NZ$": "NZD", "CA$": "CAD",
    "S$": "SGD", "HK$": "HKD", "NT$": "TWD", "C$": "CAD",
    "$": "AUD",          # Australian document convention — see docs/money.md
    "€": "EUR", "£": "GBP", "¥": "JPY", "₹": "INR", "₩": "KRW",
    "₽": "RUB", "₿": "XBT", "﹩": "AUD", "＄": "AUD",
}

_SYMBOL_ALT = "|".join(re.escape(s) for s in sorted(SYMBOL_TO_CODE, key=len, reverse=True))

# Suffix symbols only — `100$`, `50€`. Prefixed-letter forms (`A$`) never
# trail a number in this corpus and admitting them would match `5 US$`-style
# noise, so the suffix set is the bare symbols.
_SUFFIX_SYMBOLS = ("$", "€", "£", "¥", "₹", "₩", "₽")
_SUFFIX_SYMBOL_ALT = "|".join(re.escape(s) for s in _SUFFIX_SYMBOLS)


# ---------------------------------------------------------------------------
# Currency words
# ---------------------------------------------------------------------------

# A ``None`` code means "money-marked but currency unresolved" — `peso` and
# `franc` name several currencies and guessing one would be an invention. The
# span is still an extraction; its `currency` is null.
CURRENCY_WORDS: dict[str, str | None] = {
    "australian dollar": "AUD", "australian dollars": "AUD",
    "us dollar": "USD", "us dollars": "USD",
    "united states dollar": "USD", "united states dollars": "USD",
    "new zealand dollar": "NZD", "new zealand dollars": "NZD",
    "canadian dollar": "CAD", "canadian dollars": "CAD",
    "singapore dollar": "SGD", "singapore dollars": "SGD",
    "hong kong dollar": "HKD", "hong kong dollars": "HKD",
    "dollar": "AUD", "dollars": "AUD",
    "cent": "AUD", "cents": "AUD",
    "euro": "EUR", "euros": "EUR",
    "pound": "GBP", "pounds": "GBP", "sterling": "GBP",
    "pound sterling": "GBP", "pounds sterling": "GBP",
    "yen": "JPY", "yuan": "CNY", "renminbi": "CNY",
    "rupee": "INR", "rupees": "INR",
    "won": "KRW", "ruble": "RUB", "rubles": "RUB",
    "rouble": "RUB", "roubles": "RUB",
    "dirham": "AED", "dirhams": "AED",
    "peso": None, "pesos": None, "franc": None, "francs": None,
}

# Sub-unit words divide rather than multiply.
SUBUNIT_WORDS: frozenset[str] = frozenset({"cent", "cents"})

_WORD_ALT = "|".join(
    re.escape(w) for w in sorted(CURRENCY_WORDS, key=len, reverse=True)
)


# ---------------------------------------------------------------------------
# Scale
# ---------------------------------------------------------------------------

SCALES: dict[str, Decimal] = {
    "k": Decimal(10) ** 3,
    "thousand": Decimal(10) ** 3,
    "thousands": Decimal(10) ** 3,
    "m": Decimal(10) ** 6,
    "mn": Decimal(10) ** 6,
    "million": Decimal(10) ** 6,
    "millions": Decimal(10) ** 6,
    "b": Decimal(10) ** 9,
    "bn": Decimal(10) ** 9,
    "billion": Decimal(10) ** 9,
    "billions": Decimal(10) ** 9,
    "t": Decimal(10) ** 12,
    "tn": Decimal(10) ** 12,
    "trillion": Decimal(10) ** 12,
    "trillions": Decimal(10) ** 12,
}

# Single letters are multipliers only next to a currency marker (`$100m`), never
# on their own — the gate that rejects `100m road`, `50m radius`, `20m hose`.
AMBIGUOUS_SCALES: frozenset[str] = frozenset({"k", "m", "b", "t"})

_SCALE_ALT = "|".join(re.escape(s) for s in sorted(SCALES, key=len, reverse=True))


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

# Australian: comma groups, dot decimal. `1.000,50` is deliberately not an
# Australian amount (docs/money.md) — it is admitted only in international mode.
NUM_AU = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+"
NUM_INTL = r"(?:\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+,\d+)"


# ---------------------------------------------------------------------------
# Modifiers (qualifiers) — stored separately, never folded into the value
# ---------------------------------------------------------------------------

MODIFIERS: tuple[str, ...] = (
    "approximately", "approx.", "approx", "about", "around", "circa",
    "no more than", "not more than", "no less than", "not less than",
    "at least", "at most", "up to", "more than", "less than",
    "greater than", "in excess of", "over", "under", "nearly", "almost",
    "~", ">=", "<=", ">", "<",
)

_MODIFIER_ALT = "|".join(re.escape(m) for m in sorted(MODIFIERS, key=len, reverse=True))
MODIFIER_RE = re.compile(rf"(?:{_MODIFIER_ALT})\s*$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Implicit financial context (pattern 10) — low precision, default off
# ---------------------------------------------------------------------------

CONTEXT_TRIGGERS: tuple[str, ...] = (
    "cost", "price", "fee", "charge", "payment", "salary", "income", "wage",
    "expense", "budget", "appropriation", "grant", "funding", "allocation",
    "revenue", "profit", "loss", "compensation", "claim", "benefit", "rebate",
    "levy", "fine", "penalty", "premium", "excess", "deductible", "invoice",
    "quote", "estimate", "contract value", "replacement value", "sum insured",
)

CONTEXT_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in CONTEXT_TRIGGERS) + r")\w*\b",
    re.IGNORECASE,
)

# Accounting context gates bracketed negatives outside a classified column.
ACCOUNTING_TRIGGERS: tuple[str, ...] = (
    "total", "subtotal", "balance", "deficit", "surplus", "net", "gross",
    "expenditure", "expenses", "revenue", "income", "statement", "accrual",
    "depreciation", "amortisation", "amortization", "equity", "liabilities",
    "assets", "cash flow", "comprehensive income", "financial position",
)

ACCOUNTING_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in ACCOUNTING_TRIGGERS) + r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Australian false-positive classes
# ---------------------------------------------------------------------------

_MEASURE_UNITS = (
    "mm", "cm", "km", "ha", "sqm", "m2", "m3", "kg", "mg", "kt", "mt", "kw",
    "mw", "gw", "kwh", "mwh", "gwh", "ml", "gl", "kl", "tj", "pj", "db", "kb",
    "mb", "gb", "tb", "km2", "m", "kms", "metres", "meters", "kilometres",
    "kilograms", "tonnes", "hectares", "litres", "megalitres", "gigalitres",
    "days", "months", "years", "hours", "minutes", "seconds", "fte", "pages",
)

# Each entry blocks any candidate whose span overlaps it. Ordered only for
# readability — all are applied.
FALSE_POSITIVE_PATTERNS: dict[str, re.Pattern[str]] = {
    "date_numeric": re.compile(r"\b\d{1,4}[/\-.]\d{1,2}[/\-.]\d{2,4}\b"),
    "date_worded": re.compile(
        r"\b\d{1,2}\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
        r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|"
        r"nov(?:ember)?|dec(?:ember)?)\s+\d{2,4}\b",
        re.IGNORECASE,
    ),
    "date_worded_leading": re.compile(
        r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|"
        r"dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b",
        re.IGNORECASE,
    ),
    "financial_year": re.compile(r"\b(?:19|20)\d{2}\s*[-–/]\s*(?:\d{2}|(?:19|20)\d{2})\b"),
    # Dotted times must carry a meridiem / `hrs` marker: `10.30am` is a time,
    # but `$5.20` is an amount and a bare `\d{1,2}[.]\d{2}` would swallow it.
    "time": re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm|hrs|hours)?\b",
                       re.IGNORECASE),
    "time_dotted": re.compile(r"\b\d{1,2}\.\d{2}\s?(?:am|pm|hrs|hours)\b", re.IGNORECASE),
    "time_24h": re.compile(r"\b\d{4}\s*(?:hrs|hours)\b", re.IGNORECASE),
    "phone": re.compile(
        r"(?:\(0\d\)\s*\d{4}\s*\d{4}|\b0[2-8]\s*\d{4}\s*\d{4}\b|"
        r"\b04\d{2}\s*\d{3}\s*\d{3}\b|\b1[38]00\s*\d{3}\s*\d{3}\b|"
        r"\b13\s*\d{2}\s*\d{2}\b|\+61\s*\d[\d\s]{7,})"
    ),
    "abn": re.compile(r"\b\d{2}\s?\d{3}\s?\d{3}\s?\d{3}\b"),
    "acn": re.compile(r"\b\d{3}\s\d{3}\s\d{3}\b"),
    "legislative": re.compile(
        r"\b(?:section|sections|s|ss|subsection|clause|cl|schedule|sch|part|"
        r"division|div|regulation|reg|item|paragraph|para|chapter|ch|rule|"
        r"article|art|table|figure|appendix|attachment|page|pp?)\s?\.?\s?"
        r"\d+[A-Za-z]*(?:\(\d+\)|\([a-z]\))*",
        re.IGNORECASE,
    ),
    "incident_ref": re.compile(r"\b[A-Z]{2,4}[-/]?\d{3,}(?:/\d+)?\b"),
    "parcel": re.compile(r"\b(?:lot|dp|sp|pid|lga)\s?\d+\b", re.IGNORECASE),
    "measurement": re.compile(
        r"\b\d+(?:[,.]\d+)*\s?(?:" + "|".join(_MEASURE_UNITS) + r")\b",
        re.IGNORECASE,
    ),
    "temperature": re.compile(r"\b\d+(?:\.\d+)?\s?°[CF]?"),
    "percent": re.compile(r"\b\d+(?:[,.]\d+)*\s?(?:%|per\s?cent|percent)\b", re.IGNORECASE),
    "version": re.compile(r"\bv(?:ersion)?\s?\d+(?:\.\d+)+\b", re.IGNORECASE),
}

# Postcodes are rejected only where address context exists — a bare `2600` is
# far more often a money-column value than a postcode.
STATE_RE = re.compile(r"\b(?:NSW|VIC|QLD|SA|WA|TAS|NT|ACT)\b")
POSTCODE_RE = re.compile(r"\b\d{4}\b")

# Null / absent markers in financial tables. Excluded from the numeric fraction
# rather than counted against it (docs/money.md) — counting them as non-numeric
# suppresses genuine money columns.
NULL_MARKERS: frozenset[str] = frozenset({
    "", "-", "–", "—", "‒", "―", "n/a", "na", "n.a.", "nil", "none", "null",
    ".", "..", "...", "*", "tbc", "tba", "not applicable", "not available",
})


# ---------------------------------------------------------------------------
# Column header vocabulary
# ---------------------------------------------------------------------------

# Money vocabulary applied to *headers*, where docs/money.md measures it as the
# primary signal (as opposed to narrative, where it is low precision).
HEADER_MONEY_TERMS: frozenset[str] = frozenset(set(CONTEXT_TRIGGERS) | {
    "value", "amount", "total", "sum", "paid", "payable", "spend", "spending",
    "gst", "aud", "$", "$m", "$'000", "consideration", "remuneration",
    "contribution", "subsidy", "reimbursement", "turnover", "receipts",
    "outlays", "commitment", "commitments", "liability", "liabilities",
    "expenditure", "expenditures", "expenses", "costs", "fees", "payments",
    "asset", "assets", "balance", "threshold", "cap",
})

# A veto suppresses the column even when its cells are numeric and
# thousands-separated. Matching is whole-word — `age` must not veto
# `Average Cost`.
HEADER_VETO_TERMS: frozenset[str] = frozenset({
    "postcode", "postal", "abn", "acn", "arbn", "id", "ids", "identifier",
    "code", "count", "number", "no", "num", "qty", "quantity", "phone",
    "telephone", "mobile", "fax", "year", "years", "date", "dates", "month",
    "day", "time", "percent", "percentage", "%", "rate", "rates", "ratio",
    "index", "score", "rank", "age", "fte", "headcount", "staff", "latitude",
    "longitude", "lat", "lon", "lng", "postcode/suburb", "abn/acn", "version",
    "page", "pages", "row", "column", "sequence", "seq", "reference", "ref",
})

_TOKEN_RE = re.compile(r"[a-z$%']+", re.IGNORECASE)


def header_tokens(header: str) -> list[str]:
    """Lowercase word tokens of a header, for whole-word vocabulary matching."""
    return [t.lower() for t in _TOKEN_RE.findall(header or "")]


__all__ = [
    "ACCOUNTING_RE",
    "AMBIGUOUS_SCALES",
    "CODE_ALIASES",
    "CONTEXT_RE",
    "CONTEXT_TRIGGERS",
    "CURRENCY_WORDS",
    "FALSE_POSITIVE_PATTERNS",
    "HEADER_MONEY_TERMS",
    "HEADER_VETO_TERMS",
    "ISO_4217",
    "MODIFIER_RE",
    "MODIFIERS",
    "NULL_MARKERS",
    "NUM_AU",
    "NUM_INTL",
    "POSTCODE_RE",
    "SCALES",
    "STATE_RE",
    "SUBUNIT_WORDS",
    "SYMBOL_TO_CODE",
    "TIER1",
    "TIER2",
    "_SCALE_ALT",
    "_SUFFIX_SYMBOL_ALT",
    "_SYMBOL_ALT",
    "_WORD_ALT",
    "currency_tier",
    "header_tokens",
]
