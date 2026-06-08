"""Constrained, dictionary-gated OCR character-confusion repair.

This is **not** a general spell-corrector and deliberately not the rejected
substitution-table OCR fix in ``docs/decisions.md`` (which enumerated errors as
fixed rules). It validates candidates against the bundled en_AU Hunspell
dictionary (read with the pure-Python ``spylls`` engine) and only rewrites a
token when **all three gates** pass, so valid words and ambiguous cases are
never touched:

1. *Trigger gate* — the token is **out-of-dictionary**. In-dictionary words are
   immune by construction.
2. *Candidate gate* — only single-character edits are considered, and a
   candidate is kept only if it **is** in the dictionary.
3. *Unambiguity gate* — the correction is applied only when **exactly one**
   in-dictionary candidate exists; otherwise the token is left verbatim.

Two tightness tiers:

- **Tier A (homoglyph, default)** — substitutes only OCR-plausible digit→letter
  glyph confusions (``chi1d``→``child``, ``p1an``→``plan``). Length-preserving,
  near-zero false positives because an embedded digit is itself a strong OCR
  signal, and it almost never touches real proper nouns.
- **Tier B (general edit-distance-1, opt-in)** — adds insert/delete/substitute/
  transpose candidates over ``a-z``. Higher recall, but carries a real
  proper-noun corruption risk (a surname one edit from a dictionary word), which
  is why it is gated behind a flag.

The en_AU dictionary is the MIT/SCOWL Hunspell dictionary harvested from the
Australian Writing MCP; ``spylls`` is the Hunspell algorithm in pure Python.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from spylls.hunspell import Dictionary

from womblex.utils.models import resolve_local_model_path

# OCR digit→letter glyph confusions (Tier A). Lowercase targets only — the
# dictionary lookup folds case. Kept tight on purpose: an embedded digit is the
# signal, so we only ever swap the digit, never letters (that is Tier B).
_HOMOGLYPHS: dict[str, tuple[str, ...]] = {
    "0": ("o",),
    "1": ("l", "i"),
    "2": ("z",),
    "3": ("e",),
    "4": ("a",),
    "5": ("s",),
    "6": ("b", "g"),
    "7": ("t",),
    "8": ("b",),
    "9": ("g", "q"),
}

# Word-like tokens that may carry an embedded digit (so ``chi1d`` is one token).
_WORD_RE = re.compile(r"[0-9A-Za-z]+(?:['’\-][0-9A-Za-z]+)*")

_ALPHABET = "abcdefghijklmnopqrstuvwxyz"

# Guard rails: shorter tokens are too ambiguous to repair safely; tokens with a
# lot of digits are codes/ids, not OCR'd words.
_MIN_LEN = 3
_MAX_DIGITS = 2


@dataclass(frozen=True)
class Correction:
    """One applied token rewrite, recorded for the audit sidecar."""

    offset: int      # char offset of the token in the (chunk) text
    original: str
    corrected: str
    method: str      # "homoglyph" | "edit1"


@lru_cache(maxsize=4)
def _dictionary(dict_name: str) -> Dictionary:
    """Load the bundled Hunspell dictionary (resolved via ``utils.models``)."""
    base = resolve_local_model_path(dict_name)
    index = str(Path(base) / "index") if isinstance(base, Path) else f"{base}/index"
    return Dictionary.from_files(index)


def _in_dict(d: Dictionary, word: str) -> bool:
    return bool(d.lookup(word) or d.lookup(word.lower()) or d.lookup(word.capitalize()))


def _match_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement.capitalize()
    return replacement


def _homoglyph_candidates(lower: str, d: Dictionary) -> set[str]:
    """In-dict words reachable by swapping exactly one digit for a letter."""
    out: set[str] = set()
    for i, ch in enumerate(lower):
        for repl in _HOMOGLYPHS.get(ch, ()):
            cand = lower[:i] + repl + lower[i + 1 :]
            if not any(c.isdigit() for c in cand) and _in_dict(d, cand):
                out.add(cand)
    return out


def _edit1_candidates(lower: str, d: Dictionary) -> set[str]:
    """In-dict words at Damerau-Levenshtein distance 1 over ``a-z`` (Tier B)."""
    splits = [(lower[:i], lower[i:]) for i in range(len(lower) + 1)]
    cands: set[str] = set()
    for left, right in splits:
        if right:  # delete
            cands.add(left + right[1:])
        if len(right) > 1:  # transpose
            cands.add(left + right[1] + right[0] + right[2:])
        for c in _ALPHABET:
            if right:  # substitute
                cands.add(left + c + right[1:])
            cands.add(left + c + right)  # insert
    return {w for w in cands if w != lower and "0" not in w and _in_dict(d, w)}


@lru_cache(maxsize=100_000)
def _correct_token(token: str, general: bool, dict_name: str) -> tuple[str, str] | None:
    """Return ``(corrected, method)`` for one token, or ``None`` to leave it.

    Caches per (token, tier, dict) so repeated tokens across a corpus are cheap.
    """
    letters = sum(c.isalpha() for c in token)
    digits = sum(c.isdigit() for c in token)
    if len(token) < _MIN_LEN or letters < 2 or digits > _MAX_DIGITS:
        return None
    if token.isupper():  # acronyms / initialisms — never a misspelt word
        return None

    d = _dictionary(dict_name)
    if _in_dict(d, token):
        return None

    lower = token.lower()
    candidates = _homoglyph_candidates(lower, d)
    method = "homoglyph"
    if not candidates and general:
        candidates = _edit1_candidates(lower, d)
        method = "edit1"

    if len(candidates) != 1:  # unambiguity gate (also covers the empty case)
        return None
    return _match_case(token, next(iter(candidates))), method


def repair_text(
    text: str,
    *,
    general_edits: bool = False,
    dict_name: str = "en_AU",
) -> tuple[str, list[Correction]]:
    """Repair OCR character-confusions in ``text``; return ``(text, corrections)``.

    Only out-of-dictionary tokens with a single unambiguous in-dictionary
    candidate are rewritten. With ``general_edits=False`` (default) only Tier A
    homoglyph swaps fire (length-preserving); ``general_edits=True`` enables the
    broader, riskier Tier B edit-distance-1 fallback.
    """
    if not text:
        return text, []

    corrections: list[Correction] = []
    out: list[str] = []
    last = 0
    for m in _WORD_RE.finditer(text):
        token = m.group()
        result = _correct_token(token, general_edits, dict_name)
        if result is None:
            continue
        corrected, method = result
        out.append(text[last : m.start()])
        out.append(corrected)
        last = m.end()
        corrections.append(
            Correction(offset=m.start(), original=token, corrected=corrected, method=method)
        )

    if not corrections:
        return text, []
    out.append(text[last:])
    return "".join(out), corrections


__all__ = ["Correction", "repair_text"]
