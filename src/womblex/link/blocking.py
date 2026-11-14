"""Candidate narrowing for fuzzy name matching.

Scoring a mention against every name in a register is exact and, for a register
of a few hundred entities, fast. It is quadratic in practice for a register of
10⁵: each score is a ``difflib`` comparison. :class:`NameIndex` narrows first —
it indexes every name by its character trigrams, and for a mention returns only
the names that share the most trigrams with it, which are then scored exactly as
before.

Trigrams, not tokens, because the mentions are OCR output: ``earty leaming``
shares no token with ``early learning`` and most of its trigrams. A trigram that
a large share of the register holds (``pty``, ``ltd``) says nothing about which
entity is meant and is skipped unless the mention has no rarer one.

Narrowing can only lose a match the full scan would have found, never invent
one: every returned name is scored by the same function, against the same
threshold.
"""

from __future__ import annotations

from collections import Counter

from womblex.link.reference import ReferenceEntity

#: A trigram held by more than this share of the names is too common to narrow by.
_COMMON_SHARE = 0.05
_COMMON_FLOOR = 50


def _trigrams(name: str) -> set[str]:
    padded = f" {name} "
    return {padded[i:i + 3] for i in range(len(padded) - 2)}


class NameIndex:
    """Trigram index over every fuzzy name of a register."""

    def __init__(self, entities: list[ReferenceEntity]) -> None:
        self._names: list[tuple[str, ReferenceEntity]] = [
            (key, e) for e in entities for key in e.fuzzy_keys
        ]
        self._postings: dict[str, list[int]] = {}
        for i, (key, _entity) in enumerate(self._names):
            for gram in _trigrams(key):
                self._postings.setdefault(gram, []).append(i)
        self._common = max(_COMMON_FLOOR, int(len(self._names) * _COMMON_SHARE))

    def __len__(self) -> int:
        return len(self._names)

    def nearest(self, query: str, limit: int) -> list[tuple[str, ReferenceEntity]]:
        """Up to *limit* names sharing the most trigrams with *query*, best first."""
        held = [p for g in _trigrams(query) if (p := self._postings.get(g)) is not None]
        rare = [p for p in held if len(p) <= self._common]
        counts: Counter[int] = Counter()
        for posting in rare or held:
            counts.update(posting)
        # Ties break on position in the register, so the result does not depend on hash order.
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
        return [self._names[i] for i, _n in ranked]


__all__ = ["NameIndex"]
