"""Link candidates read straight from the text (``linking.candidate_source: text``).

The enrichment-driven path matches the mentions a model found. This one needs no
model and sends nothing anywhere: it looks for the registers' own names in the
narrative, by exact phrase match on normalised words, longest name first. It is
how a place gazetteer or an entity register is applied to a corpus that is never
enriched.

A name is a mention only where the text writes it as a proper noun (its first
and last word capitalised), and never across a paragraph break.

**One-word names need a second look.** ``Young`` and ``Grant`` are towns and
surnames; ``young people`` is neither. Each register says how its one-word names
are treated (``single_word``): linked, never linked, or ``cued`` — linked only
beside something that locates them, and never where the words around them make a
person's name:

=====================  ===================================================
``person:honorific``   ``Mr Young`` — not linked
``person:name``        ``Peter Young``, ``Grant Thornton`` — not linked
``cue:<word>``         ``in Young``, ``Young Shire``, ``City of Young``
``cue:state``          ``Young NSW``, ``Young, 2594``
``cue:heading``        the name in a heading
``uncued``             none of these — not linked
=====================  ===================================================

A name that two entities share (``Richmond``) is not linked either
(``entities:<n>``). Every such mention is still written, unmatched, with the
reason, so what was left out can be inspected.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise

from womblex.config import ReferenceConfig
from womblex.link.matcher import Candidate, Link
from womblex.link.normalise import normalise_name
from womblex.link.reference import ReferenceEntity, ReferenceTable

_WORD_RE = re.compile(r"\w+")
# Between two words of one name: spaces, a line wrap, the punctuation names carry.
_NAME_GAP_RE = re.compile(r"[^\S\n]*[.,&'’()\-/]?[^\S\n]*\n?[^\S\n]*[(]?")
_HONORIFICS = frozenset({
    "mr", "mrs", "ms", "miss", "mx", "dr", "prof", "professor", "sir", "dame", "hon",
    "cr", "councillor", "senator", "minister", "justice", "judge", "commissioner",
})
_LOCATIVES = frozenset({
    "in", "at", "near", "from", "to", "around", "across", "between", "within", "outside",
    "throughout",
})
_PLACE_TYPES = frozenset({
    "shire", "council", "city", "town", "township", "region", "district", "area",
    "municipality", "electorate", "lga", "suburb", "valley",
})
# A postcode in brackets, or a year after a comma, is an author-year citation
# (`Young (2019)`, `Young, 2019`), not a place.
_STATE_RE = re.compile(
    r"[^\S\n]*,?[^\S\n]*\(?(?:NSW|VIC|Vic|QLD|Qld|SA|WA|TAS|Tas|NT|ACT)\b"
    r"|[^\S\n]*\d{4}\b|[^\S\n]*,[^\S\n]*(?!(?:19|20)\d{2})\d{4}\b")
_SENTENCE_END = frozenset(".!?:;")


@dataclass(frozen=True, slots=True)
class _Word:
    text: str       # as written
    key: str        # casefolded
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _Named:
    entity: ReferenceEntity
    register: ReferenceConfig
    alias: bool


class PhraseBook:
    """Every name of every register, keyed by its normalised words."""

    def __init__(self, registers: list[tuple[ReferenceConfig, ReferenceTable]]) -> None:
        self._names: dict[tuple[str, ...], list[_Named]] = defaultdict(list)
        for config, table in registers:
            for entity in table.entities:
                for name in {normalise_name(entity.name), *entity.fuzzy_keys}:
                    self._add(name, _Named(entity, config, alias=False))
            for alias, entity_id in table.aliases.items():
                if (target := table.entity_by_id(entity_id)) is not None:
                    self._add(alias, _Named(target, config, alias=True))
        self._lengths: dict[str, list[int]] = defaultdict(list)
        for words in self._names:
            if len(words) not in self._lengths[words[0]]:
                self._lengths[words[0]].append(len(words))
        for lengths in self._lengths.values():
            lengths.sort(reverse=True)

    def _add(self, name: str, named: _Named) -> None:
        words = tuple(name.split())
        if words and all(n.entity is not named.entity for n in self._names[words]):
            self._names[words].append(named)

    def __len__(self) -> int:
        return len(self._names)

    def find(
        self, text: str, source_hash: str, *, text_layer: str,
        heading_spans: list[tuple[int, int]] | None = None,
    ) -> list[Link]:
        """Every register name written in *text*, as links; offsets index *text*."""
        words = [_Word(m.group(), m.group().casefold(), m.start(), m.end())
                 for m in _WORD_RE.finditer(text)]
        links: list[Link] = []
        i = 0
        while i < len(words):
            n = self._longest(text, words, i)
            if n == 0:
                i += 1
                continue
            first, last = words[i], words[i + n - 1]
            named = self._names[tuple(w.key for w in words[i:i + n])]
            candidate = Candidate(
                text=text[first.start:last.end], kind="text", source_hash=source_hash,
                mention_start=first.start, mention_end=last.end, text_layer=text_layer,
                mention_text=text[first.start:last.end],
            )
            in_heading = any(s <= first.start < e for s, e in heading_spans or ())
            link = _decide(candidate, named, text, words, i, n, in_heading)
            if link is not None:
                links.append(link)
            i += n
        return links

    def _longest(self, text: str, words: list[_Word], i: int) -> int:
        """Length in words of the longest name starting at ``words[i]``; 0 for none."""
        for n in self._lengths.get(words[i].key, ()):
            run = words[i:i + n]
            if len(run) < n or tuple(w.key for w in run) not in self._names:
                continue
            if not (_capitalised(run[0]) and _capitalised(run[-1])):
                continue
            if all(_NAME_GAP_RE.fullmatch(text, a.end, b.start) for a, b in pairwise(run)):
                return n
        return 0


def _capitalised(word: _Word) -> bool:
    return word.text[0].isupper() or word.text[0].isdigit()


def _decide(
    candidate: Candidate, named: list[_Named], text: str, words: list[_Word],
    i: int, n: int, in_heading: bool,
) -> Link | None:
    """The link for one name found in the text; ``None`` where its register skips it."""
    if n == 1:
        named = [x for x in named if x.register.single_word != "skip"]
        if not named:
            return None
    if len(named) > 1:
        return Link(candidate, None, 0.0, "ambiguous", detail=f"entities:{len(named)}")
    one = named[0]
    detail: str | None = None
    if n == 1 and one.register.single_word == "cued":
        detail = _cue(text, words, i, in_heading, one.register)
        if not detail.startswith("cue:"):
            return Link(candidate, None, 0.0, "ambiguous", detail=detail)
    return Link(candidate, one.entity, 1.0, "alias" if one.alias else "name_exact", detail=detail)


def _cue(
    text: str, words: list[_Word], i: int, in_heading: bool, register: ReferenceConfig,
) -> str:
    """What the words around a one-word name say it is (see the module table)."""
    word = words[i]
    before = _neighbour(text, words, i, -1)
    after = _neighbour(text, words, i, +1)
    if before is not None and before.key in _HONORIFICS:
        return "person:honorific"
    locatives = _LOCATIVES | {c.casefold() for c in register.cue_words}
    located = before is not None and before.key in locatives
    # `City of Young`: `of` locates only after a word that names a kind of place.
    if before is not None and before.key == "of":
        two_back = _neighbour(text, words, i - 1, -1)
        located = two_back is not None and two_back.key in _PLACE_TYPES
    if after is not None and after.key in _PLACE_TYPES:
        return f"cue:{after.key}"
    if _STATE_RE.match(text, word.end):
        return "cue:state"
    # A capitalised neighbour that is not a cue makes a longer name the register does not hold.
    if after is not None and _capitalised(after) and not after.text[0].isdigit():
        return "person:name"
    if located:
        return f"cue:{before.key}"  # type: ignore[union-attr]
    if before is not None and before.text[0].isupper():
        return "person:name"
    return "cue:heading" if in_heading else "uncued"


def _neighbour(text: str, words: list[_Word], i: int, step: int) -> _Word | None:
    """The word next to ``words[i]`` in the same clause, or ``None``."""
    j = i + step
    if not 0 <= i < len(words) or not 0 <= j < len(words):
        return None
    left, right = (words[j], words[i]) if step < 0 else (words[i], words[j])
    gap = text[left.end:right.start]
    if "\n" in gap or any(ch in _SENTENCE_END for ch in gap) or (step > 0 and "," in gap):
        return None
    return words[j]


__all__ = ["PhraseBook"]
