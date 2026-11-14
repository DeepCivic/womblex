"""Vocabulary for the ``stats`` annotation op: measures and cues.

Measures are never built in. A corpus declares each one, with the phrases that
name it, in its config or in a vocabulary file the config points at; the library
only matches them. Cues are the phrases that qualify a figure, in four types:

=============== ==========================================================
``calculation`` how the figure was derived (``per 100,000``, ``share of``,
                ``in real terms``, ``seasonally adjusted``)
``source``      where it came from (``Source: …``, ``cat. no. 6202.0``)
``gap``         that a figure is absent (``not available``, ``n.p.``)
``forecast``    that it is not an observation (``projected``, ``forecast``)
=============== ==========================================================

The built-in cues are the general vocabulary of statistical publications; a
corpus adds its own per type. Every phrase is matched whole-word and on one
line. :func:`vocabulary_digest` is what the sidecar footer records, so a file
names the vocabulary that produced it.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from womblex.config import StatsConfig, StatsVocabulary

_WS = r"[^\S\n]+"
_N = r"(?:\d{1,3}(?:,\d{3})+|\d+)"

# (cue type, canonical name, pattern). `(?-i:…)` keeps an acronym case-sensitive.
_BUILTIN_CUES: tuple[tuple[str, str, str], ...] = (
    ("calculation", "per_n", rf"per{_WS}(?P<n>{_N})(?!\.\d)"),
    ("calculation", "per_capita", rf"per{_WS}(?:capita|person|head)"),
    ("calculation", "share", rf"(?:share|proportion|percentage){_WS}of"),
    ("calculation", "real_terms", rf"(?:in{_WS})?real{_WS}terms"),
    ("calculation", "nominal_terms", rf"(?:in{_WS})?nominal{_WS}terms"),
    ("calculation", "seasonally_adjusted", rf"seasonally{_WS}adjusted"),
    ("calculation", "trend", rf"(?:in{_WS})?trend{_WS}(?:terms|estimates?|series)"),
    ("calculation", "chain_volume", rf"chain{_WS}volume(?:{_WS}measures?)?"),
    ("calculation", "current_prices", rf"(?:in{_WS})?current{_WS}prices"),
    ("calculation", "annualised", r"annuali[sz]ed"),
    ("calculation", "age_standardised", r"age[- ]standardi[sz]ed"),
    ("source", "source_line", r"sources?[^\S\n]*:[^\S\n]*[^\n]+"),
    ("source", "catalogue_number", rf"cat(?:alogue)?\.?{_WS}no\.?[^\S\n]*\d{{4}}\.\d(?:\.\d{{2}}\.\d{{3}})?"),
    ("source", "abs", rf"Australian{_WS}Bureau{_WS}of{_WS}Statistics|(?-i:ABS)"),
    ("source", "according_to", rf"according{_WS}to"),
    ("gap", "not_available", rf"not{_WS}(?:yet{_WS})?available|(?-i:n\.a\.)"),
    ("gap", "not_published", rf"not{_WS}(?:for{_WS}publication|published)|(?-i:n\.p\.)"),
    ("gap", "no_data", rf"no{_WS}data"),
    ("gap", "not_collected", rf"not{_WS}collected"),
    ("gap", "not_applicable", rf"not{_WS}applicable"),
    ("gap", "suppressed", r"suppressed|confidentiali[sz]ed"),
    ("forecast", "forecast", r"forecast(?:s|ed)?"),
    ("forecast", "projected", r"project(?:ed|ions?)"),
    ("forecast", "expected", rf"expected{_WS}to"),
    ("forecast", "anticipated", r"anticipated"),
)


@dataclass(frozen=True, slots=True)
class Measure:
    id: str
    label: str
    unit: str | None


@dataclass(frozen=True, slots=True)
class Found:
    """A matched phrase: ``[start, end)`` of the text searched, and what it is."""

    start: int
    end: int
    kind: str      # a measure id, or a cue's canonical name
    cue_type: str | None = None


def _phrase(term: str) -> str:
    """*term* as a whole-word, one-line pattern; inner whitespace is flexible."""
    return rf"(?<!\w){_WS.join(re.escape(t) for t in term.split())}(?!\w)"


def _slug(term: str) -> str:
    return "_".join(re.findall(r"\w+", term.lower()))


def _without_overlaps(found: list[Found]) -> list[Found]:
    """Longest match first; a phrase inside a longer one is the longer one."""
    kept: list[Found] = []
    for f in sorted(found, key=lambda f: (f.start - f.end, f.start)):
        if not any(f.start < k.end and k.start < f.end for k in kept):
            kept.append(f)
    return sorted(kept, key=lambda f: f.start)


class Vocabulary:
    """Compiled measures and cues. Built once per stage run."""

    def __init__(self, declared: StatsVocabulary) -> None:
        self.measures = {
            m.id: Measure(m.id, m.label or m.id, m.unit) for m in declared.measures
        }
        self._measure_patterns = [
            (m.id, re.compile(_phrase(term), re.IGNORECASE))
            for m in declared.measures for term in m.terms if term.strip()
        ]
        cues = [(t, name, rf"(?<!\w)(?:{p})(?!\w)") for t, name, p in _BUILTIN_CUES]
        cues += [
            (cue_type, _slug(term), _phrase(term))
            for cue_type, terms in sorted(declared.cues.items())
            for term in terms if term.strip()
        ]
        self._cue_patterns = [(t, name, re.compile(p, re.IGNORECASE)) for t, name, p in cues]
        self.digest = vocabulary_digest(declared)

    def find_measures(self, text: str) -> list[Found]:
        return _without_overlaps([
            Found(m.start(), m.end(), measure_id)
            for measure_id, pattern in self._measure_patterns for m in pattern.finditer(text)
        ])

    def find_cues(self, text: str) -> list[Found]:
        found = []
        for cue_type, name, pattern in self._cue_patterns:
            for m in pattern.finditer(text):
                if name == "per_n":  # the base is the cue: per_100000, not per_n
                    name_here = "per_" + m["n"].replace(",", "")
                else:
                    name_here = name
                found.append(Found(m.start(), m.end(), name_here, cue_type))
        return _without_overlaps(found)


def load_vocabulary(config: StatsConfig) -> StatsVocabulary:
    """The config's own measures and cues, with its vocabulary file's merged in.

    Raises for a file that is missing, unreadable or not in the declared shape:
    callers run this before any batch, so a bad vocabulary refuses the stage.
    """
    measures = list(config.measures)
    cues = {k: list(v) for k, v in config.cues.items()}
    if config.vocabulary is not None:
        path = Path(config.vocabulary)
        if not path.is_file():
            raise FileNotFoundError(f"stats.vocabulary does not resolve on this host: {path}")
        with open(path, encoding="utf-8") as f:
            file_vocab = StatsVocabulary(**(yaml.safe_load(f) or {}))
        measures += file_vocab.measures
        for cue_type, terms in file_vocab.cues.items():
            cues.setdefault(cue_type, []).extend(terms)
    return StatsVocabulary(measures=measures, cues=cues)


def vocabulary_digest(declared: StatsVocabulary) -> str:
    """``sha256:…`` over the merged vocabulary, whatever file it came from."""
    canonical = json.dumps(declared.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


__all__ = ["Found", "Measure", "Vocabulary", "load_vocabulary", "vocabulary_digest"]
