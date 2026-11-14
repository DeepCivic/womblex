"""Tests for the pure core of the stats op: vocabulary, cues and figures."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from womblex.config import StatsConfig, StatsVocabulary
from womblex.process.stats import find_stats, read_cell_figure
from womblex.process.stats_vocab import Vocabulary, load_vocabulary, vocabulary_digest

MEASURES = [
    {"id": "median_income", "label": "Median income",
     "terms": ["median income", "median weekly income"]},
    {"id": "population", "terms": ["population", "estimated resident population"],
     "unit": "persons"},
    {"id": "unemployment_rate", "terms": ["unemployment rate"]},
    {"id": "area_burnt", "terms": ["hectares"], "unit": "ha"},
]


@pytest.fixture
def vocab() -> Vocabulary:
    return Vocabulary(StatsVocabulary(measures=MEASURES))


def _kind(spans, kind):
    return [s for s in spans if s.kind == kind]


def test_a_stated_figure_carries_its_measure_unit_and_rate(vocab):
    text = "Median income $1,242/week"
    (figure,) = _kind(find_stats(text, vocab), "figure")
    assert text[figure.start:figure.end] == figure.text == "$1,242"
    assert (figure.measure.id, figure.value, figure.unit) == (
        "median_income", Decimal(1242), "AUD")
    assert (figure.rate_per, figure.status) == ("week", "stated")


def test_a_number_without_a_measure_is_not_a_figure(vocab):
    spans = find_stats("The pool cost $3 million and holds 300 people.", vocab)
    assert _kind(spans, "figure") == []


def test_no_vocabulary_means_no_figures_but_periods_and_cues_remain():
    spans = find_stats("Rents rose 10 per cent in 2021, in real terms.",
                       Vocabulary(StatsVocabulary()))
    assert _kind(spans, "figure") == [] and _kind(spans, "measure") == []
    assert [s.period.label for s in _kind(spans, "period")] == ["2021"]
    assert [s.cue for s in _kind(spans, "cue")] == ["real_terms"]


def test_each_figure_takes_the_nearest_period_in_its_sentence(vocab):
    text = "In 2021 the median income was $1,242 per week, up from $1,000 in 2016."
    first, second = _kind(find_stats(text, vocab), "figure")
    assert (first.period.label, first.period_source) == ("2021", "sentence")
    assert (second.period.label, second.period_source) == ("2016", "sentence")


def test_a_figure_without_a_period_inherits_the_headings(vocab):
    from womblex.process.stats_periods import find_periods

    (heading,) = find_periods("Dubbo, 2021 Census")
    (figure,) = _kind(find_stats("The population was 43,516.", vocab, inherited=heading), "figure")
    assert (figure.period.label, figure.period_source) == ("2021", "heading")
    assert figure.unit == "persons", "a bare number takes the measure's declared unit"

    (alone,) = _kind(find_stats("The population was 43,516.", vocab), "figure")
    assert alone.period is None and alone.period_source is None


def test_the_longest_term_wins(vocab):
    (measure,) = _kind(find_stats("The estimated resident population grew.", vocab), "measure")
    assert measure.text == "estimated resident population"


def test_cues_are_typed_and_attach_to_their_sentences_figures(vocab):
    text = ("The population is projected to reach 1.2 million by 2030, a share of 4.5% "
            "(ABS, cat. no. 3222.0). The unemployment rate was 4.1 per cent.")
    spans = find_stats(text, vocab)
    cues = {(s.cue_type, s.cue) for s in _kind(spans, "cue")}
    assert cues == {("forecast", "projected"), ("calculation", "share"),
                    ("source", "abs"), ("source", "catalogue_number")}
    million, share, rate = _kind(spans, "figure")
    assert (million.value, million.multiplier, million.status) == (
        Decimal("1200000.0"), "million", "forecast")
    assert (share.unit, share.calculation, share.source) == ("percent", "share", "ABS")
    assert (rate.measure.id, rate.status, rate.calculation) == (
        "unemployment_rate", "stated", None), "a cue stays in its own sentence"


def test_per_n_names_its_base_and_its_number_is_not_a_figure(vocab):
    spans = find_stats("The unemployment rate was 12.5 per 100,000 persons.", vocab)
    (figure,) = _kind(spans, "figure")
    assert (figure.value, figure.calculation) == (Decimal("12.5"), "per_100000")


def test_a_gap_cue_names_the_measure_it_sits_beside(vocab):
    (cue,) = _kind(find_stats("Population data are not available for 2019–20.", vocab), "cue")
    assert (cue.cue_type, cue.cue, cue.measure.id) == ("gap", "not_available", "population")


def test_a_bare_year_is_neither_a_figure_nor_a_period_in_running_text(vocab):
    spans = find_stats("Population 2021 was 43,516 and fell -2.5%.", vocab)
    assert _kind(spans, "period") == []
    assert [(f.text, f.value) for f in _kind(spans, "figure")] == [
        ("43,516", Decimal(43516)), ("-2.5%", Decimal("-2.5"))]


def test_a_source_line_is_one_cue_and_a_boundary_never_splits_a_phrase(vocab):
    text = "Population 43,516 (cat. no. 3218.0). Source: ABS Regional Population"
    sources = [s.text for s in _kind(find_stats(text, vocab), "cue")]
    assert sources == ["cat. no. 3218.0", "Source: ABS Regional Population"]


def test_corpus_cues_extend_the_builtin_set():
    vocab = Vocabulary(StatsVocabulary(cues={"gap": ["data pending"]}))
    (cue,) = _kind(find_stats("Data  pending for this region.", vocab), "cue")
    assert (cue.cue_type, cue.cue) == ("gap", "data_pending")


@pytest.mark.parametrize(("cell", "expected"), [
    ("$1,746", ("$1,746", Decimal(1746), "AUD", None)),
    ("4.5%", ("4.5%", Decimal("4.5"), "percent", None)),
    (" 43,516 ", ("43,516", Decimal(43516), None, None)),
    ("2021", ("2021", Decimal(2021), None, None)),
    ("$375/week", ("$375", Decimal(375), "AUD", "week")),
])
def test_a_cell_that_is_one_number_is_a_figure(cell, expected):
    span = read_cell_figure(cell)
    assert (span.text, span.value, span.unit, span.rate_per) == expected
    assert cell[span.start:span.end] == span.text


@pytest.mark.parametrize("cell", ["n.a.", "12 to 14", "about 40 people", "", "Dubbo"])
def test_any_other_cell_is_not(cell):
    assert read_cell_figure(cell) is None


def test_a_vocabulary_file_merges_with_the_config(tmp_path: Path):
    path = tmp_path / "vocab.yaml"
    path.write_text(
        "measures:\n  - id: median_rent\n    terms: [median rent]\n"
        "cues:\n  source: [regional insights]\n")
    config = StatsConfig(vocabulary=path, measures=MEASURES[:1], cues={"source": ["the register"]})
    merged = load_vocabulary(config)
    assert [m.id for m in merged.measures] == ["median_income", "median_rent"]
    assert merged.cues == {"source": ["the register", "regional insights"]}
    assert vocabulary_digest(merged) == vocabulary_digest(load_vocabulary(config))
    assert vocabulary_digest(merged) != vocabulary_digest(StatsVocabulary())


def test_a_bad_vocabulary_is_refused(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match=r"stats\.vocabulary"):
        load_vocabulary(StatsConfig(vocabulary=tmp_path / "absent.yaml"))
    bad = tmp_path / "bad.yaml"
    bad.write_text("measures:\n  - id: a\n    terms: []\n")
    with pytest.raises(ValidationError):
        load_vocabulary(StatsConfig(vocabulary=bad))
    with pytest.raises(ValidationError, match="more than once"):
        StatsVocabulary(measures=[MEASURES[0], MEASURES[0]])
    with pytest.raises(ValidationError, match="unknown cue type"):
        StatsVocabulary(cues={"mood": ["glum"]})


@pytest.mark.parametrize("text", [
    "4.21 Population data were collated across the region.",        # a paragraph number
    "The population target is a rolling average’.46 Queensland differs.",  # a footnote marker
    "Sources of population data include:96 the census.",
    "Population by region 442 Figure 86: Recovery",                   # a contents page
    "Population counts for the period 1 January to 31 March 2020.56 were revised.",
    "Population is counted in towns, with around 120 horses nearby.",  # a clause away
])
def test_a_unitless_number_must_sit_next_to_its_measure(vocab, text):
    assert _kind(find_stats(text, vocab), "figure") == []


@pytest.mark.parametrize(("text", "value"), [
    ("The population was 43,516.", "43516"),
    ("The population in 2021 was 43,516.", "43516"),
    ("Population: 43,516", "43516"),
    ("A population of 1.2 million is expected.", "1200000.0"),
    ("Some 43,516 people make up the population.", "43516"),
    ("More than 271,000 hectares were burnt.", "271000"),            # a measurement is a figure
])
def test_a_unitless_number_beside_its_measure_is_a_figure(vocab, text, value):
    (figure,) = _kind(find_stats(text, vocab), "figure")
    assert figure.value == Decimal(value)


def test_a_footnote_marker_does_not_unmake_a_dated_period(vocab):
    spans = find_stats("Costs were waived until 31 March 2020.56 Before then", vocab)
    assert [s.period.label for s in _kind(spans, "period")] == ["2020-03-31"]


@pytest.mark.parametrize(("text", "values"), [
    ("The unemployment rate moved –0.3% over the year.", ["-0.3"]),   # en dash: a minus
    ("The unemployment rate was 4–5% over the year.", ["5"]),         # between numbers: a range
    ("Median income changed by –12.5 in real terms.", ["-12.5"]),
])
def test_an_en_dash_that_opens_a_number_is_a_minus_sign(vocab, text, values):
    assert [f.value for f in _kind(find_stats(text, vocab), "figure")] == [Decimal(v) for v in values]
    assert read_cell_figure("–3.2").value == Decimal("-3.2")
