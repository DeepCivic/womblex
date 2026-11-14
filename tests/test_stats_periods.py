"""Tests for the reference-period grammar of the stats op."""

from __future__ import annotations

from datetime import date

import pytest

from womblex.process.stats_periods import find_periods


def _one(text: str, **kwargs):
    periods = find_periods(text, **kwargs)
    assert len(periods) == 1, [p.text for p in periods]
    return periods[0]


@pytest.mark.parametrize("text", [
    "2019–20", "2019-20", "2019/20", "FY20", "FY2020", "FY 2019-20",
    "financial year 2019-2020", "the 2019-20 financial year",
])
def test_every_financial_year_spelling_resolves_to_one_label(text: str):
    p = _one(text)
    assert (p.kind, p.label) == ("financial_year", "2019-20")
    assert (p.period_start, p.period_end) == (date(2019, 7, 1), date(2020, 6, 30))


def test_a_century_boundary_financial_year():
    p = _one("1999-00")
    assert (p.label, p.period_end) == ("1999-00", date(2000, 6, 30))


@pytest.mark.parametrize("text", [
    "2025-Q1", "Q1 2025", "2025 Q1", "March quarter 2025", "March 2025 quarter",
    "Mar qtr 2025", "March quarter of 2025",
])
def test_every_quarter_spelling_resolves_to_one_label(text: str):
    p = _one(text)
    assert (p.kind, p.label) == ("quarter", "2025-Q1")
    assert (p.period_start, p.period_end) == (date(2025, 1, 1), date(2025, 3, 31))


def test_a_quarter_without_its_year_has_no_bounds():
    p = _one("in the March quarter")
    assert (p.kind, p.label, p.period_start, p.period_end) == ("quarter", "Q1", None, None)


def test_a_quarter_is_named_by_its_last_month_only():
    assert find_periods("the April quarter") == []


@pytest.mark.parametrize("text", ["2026-H1", "H1 2026", "first half of 2026"])
def test_halves(text: str):
    p = _one(text)
    assert (p.kind, p.label) == ("half", "2026-H1")
    assert (p.period_start, p.period_end) == (date(2026, 1, 1), date(2026, 6, 30))


@pytest.mark.parametrize("text", ["CY2025", "CY 2025", "calendar year 2025"])
def test_calendar_years(text: str):
    p = _one(text)
    assert (p.kind, p.label) == ("calendar_year", "CY2025")
    assert (p.period_start, p.period_end) == (date(2025, 1, 1), date(2025, 12, 31))


def test_months_and_dates():
    month = _one("March 2025")
    assert (month.kind, month.label, month.period_end) == ("month", "2025-03", date(2025, 3, 31))
    iso = _one("2025-03")
    assert (iso.kind, iso.label) == ("month", "2025-03")
    day = _one("as at 30 June 2021")
    assert (day.kind, day.label, day.period_start) == ("date", "2021-06-30", date(2021, 6, 30))


def test_a_span_of_years_is_a_range_with_no_bounds():
    """`2019–21` may be calendar or financial years: the text does not say."""
    for text, label in (("2019–21", "2019-21"), ("2019–2020", "2019-2020")):
        p = _one(text)
        assert (p.kind, p.label, p.period_start, p.period_end) == ("year_range", label, None, None)


def test_a_year_needs_a_cue_in_running_text():
    assert _one("in 2021").text == "2021"
    assert _one("the 2021 Census").label == "2021"
    assert find_periods("there were 2025 people") == []
    assert _one("Dubbo 2021", bare_years=True).label == "2021"


@pytest.mark.parametrize("text", [
    "2,025 dwellings", "$2020", "ref 2019-05-01", "12.2021", "A2021",
    "2021%", "2021 per cent",
])
def test_numbers_that_are_not_periods(text: str):
    assert find_periods(text, bare_years=True) == []


def test_lowercase_may_is_a_verb_not_a_month():
    assert find_periods("it may 2025 be") == []
    assert _one("May 2025").label == "2025-05"


def test_periods_come_back_in_order_with_their_offsets():
    text = "Between 2019–20 and the June quarter 2024, and again in 2025."
    periods = find_periods(text)
    assert [p.label for p in periods] == ["2019-20", "2024-Q2", "2025"]
    assert all(text[p.start:p.end] == p.text for p in periods)


def test_a_period_never_spans_two_lines():
    assert [p.label for p in find_periods("March\nquarter 2025")] == []


@pytest.mark.parametrize(("text", "label"), [("FY99", "1998-99"), ("FY 69", "1968-69"), ("FY68", "2067-68")])
def test_a_two_digit_financial_year_takes_its_century_from_the_year(text: str, label: str):
    assert _one(text).label == label
