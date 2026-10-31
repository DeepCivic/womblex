"""A run's status sentence: built from per-stage counts, one wording for every surface."""

from __future__ import annotations

import pytest

from womblex.cloud.run_status import run_sentence, stage_counts


def test_counts_group_by_stage_in_pipeline_order_and_skip_bookkeeping_rows():
    rows = [
        ("stage", "embed", "pending"), ("batch", None, "done"), ("batch", None, "failed"),
        ("stage", "chunk", "done"), ("downstream", None, "running"), ("trigger", None, "done"),
    ]
    assert stage_counts(rows) == {
        "extract": {"done": 1, "failed": 1}, "chunk": {"done": 1}, "embed": {"pending": 1},
    }
    assert list(stage_counts(rows)) == ["extract", "chunk", "embed"]


def test_the_sentence_names_each_stages_exact_counts():
    stages = {"extract": {"done": 4}, "chunk": {"done": 1, "pending": 2, "failed": 1}}
    assert run_sentence("run-a", "failed", stages) == (
        "Run run-a is failed (extract: 4 done; chunk: 1 done, 2 pending, 1 failed)."
    )


def test_a_skipped_stage_reads_skipped_after_done():
    stages = {"extract": {"done": 2}, "embed": {"skipped": 1}, "chunk": {"pending": 1, "skipped": 1, "done": 1}}
    assert run_sentence("r", "running", stages) == (
        "Run r is running (extract: 2 done; embed: 1 skipped; chunk: 1 done, 1 skipped, 1 pending)."
    )


def test_a_run_with_no_work_and_an_orphaned_firing():
    assert run_sentence("run-a", "pending", {}) == "Run run-a is pending."
    assert run_sentence("run-a", "running", {"extract": {"running": 1}}, orphaned=True).endswith(
        "Warning: a scheduled firing could not dispatch its stages."
    )


@pytest.mark.parametrize("state", ["pending", "running", "done", "failed", "cancelled"])
def test_the_sentence_uses_the_apis_state_words(state):
    assert f"is {state}" in run_sentence("r", state, {})
