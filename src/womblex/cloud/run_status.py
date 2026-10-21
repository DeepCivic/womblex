"""A run's status as one sentence, built from exact per-stage counts.

The words are the API's (`jobs.run_state`: ``pending`` / ``running`` / ``done`` /
``failed`` / ``cancelled``). The CLI, the API and the console all print what
`run_sentence` returns, so the three never describe a run differently.
"""

from __future__ import annotations

from collections.abc import Iterable

from womblex.pipeline_order import sort_by_pipeline

#: Counts are listed in this order; a status with no jobs is left out.
_ORDER = ("done", "running", "pending", "failed", "cancelled")

#: The name extraction batches count under, beside the downstream stages.
EXTRACT = "extract"


def stage_counts(rows: Iterable[tuple[str | None, str | None, str]]) -> dict[str, dict[str, int]]:
    """Counts by status per stage from ``(kind, stage, status)`` rows, in pipeline order.

    Extraction batches count under ``extract``, stage workflows under their stage;
    coordinators and trigger firings are bookkeeping, not work, and are left out.
    """
    out: dict[str, dict[str, int]] = {}
    for kind, stage, status in rows:
        name = EXTRACT if kind == "batch" else stage if kind == "stage" else None
        if name:
            counts = out.setdefault(name, {})
            counts[status] = counts.get(status, 0) + 1
    return sort_by_pipeline(out)


def run_sentence(run_id: str, state: str, stages: dict[str, dict[str, int]], *, orphaned: bool = False) -> str:
    """One sentence: the run's state, each stage's counts, and a warning for an undispatched firing."""
    parts = [
        f"{stage}: " + ", ".join(f"{counts[s]} {s}" for s in _ORDER if counts.get(s))
        for stage, counts in stages.items() if any(counts.values())
    ]
    sentence = f"Run {run_id} is {state}"
    if parts:
        sentence += " (" + "; ".join(parts) + ")"
    sentence += "."
    if orphaned:
        sentence += " Warning: a scheduled firing could not dispatch its stages."
    return sentence
