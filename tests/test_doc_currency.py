"""Drift between the docs and the code fails in the PR that causes it (D-2.5).

The README command table is held to the CLI parser, and CLAUDE.md's module table to
the `src/` tree. Run by CI with the rest of the suite; nothing here is a benchmark.
"""

from __future__ import annotations

import re
from pathlib import Path

from womblex.cli import ALL_COMMANDS

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "womblex"

#: Modules the table has never listed. A new module is not added here: it gets a row.
UNLISTED = {
    "config/process.py", "ingest/grid_projection.py", "ingest/heuristics_cv2.py",
    "ingest/heuristics_numpy.py", "ingest/interfaces/protocols.py",
    "ingest/markdown.py", "ingest/redaction.py", "ingest/spreadsheet.py", "ingest/strategies.py",
    "ingest/views.py", "process/quality.py", "process/quality_stage.py", "profile/columns.py",
    "redact/batch.py", "redact/utils.py", "score.py", "store/checkpoint.py", "store/demo_corpus.py",
    "store/enrichment_output.py", "store/ground_truth_output.py", "store/quality_output.py",
    "store/retention.py", "store/shard_audit.py", "utils/availability.py", "utils/checksum.py",
    "utils/money_metrics.py", "utils/run_log.py",
}
#: Covered elsewhere: package markers, and the CLI (the README command table).
EXEMPT_PREFIXES = ("cli/",)


def _readme_commands() -> tuple[set[str], str]:
    """The names in the command table's first column, and the table's text."""
    text = (ROOT / "README.md").read_text()
    table = text.split("### Commands", 1)[1].split("\n## ", 1)[0]
    first = [row.split("|")[1] for row in table.splitlines() if row.startswith("| `")]
    return {name for cell in first for name in re.findall(r"`([^`]+)`", cell)}, table


def test_the_readme_command_table_matches_the_parser() -> None:
    listed, table = _readme_commands()
    parser = {c.name for c in ALL_COMMANDS}
    mentioned = set(re.findall(r"`([^`\s]+)`", table))  # a deprecated alias is named in a description
    assert parser - mentioned == set(), "commands the README table does not list"
    assert listed - parser == set(), "README table lists commands the parser lacks"
    assert len(ALL_COMMANDS) == len(parser), "a command name is registered twice"
    assert table  # the table was found


def _module_table() -> tuple[set[str], set[str]]:
    """The paths CLAUDE.md's module table names, and the .py files they cover under src/womblex."""
    rows = [r for r in (ROOT / "CLAUDE.md").read_text().splitlines() if r.startswith("| `")]
    named = {t for r in rows for t in re.findall(r"`([^`]+)`", r.split("|")[1])}
    covered: set[str] = set()
    for t in named:
        if "*" in t:
            covered |= {str(p.relative_to(SRC)) for p in SRC.glob(t)}
        elif t.endswith("/"):
            covered |= {str(p.relative_to(SRC)) for p in (SRC / t).rglob("*.py")}
        else:
            covered.add(t)
    return named, covered


def test_every_module_the_claude_md_table_names_exists() -> None:
    named, _ = _module_table()
    gone = sorted(t for t in named if not (list(SRC.glob(t)) if "*" in t else (SRC / t).exists()))
    assert gone == [], f"CLAUDE.md names modules that are not in src/womblex: {gone}"


def test_every_module_in_src_is_in_the_table_or_a_known_gap() -> None:
    _, covered = _module_table()
    actual = {str(p.relative_to(SRC)) for p in SRC.rglob("*.py")
              if p.name not in {"__init__.py", "__main__.py"}
              and not str(p.relative_to(SRC)).startswith(EXEMPT_PREFIXES)}
    assert sorted(actual - covered - UNLISTED) == [], "modules with no CLAUDE.md row: add one"
    assert sorted(UNLISTED & covered) == [], "listed now: remove from UNLISTED"
    assert sorted(UNLISTED - actual) == [], "gone from src/: remove from UNLISTED"
