"""The record of one delivery to a workflow destination: ``<run>/deliveries/<destination>.json``.

Written after each delivery, delivered or failed, beside the run's ``documents/`` and
``manifest.parquet``, and read by the run record (`store/run_manifest.py`), which lists every
delivery. One file per destination: a redelivery replaces it, as it replaces the bundle.
Self-contained, like the other ``store/*_output.py`` modules.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DELIVERIES_DIRNAME = "deliveries"


def delivery_key(destination: str) -> str:
    """The run-relative key of *destination*'s delivery record."""
    return f"{DELIVERIES_DIRNAME}/{destination}.json"


def dump_delivery(record: dict[str, Any]) -> str:
    return json.dumps(record, indent=2, sort_keys=True) + "\n"


def parse_deliveries(texts: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """Each (name, text) pair as a record, by destination name; one that is not a JSON object is
    skipped with a warning, so an unreadable record is missing from the list, not fatal to it."""
    out: list[dict[str, Any]] = []
    for name, text in sorted(texts):
        try:
            record = json.loads(text)
        except json.JSONDecodeError:
            record = None
        if isinstance(record, dict):
            out.append(record)
        else:
            logger.warning("delivery record %s is not a JSON object; left out of the run record", name)
    return out


def read_delivery_records(run_root: Path) -> list[dict[str, Any]]:
    """The delivery records under a local *run_root*, by destination name; none is an empty list."""
    directory = run_root / DELIVERIES_DIRNAME
    if not directory.is_dir():
        return []
    return parse_deliveries([(p.name, p.read_text(encoding="utf-8")) for p in directory.glob("*.json")])


__all__ = ["DELIVERIES_DIRNAME", "delivery_key", "dump_delivery", "parse_deliveries", "read_delivery_records"]
