"""The record of one delivery to a workflow destination: ``<run>/deliveries/<destination>.json``.

Written after each delivery, delivered or failed, beside the run's ``documents/`` and
``manifest.parquet``, and read by the run record (`store/run_manifest.py`), which lists every
delivery. ``<destination>.json`` is the latest; a redelivery keeps the record it replaces as
``<destination>.<seq>.json`` and writes the next ``seq``. Each record carries its ``seq``, the
``seq`` it ``replaces`` and, from the manifest at the time, each document's ``content_digest``
(``document_digests``) with the documents whose digest moved since the replaced delivery
(``changed_documents``). An earlier record is never rewritten (a consumer's acknowledgement
appends to the record it names). Which record replaced it is derived on read. Self-contained,
like the other ``store/*_output.py`` modules.
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


def earlier_key(destination: str, seq: int) -> str:
    """The run-relative key an earlier delivery's record is kept under."""
    return f"{DELIVERIES_DIRNAME}/{destination}.{seq}.json"


def seq_of(record: dict[str, Any]) -> int:
    """A record's sequence number; one written before sequences is the first delivery."""
    return int(record.get("seq") or 1)


def next_record(
    outcome: dict[str, Any], previous: dict[str, Any] | None, digests: dict[str, str] | None,
) -> dict[str, Any]:
    """*outcome* as the record of the delivery after *previous* (`None`: the first).

    *digests* maps each document's ``source_hash`` to its ``content_digest`` in the manifest
    delivered from (`None` where it could not be read). ``changed_documents`` is `None` where there
    is nothing to compare: a first delivery, or a previous one that recorded no digests.
    """
    before = previous.get("document_digests") if previous is not None else None
    changed = None if before is None or digests is None else sorted(
        h for h, d in digests.items() if before.get(h) != d)
    return {**outcome, "seq": seq_of(previous) + 1 if previous is not None else 1,
            "replaces": seq_of(previous) if previous is not None else None,
            "document_digests": digests, "changed_documents": changed}


def run_record_view(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The records as the run record lists them: by destination then sequence, each saying which
    delivery replaced it, with the changed documents reduced to a count and the per-document digests left out."""
    prepared = [{**r, "seq": seq_of(r), "replaces": r.get("replaces")} for r in records]
    latest: dict[str, int] = {}
    for r in prepared:
        dest = r.get("destination", "")
        latest[dest] = max(latest.get(dest, 0), r["seq"])
    out = []
    for r in sorted(prepared, key=lambda r: (str(r.get("destination", "")), r["seq"])):
        dest = r.get("destination", "")
        later = [x["seq"] for x in prepared if x.get("destination", "") == dest and x["seq"] > r["seq"]]
        r.pop("document_digests", None)
        changed = r.pop("changed_documents", None)
        r["replaced_by"] = min(later) if later else None
        r["documents_changed"] = None if changed is None else len(changed)
        out.append(r)
    return out


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
    return run_record_view(out)


def read_delivery_records(run_root: Path) -> list[dict[str, Any]]:
    """The delivery records under a local *run_root*, by destination name; none is an empty list."""
    directory = run_root / DELIVERIES_DIRNAME
    if not directory.is_dir():
        return []
    return parse_deliveries([(p.name, p.read_text(encoding="utf-8")) for p in directory.glob("*.json")])


__all__ = [
    "DELIVERIES_DIRNAME", "delivery_key", "dump_delivery", "earlier_key", "next_record", "parse_deliveries",
    "read_delivery_records", "run_record_view", "seq_of",
]
