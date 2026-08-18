"""Operator-saved ingest/output location override (docs/ui-ingest-plan.md merge 3a).

Env and compose values are *defaults*, not the only source: the Resources
Console can save a location the deployment did not set, or update one it did,
without a restart. The override lives in one small JSON file —
``locations.json`` — in a writable settings dir the operator names
(``--settings-dir`` / ``$WOMBLEX_UI_SETTINGS_DIR``). It cannot live in the
output store, because it is the file that *names* the output store.

Same self-contained shape as ``ui/presets.py``'s save-parse-validate file
helpers: validate on write, tolerate a missing or corrupt file on read (a
hand-edited or partially-written file must not take the console down — it
degrades to "no override", not a crash), refuse anything but the two known
keys. ``ui/deps.py`` owns applying the parsed override onto ``UISettings``;
this module owns only the file's shape.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: The single file a saved override lives in, inside the settings dir.
LOCATIONS_FILENAME = "locations.json"


@dataclass(frozen=True)
class SavedLocations:
    """The operator-saved override. Either field absent (``None``) means
    that location has no override — it resolves to its flag/env default."""

    ingest_uri: str | None = None
    store_uri: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Only the keys that are actually set — an absent key round-trips
        as "no override", not a stored ``null``."""
        record: dict[str, Any] = {}
        if self.ingest_uri is not None:
            record["ingest_uri"] = self.ingest_uri
        if self.store_uri is not None:
            record["store_uri"] = self.store_uri
        return record


def locations_path(settings_dir: Path) -> Path:
    """The one file a saved override lives in, under *settings_dir*."""
    return settings_dir / LOCATIONS_FILENAME


def read_saved_locations(settings_dir: Path) -> SavedLocations:
    """The saved override, or an empty one if the file is absent or will not parse.

    Skip-and-continue, not fatal (like ``presets.parse_saved_preset``): a
    corrupt or hand-edited file must not take the console's settings
    resolution down. Only the two known keys are read; anything else in the
    file is ignored rather than rejected, so a forward-compatible extra key
    does not itself become a fault here.
    """
    path = locations_path(settings_dir)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return SavedLocations()
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("settings_store: %s unreadable, ignoring saved locations: %s", path, e)
        return SavedLocations()
    if not isinstance(raw, dict):
        logger.warning("settings_store: %s is not a JSON object, ignoring", path)
        return SavedLocations()
    ingest = raw.get("ingest_uri")
    store = raw.get("store_uri")
    return SavedLocations(
        ingest_uri=ingest if isinstance(ingest, str) else None,
        store_uri=store if isinstance(store, str) else None,
    )


def write_saved_locations(settings_dir: Path, locations: SavedLocations) -> None:
    """Persist *locations* as the whole override — a full replace, matching
    ``PUT /api/resources/locations``, not a merge with what was there before."""
    settings_dir.mkdir(parents=True, exist_ok=True)
    body = json.dumps(locations.as_dict(), indent=2)
    locations_path(settings_dir).write_text(body, encoding="utf-8")
