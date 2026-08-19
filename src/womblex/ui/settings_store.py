"""Operator-saved ingest/output location override (docs/ui-ingest-plan.md merge 3a).

Env and compose values are *defaults*: the Resources Console can add or
update a location without a restart, persisted to one ``locations.json`` in a
writable settings dir (``--settings-dir`` / ``$WOMBLEX_UI_SETTINGS_DIR``). It
cannot live in the output store, because it is the file that *names* the
output store.

Same self-contained shape as ``ui/presets.py``'s file helpers: validate on
write, tolerate a missing or corrupt file on read. ``ui/deps.py`` owns
applying the parsed override; this module owns only the file's shape.
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
    that location has no override — it resolves to its flag/env default.

    ``s3_access_key_id`` / ``s3_secret_access_key`` are an optional S3
    credential override for the store. When set, they take precedence over the
    ``WOMBLEX_S3_*`` / ``AWS_*`` env vars the Dockerfile baked in — the case an
    operator rotates a key through the Resources Console and the process must
    use the new one *moving forward*, without a container rebuild. Both must be
    present to take effect (a half-set pair is ignored, the same rule
    :func:`~womblex.store.remote.storage_options_from_env` applies to the env).
    Stored on disk in the settings volume, so treat that volume as sensitive.
    """

    ingest_uri: str | None = None
    store_uri: str | None = None
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Only the keys that are actually set — an absent key round-trips
        as "no override", not a stored ``null``."""
        record: dict[str, Any] = {}
        if self.ingest_uri is not None:
            record["ingest_uri"] = self.ingest_uri
        if self.store_uri is not None:
            record["store_uri"] = self.store_uri
        if self.s3_access_key_id is not None:
            record["s3_access_key_id"] = self.s3_access_key_id
        if self.s3_secret_access_key is not None:
            record["s3_secret_access_key"] = self.s3_secret_access_key
        return record

    def s3_credentials(self) -> tuple[str, str] | None:
        """The saved ``(key, secret)`` pair, or ``None`` unless both are set."""
        if self.s3_access_key_id and self.s3_secret_access_key:
            return self.s3_access_key_id, self.s3_secret_access_key
        return None


def locations_path(settings_dir: Path) -> Path:
    """The one file a saved override lives in, under *settings_dir*."""
    return settings_dir / LOCATIONS_FILENAME


def read_saved_locations(settings_dir: Path) -> SavedLocations:
    """The saved override, or an empty one if the file is absent or will not parse.

    Skip-and-continue, not fatal: a corrupt or hand-edited file must not take
    the console's settings resolution down. Only the two known keys are read;
    anything else is ignored, so a forward-compatible extra key is not a fault.
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
    key = raw.get("s3_access_key_id")
    secret = raw.get("s3_secret_access_key")
    return SavedLocations(
        ingest_uri=ingest if isinstance(ingest, str) else None,
        store_uri=store if isinstance(store, str) else None,
        s3_access_key_id=key if isinstance(key, str) else None,
        s3_secret_access_key=secret if isinstance(secret, str) else None,
    )


def write_saved_locations(settings_dir: Path, locations: SavedLocations) -> None:
    """Persist *locations* as the whole override — a full replace, matching
    ``PUT /api/resources/locations``, not a merge with what was there before."""
    settings_dir.mkdir(parents=True, exist_ok=True)
    body = json.dumps(locations.as_dict(), indent=2)
    locations_path(settings_dir).write_text(body, encoding="utf-8")
