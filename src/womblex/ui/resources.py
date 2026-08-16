"""The Resources Console's read model (docs/ui-plan.md merge 10).

Three connection cards, none backed by new detection logic — the plan's §3
row for this screen says connection *testing* already exists as library
code, so this module's job is only to call it and shape the answer for a
screen:

- **Run store** — `store/remote.is_remote_uri` / `storage_options_from_env`
  describe how the console is configured to reach state; the live
  reachability check is the same `RemoteStore.list_dirs` read
  `womblex.ui.dashboard` already performs to find checkpoints.
- **Job queue** — `womblex.ui.dashboard.queue_section` is reused rather than
  reimplemented: it is already "connect, read stats/workers/stale jobs,
  report a failure instead of raising", which is exactly what the queue
  card's fleet + queue-depth state needs.
- **Isaacus** — `utils.isaacus_client.unserved_models()` reports whether the
  configured deployment (hosted API or SageMaker) covers the models Womblex
  actually calls, with no network request of its own.

Credentials never appear in a response body: the store card reports whether
AWS keys are configured, not their values, and the Isaacus card masks the
API key to its last four characters.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit, urlunsplit

from womblex.store.remote import is_remote_uri, storage_options_from_env
from womblex.ui import dashboard
from womblex.ui.deps import UISettings
from womblex.utils.isaacus_client import (
    API_KEY_ENV,
    endpoints_from_env,
    unserved_models,
)

logger = logging.getLogger(__name__)

#: Models the pipeline actually calls today (embed + enrich stages), so the
#: Isaacus card's "unserved" check means something concrete rather than an
#: arbitrary probe list.
ISAACUS_MODELS = ("kanon-2-embedder", "kanon-2-enricher")


def _mask_secret(value: str | None) -> str | None:
    """Enough of *value* to recognise it, never enough to reuse it."""
    if not value:
        return None
    tail = value[-4:] if len(value) > 4 else value
    return f"{'•' * 8}{tail}"


def _mask_dsn(dsn: str | None) -> str | None:
    """*dsn* with its password blanked; everything else (host, db) is not a secret."""
    if not dsn:
        return None
    try:
        parts = urlsplit(dsn)
    except ValueError:
        return "***"
    if not parts.password:
        return dsn
    creds, _, host = parts.netloc.rpartition("@")
    user = creds.split(":", 1)[0]
    netloc = f"{user}:***@{host}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _store_options_summary(uri: str) -> dict:
    """AWS options `storage_options_from_env` would pass — presence, not values."""
    opts = storage_options_from_env(uri)
    client_kwargs = opts.get("client_kwargs", {})
    return {
        "credentials_configured": bool(opts.get("key") and opts.get("secret")),
        "endpoint_url": client_kwargs.get("endpoint_url"),
        "region": client_kwargs.get("region_name"),
    }


def get_store_card(settings: UISettings) -> dict:
    """Where runs are read from, and how (docs/ui-plan.md §3)."""
    if settings.is_remote:
        uri = cast(str, settings.store_uri)
        return {
            "kind": "remote",
            "uri": uri,
            "is_object_store": is_remote_uri(uri),
            "options": _store_options_summary(uri),
        }
    return {
        "kind": "local",
        "uri": str(cast(Path, settings.output_root)),
        "is_object_store": False,
        "options": {},
    }


def get_queue_card(settings: UISettings) -> dict:
    """Whether a job queue is configured, credential-masked."""
    return {"configured": bool(settings.db_dsn), "dsn_masked": _mask_dsn(settings.db_dsn)}


def get_isaacus_card() -> dict:
    """Deployment shape and model coverage — no network request of its own."""
    endpoints = endpoints_from_env()
    api_key = os.environ.get(API_KEY_ENV)
    return {
        "deployment": "sagemaker" if endpoints else "hosted",
        "endpoints": [
            {"name": e.name, "region": e.region, "models": list(e.models) if e.models else None}
            for e in endpoints
        ],
        "api_key_configured": bool(api_key),
        "api_key_masked": _mask_secret(api_key),
        "models_checked": list(ISAACUS_MODELS),
        "unserved_models": unserved_models(ISAACUS_MODELS) if endpoints else [],
    }


def get_resources(settings: UISettings) -> dict:
    """The three connection cards, cheap enough to read on every page load."""
    return {
        "store": get_store_card(settings),
        "queue": get_queue_card(settings),
        "isaacus": get_isaacus_card(),
    }


def test_store(settings: UISettings) -> dict:
    """Live reachability check behind the store card's "Test" action."""
    if not settings.is_remote:
        root = cast(Path, settings.output_root)
        ok = root.is_dir()
        return {"reachable": ok, "error": None if ok else f"{root} is not a directory"}
    uri = cast(str, settings.store_uri)
    try:
        from womblex.store.remote import RemoteStore

        RemoteStore.from_uri(uri).list_dirs("runs")
    except Exception as e:
        logger.warning("resources: store unreachable: %s", e)
        return {"reachable": False, "error": str(e)}
    return {"reachable": True, "error": None}


def test_queue(
    settings: UISettings,
    *,
    stale_after: float = dashboard.DEFAULT_STALE_AFTER,
    window_seconds: float = dashboard.DEFAULT_THROUGHPUT_WINDOW,
    job_limit: int = 200,
) -> dict:
    """Fleet + queue-depth state — `dashboard.queue_section` *is* the connectivity test."""
    if not settings.db_dsn:
        return {"reachable": False, "error": "no queue configured", "queue": None}
    queue, error = dashboard.queue_section(
        settings, None, stale_after=stale_after, window_seconds=window_seconds,
        job_limit=job_limit,
    )
    return {"reachable": queue is not None, "error": error, "queue": queue}
