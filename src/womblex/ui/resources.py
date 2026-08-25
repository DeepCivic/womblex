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
  actually calls, with no network request of its own. `misconfigured_endpoints_var()`
  names a near-miss of `ISAACUS_SAGEMAKER_ENDPOINTS` (e.g. the singular
  `…_ENDPOINT`) so a silent hosted-API fallback reads as a fixable cause
  rather than a bare "No API key".

Credentials never appear in a response body: the store card reports whether
AWS keys are configured, not their values, and the Isaacus card masks the
API key to its last four characters.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit, urlunsplit

from womblex.config import EmbeddingConfig, EnrichmentConfig
from womblex.store.remote import (
    assert_disjoint_locations,
    is_remote_uri,
    storage_options_from_env,
    validate_location_uri,
)
from womblex.ui import dashboard
from womblex.ui.deps import UISettings, apply_saved_locations
from womblex.ui.settings_store import SavedLocations, read_saved_locations, write_saved_locations
from womblex.utils.isaacus_client import (
    API_KEY_ENV,
    endpoints_from_env,
    misconfigured_endpoints_var,
    unserved_models,
)

logger = logging.getLogger(__name__)

#: The models the pipeline actually calls, read off the config fields that
#: name them rather than re-typed here — the same rule the composer's schema
#: endpoint follows ("no hand-typed mirror of `config.py`") and the dashboard's
#: `CHECKPOINT_DIRNAMES` follows for `STAGE_CONTRACTS`. These are the schema
#: *defaults*: the sidecar loads no config (a run's own config is not among the
#: artefacts it reads), so a deployment that overrides `enrichment.model` is
#: checking coverage for the stock model here. That is the honest scope of the
#: check, and the card names which models it checked.
ISAACUS_MODELS: tuple[str, ...] = tuple(
    dict.fromkeys(
        str(config.model_fields["model"].default)
        for config in (EmbeddingConfig, EnrichmentConfig)
    )
)

#: libpq's keyword/value DSN form (``host=… password=…``), which psycopg
#: accepts alongside the URI form. Quoted values are matched whole so a
#: password with a space in it does not leave its tail behind.
_KEYWORD_PASSWORD_RE = re.compile(r"(password\s*=\s*)('[^']*'|\S+)", re.IGNORECASE)


def _mask_secret(value: str | None) -> str | None:
    """Enough of *value* to recognise it, never enough to reuse it.

    A value too short to have a non-revealing tail is masked whole rather
    than shown — the tail convention exists to aid recognition, not to make
    an exception for the secrets it would print in full.
    """
    if not value:
        return None
    if len(value) <= 4:
        return "•" * 8
    return f"{'•' * 8}{value[-4:]}"


def _mask_dsn(dsn: str | None) -> str | None:
    """*dsn* with its password blanked; everything else (host, db) is not a secret.

    Covers both DSN forms psycopg accepts, because ``JobQueue`` passes
    whatever the operator set straight through: the URI form
    (``postgresql://user:pw@host/db``), whose password ``urlsplit`` finds,  # pragma: allowlist secret -- docstring example, not a real credential
    and libpq's keyword/value form (``host=… password=…``), which has no
    netloc at all and so would otherwise be returned verbatim — a full
    credential leak from an endpoint with no auth in front of it (plan §6).
    """
    if not dsn:
        return None
    try:
        parts = urlsplit(dsn)
    except ValueError:
        return "***"
    if parts.password:
        creds, _, host = parts.netloc.rpartition("@")
        user = creds.split(":", 1)[0]
        dsn = urlunsplit(
            (parts.scheme, f"{user}:***@{host}", parts.path, parts.query, parts.fragment)
        )
    # Runs over the whole string, so it also catches a password handed in as a
    # URI query parameter (`?password=…`), which is not `parts.password` either.
    return _KEYWORD_PASSWORD_RE.sub(r"\1***", dsn)


def _store_options_summary(uri: str, credentials: tuple[str, str] | None = None) -> dict:
    """AWS options `storage_options_from_env` would pass — presence, not values.

    *credentials*, when given, is the operator-saved S3 override; it is folded
    in the same way :func:`storage_options_from_env` folds it, so the card
    reports ``credentials_configured`` true for a console whose keys come from
    the saved override rather than the env. The values themselves never leave
    this function — only whether a key/secret pair is present, where it came
    from (``"saved"`` when the override supplied it, else ``"env"``), and the
    access-key id masked to its last four characters for recognition.
    """
    opts = storage_options_from_env(uri, credentials=credentials)
    client_kwargs = opts.get("client_kwargs", {})
    configured = bool(opts.get("key") and opts.get("secret"))
    return {
        "credentials_configured": configured,
        "credentials_source": (
            "saved" if (credentials is not None and configured) else ("env" if configured else None)
        ),
        "credentials_masked": _mask_secret(opts.get("key")) if configured else None,
        "endpoint_url": client_kwargs.get("endpoint_url"),
        "region": client_kwargs.get("region_name"),
    }


def _location_source(configured_uri: str | None, saved_uri: str | None, env_uri: str | None) -> str:
    """Where an effective location came from: saved override, env, or flag.

    Checked highest-precedence first (flag < env < saved), so a saved value
    matching the env default still reports ``"saved"`` — that is the source
    an edit would update.
    """
    if configured_uri is None:
        return "flag"
    if saved_uri is not None and saved_uri == configured_uri:
        return "saved"
    if env_uri is not None and env_uri == configured_uri:
        return "env"
    return "flag"


def _ingest_source(settings: UISettings) -> str:
    saved = read_saved_locations(settings.settings_dir).ingest_uri if settings.settings_dir else None
    return _location_source(settings.ingest_uri, saved, os.environ.get("WOMBLEX_INGEST_URI"))


def _store_source(settings: UISettings) -> str:
    """Where the effective output location came from.

    ``output_root`` mode (the legacy local read-only tree) has no override
    path at all — it can only come from ``--output-root`` /
    ``$WOMBLEX_UI_OUTPUT_ROOT`` — so it is always reported as ``"flag"``.
    """
    if not settings.is_remote:
        return "flag"
    saved = read_saved_locations(settings.settings_dir).store_uri if settings.settings_dir else None
    return _location_source(settings.store_uri, saved, os.environ.get("WOMBLEX_STORE_URI"))


def get_store_card(settings: UISettings) -> dict:
    """Where runs are read from, and how (docs/ui-plan.md §3).

    ``editable`` is just "this deployment has a writable settings dir" — true
    in both branches, since saving a location is what switches a local
    deployment into ``store_uri`` mode.
    """
    editable = settings.settings_writable
    if settings.is_remote:
        uri = cast(str, settings.store_uri)
        return {
            "kind": "remote",
            "uri": uri,
            "is_object_store": is_remote_uri(uri),
            "options": _store_options_summary(uri, settings.s3_credentials),
            "source": _store_source(settings),
            "editable": editable,
        }
    return {
        "kind": "local",
        "uri": str(cast(Path, settings.output_root)),
        "is_object_store": False,
        "options": {},
        "source": "flag",
        "editable": editable,
    }


def get_queue_card(settings: UISettings) -> dict:
    """Whether a job queue is configured, credential-masked."""
    return {"configured": bool(settings.db_dsn), "dsn_masked": _mask_dsn(settings.db_dsn)}


def get_isaacus_card() -> dict:
    """Deployment shape and model coverage — no network request of its own.

    ``endpoints_typo`` names a near-miss of ``ISAACUS_SAGEMAKER_ENDPOINTS``
    (e.g. the singular ``…_ENDPOINT``) when one is set but the canonical var
    is not: the deployment silently falls back to the hosted API, so the card
    would otherwise read as a bare "No API key" with no hint at the cause.
    """
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
        "endpoints_typo": misconfigured_endpoints_var() if not endpoints else None,
    }


def get_ingest_card(settings: UISettings) -> dict:
    """Where source documents are read from, if configured.

    Unlike the store card, ingest is optional, so ``configured`` is the
    first thing the card reports rather than assuming one of two shapes.
    """
    editable = settings.settings_writable
    if not settings.ingest_uri:
        return {
            "configured": False, "uri": None, "is_object_store": False, "options": {},
            "source": "flag", "editable": editable,
        }
    uri = settings.ingest_uri
    return {
        "configured": True,
        "uri": uri,
        "is_object_store": is_remote_uri(uri),
        "options": _store_options_summary(uri, settings.s3_credentials),
        "source": _ingest_source(settings),
        "editable": editable,
    }


def get_resources(settings: UISettings) -> dict:
    """The four connection cards, cheap enough to read on every page load."""
    return {
        "store": get_store_card(settings),
        "ingest": get_ingest_card(settings),
        "queue": get_queue_card(settings),
        "isaacus": get_isaacus_card(),
    }


def test_store(settings: UISettings) -> dict:
    """Live reachability check behind the store card's "Test" action.

    The two deployments answer deliberately different questions, and a
    nonexistent path reads differently under each. Locally, a missing
    ``output_root`` is the classic misconfiguration — a bind mount that did
    not land — so it is a failure worth naming. Remotely, the check is that
    the listing call *completed*: an object store has no real directories,
    so a valid but still-empty bucket has no ``runs/`` prefix, and demanding
    one would report every store unreachable until its first run finished.
    A store that genuinely cannot be reached still fails here, because the
    connection error propagates out of the listing rather than being
    flattened into an empty result.
    """
    if not settings.is_remote:
        root = cast(Path, settings.output_root)
        ok = root.is_dir()
        return {"reachable": ok, "error": None if ok else f"{root} is not a directory"}
    uri = cast(str, settings.store_uri)
    try:
        from womblex.store.remote import RemoteStore

        RemoteStore.from_uri(uri, credentials=settings.s3_credentials).list_dirs("runs")
    except Exception as e:
        logger.warning("resources: store unreachable: %s", e)
        return {"reachable": False, "error": str(e)}
    return {"reachable": True, "error": None}


def test_ingest(settings: UISettings) -> dict:
    """Live reachability check behind the ingest card's "Test" action.

    Same shape as :func:`test_store`: a listing call that completes — even
    against an empty prefix — counts as reachable.
    """
    if not settings.ingest_uri:
        return {"reachable": False, "error": "no ingest location configured"}
    uri = settings.ingest_uri
    try:
        from womblex.store.remote import RemoteStore

        RemoteStore.from_uri(uri, credentials=settings.s3_credentials).list_files("", "*")
    except Exception as e:
        logger.warning("resources: ingest unreachable: %s", e)
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


def save_locations(
    base: UISettings,
    *,
    ingest_uri: str | None,
    store_uri: str | None,
    s3_access_key_id: str | None = None,
    s3_secret_access_key: str | None = None,
    clear_credentials: bool = False,
) -> dict:
    """Persist an ingest/output override and return the refreshed cards.

    *base* is the pre-overlay settings (:func:`~womblex.ui.deps.get_base_settings`),
    so a cleared field (``None``) falls back to the flag/env default rather
    than to whatever was previously saved. A full replace (``PUT``), not a
    merge — a caller keeping one field must resubmit its current value.

    The S3 credential pair is the one exception to "full replace": the console
    masks the saved secret in every response, so the frontend cannot resubmit
    a secret it can no longer read. A save that omits both credential fields
    therefore *keeps* whatever was saved (preserve-on-omit); passing both sets
    a new pair; ``clear_credentials`` removes it and reverts to the env keys.
    A half-set pair (one field only) is refused rather than half-stored.

    Raises ``ValueError`` (→ 400) on a location a store cannot open, on a
    pair that would overlap once effective, or on a half-set credential pair.
    Reachability is *reported*, not required: naming a bucket ahead of
    provisioning it is normal.
    """
    for value in (ingest_uri, store_uri):
        if value is not None:
            validate_location_uri(value)

    effective_ingest = ingest_uri or base.ingest_uri
    effective_store = store_uri or base.store_uri
    if effective_ingest and effective_store:
        assert_disjoint_locations(effective_ingest, effective_store)
    elif effective_ingest and base.output_root is not None:
        # Legacy output_root mode has no `runs/` prefix of its own, so the
        # tree itself is the output — an ingest inside it still overlaps.
        assert_disjoint_locations(effective_ingest, str(base.output_root), runs_prefix="")

    settings_dir = cast(Path, base.settings_dir)
    key, secret = _resolve_saved_credentials(
        settings_dir,
        s3_access_key_id=s3_access_key_id,
        s3_secret_access_key=s3_secret_access_key,
        clear_credentials=clear_credentials,
    )
    saved = SavedLocations(
        ingest_uri=ingest_uri, store_uri=store_uri,
        s3_access_key_id=key, s3_secret_access_key=secret,
    )
    write_saved_locations(settings_dir, saved)

    refreshed = apply_saved_locations(base, saved)
    return {
        "ingest": get_ingest_card(refreshed),
        "store": get_store_card(refreshed),
        "ingest_test": test_ingest(refreshed),
        "store_test": test_store(refreshed),
    }


def _resolve_saved_credentials(
    settings_dir: Path,
    *,
    s3_access_key_id: str | None,
    s3_secret_access_key: str | None,
    clear_credentials: bool,
) -> tuple[str | None, str | None]:
    """The ``(key, secret)`` to persist, given the request and what was saved.

    ``clear_credentials`` wins — both fields drop to ``None`` and the store
    reverts to the env keys. A submitted pair (both fields) replaces the saved
    one. Neither given keeps the saved pair intact (preserve-on-omit), because
    the masked response the operator edited never carried the real secret back.
    A half-set pair is a caller error, not a silent half-store.
    """
    if clear_credentials:
        return None, None
    if s3_access_key_id and s3_secret_access_key:
        return s3_access_key_id, s3_secret_access_key
    if s3_access_key_id or s3_secret_access_key:
        raise ValueError(
            "both s3_access_key_id and s3_secret_access_key are required to set a "
            "credential override (pass neither to keep the saved pair, or "
            "clear_credentials to remove it)"
        )
    previous = read_saved_locations(settings_dir)
    return previous.s3_access_key_id, previous.s3_secret_access_key
