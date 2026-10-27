"""What a deployment is wired to: its run store, ingest location, job queue and Isaacus deployment.

A description, read from settings and the environment with no network call,
plus the live checks behind it (`check_store`, `check_ingest`; the queue's is
`reachability.queue_reachable`). Credentials never leave here: AWS keys are
reported as configured or not with the access-key id masked to its last four
characters, a DSN's password is blanked in both libpq forms, and the Isaacus
API key is masked like the access-key id.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit, urlunsplit

from womblex.cloud.reachability import store_reachable
from womblex.config import EmbeddingConfig, EnrichmentConfig
from womblex.providers.isaacus.client import API_KEY_ENV, endpoints_from_env, misconfigured_endpoints_var
from womblex.providers.isaacus.deployment import SAGEMAKER
from womblex.store.remote import is_remote_uri, storage_options_from_env
from womblex.utils.deployment_registry import resolve

if TYPE_CHECKING:
    from womblex.cloud.deployment import DeploymentSettings

logger = logging.getLogger(__name__)

#: The models the pipeline calls, at the config's schema defaults: a deployment
#: carries no run's config, so this is the scope of the coverage check, and the
#: description names the models it checked.
ISAACUS_MODELS: tuple[str, ...] = tuple(dict.fromkeys(
    str(config.model_fields["model"].default) for config in (EmbeddingConfig, EnrichmentConfig)
))

#: libpq's keyword/value form (``host=… password=…``); a quoted value is matched whole.
_KEYWORD_PASSWORD = re.compile(r"(password\s*=\s*)('[^']*'|\S+)", re.IGNORECASE)


def mask_secret(value: str | None) -> str | None:
    """The last four characters behind a mask; a value too short for that is masked whole."""
    if not value:
        return None
    return "•" * 8 if len(value) <= 4 else f"{'•' * 8}{value[-4:]}"


def mask_dsn(dsn: str | None) -> str | None:
    """*dsn* with its password blanked, in the URI form, a ``?password=`` query or the keyword form."""
    if not dsn:
        return None
    try:
        parts = urlsplit(dsn)
    except ValueError:
        return "***"
    if parts.password:
        creds, _, host = parts.netloc.rpartition("@")
        dsn = urlunsplit((parts.scheme, f"{creds.split(':', 1)[0]}:***@{host}", parts.path, parts.query, parts.fragment))
    return _KEYWORD_PASSWORD.sub(r"\1***", dsn)


def _location(uri: str) -> dict[str, Any]:
    opts = storage_options_from_env(uri)
    client_kwargs = opts.get("client_kwargs", {})
    configured = bool(opts.get("key") and opts.get("secret"))
    return {
        "uri": uri,
        "is_object_store": is_remote_uri(uri),
        "options": {
            "credentials_configured": configured,
            "credentials_masked": mask_secret(opts.get("key")) if configured else None,
            "endpoint_url": client_kwargs.get("endpoint_url"),
            "region": client_kwargs.get("region_name"),
        },
    }


def isaacus_description() -> dict[str, Any]:
    """Hosted or SageMaker, the endpoints, whether a key is set, and models no endpoint serves.

    ``endpoints_typo`` names a near-miss of ``ISAACUS_SAGEMAKER_ENDPOINTS``,
    which would otherwise fall back to the hosted API silently.
    """
    endpoints = endpoints_from_env()
    api_key = os.environ.get(API_KEY_ENV)
    return {
        "deployment": "sagemaker" if endpoints else "hosted",
        "endpoints": [{"name": e.name, "region": e.region, "models": list(e.models) if e.models else None}
                      for e in endpoints],
        "api_key_configured": bool(api_key),
        "api_key_masked": mask_secret(api_key),
        "models_checked": list(ISAACUS_MODELS),
        "unserved_models": [m for m in ISAACUS_MODELS if not resolve(SAGEMAKER).serves(m)] if endpoints else [],
        "endpoints_typo": misconfigured_endpoints_var() if not endpoints else None,
    }


def deployments_description() -> list[dict[str, Any]]:
    """Each registered deployment and the registered models it serves, without a network call.

    The variables a deployment reads are named and reported set or not; no value is shown. A model
    is listed under every deployment that may serve it, ``served`` saying whether this host's
    environment makes the deployment serve it now.
    """
    from womblex.analyse import embed, enrich  # noqa: F401 -- register the built-in models
    from womblex.ingest import paddle_ocr  # noqa: F401
    from womblex.utils.deployment_registry import deployments_for, distribution_version, known_deployments
    from womblex.utils.deployment_registry import resolve as resolve_deployment
    from womblex.utils.model_registry import SLOTS, known_names
    from womblex.utils.model_registry import resolve as resolve_model

    entries = [resolve_model(slot, name) for slot in SLOTS for name in known_names(slot)]
    out: list[dict[str, Any]] = []
    for name in known_deployments():
        deployment = resolve_deployment(name)
        distribution, version = distribution_version(deployment)
        out.append({
            "name": deployment.name,
            "distribution": distribution,
            "version": version,
            "needs_network": deployment.needs_network,
            "billed": deployment.billed,
            "variables": [{"names": list(group), "configured": group not in deployment.unset()}
                          for group in deployment.env],
            "optional_variables": list(deployment.env_optional),
            "models": [{"slot": e.slot, "name": e.name, "vendor_model": e.vendor_model or e.name,
                        "served": deployment.serves(e.vendor_model or e.name)}
                       for e in entries if deployment.name in deployments_for(e)],
        })
    return out


def deployment_resources(settings: DeploymentSettings) -> dict[str, Any]:
    """``store``, ``ingest``, ``queue``, ``isaacus`` and ``deployments``, described without a network call."""
    if settings.is_remote:
        store = {"kind": "remote", **_location(cast(str, settings.store_uri))}
    else:
        store = {"kind": "local", "uri": str(settings.output_root), "is_object_store": False, "options": {}}
    ingest = ({"configured": True, **_location(settings.ingest_uri)} if settings.ingest_uri
              else {"configured": False, "uri": None, "is_object_store": False, "options": {}})
    return {
        "store": store,
        "ingest": ingest,
        "queue": {"configured": bool(settings.db_dsn), "dsn_masked": mask_dsn(settings.db_dsn)},
        "isaacus": isaacus_description(),
        "deployments": deployments_description(),
    }


def check_store(settings: DeploymentSettings) -> tuple[bool, str | None]:
    """Whether the run source answers: a store's listing completes, or a local root is a directory."""
    if settings.is_remote:
        return store_reachable(cast(str, settings.store_uri))
    root = cast(Path, settings.output_root)
    return (True, None) if root.is_dir() else (False, f"{root} is not a directory")


def check_ingest(settings: DeploymentSettings) -> tuple[bool, str | None]:
    """Whether the ingest location answers a listing; an empty one does."""
    if not settings.ingest_uri:
        return False, "no ingest location configured"
    try:
        from womblex.store.remote import RemoteStore

        RemoteStore.from_uri(settings.ingest_uri).list_files("", "*")
    except Exception as e:
        logger.warning("ingest unreachable: %s", e)
        return False, str(e)
    return True, None
