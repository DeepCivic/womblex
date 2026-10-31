"""Which deployment each model ran on, as a Parquet footer key (`womblex.deployments`).

Written beside `womblex.slot_models` by the run stamp: one entry per model this process used on
a deployment (slot, model, deployment, the distribution and version that supplied the deployment,
and what the deployment may say about where it ran). An ARN is reduced to its endpoint name and
region; no key, secret or account id reaches it. A process that used no deployed model writes no key.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

from womblex.store.source_provenance import NAMESPACE
from womblex.utils.deployment_registry import distribution_version, resolve
from womblex.utils.model_registry import resolve as resolve_model
from womblex.utils.model_registry import used_deployments

DEPLOYMENTS_KEY = f"{NAMESPACE}.deployments"


def _vendor_id(slot: str, name: str) -> str:
    """The id a deployment knows *name* by; AI chunking's model (slot ``chunking``) is already one."""
    try:
        return resolve_model(slot, name).vendor_model or name
    except ValueError:
        return name


def deployments_payload() -> list[dict[str, str]]:
    """The entries for the footer: the deployments this process's models ran on."""
    out: list[dict[str, str]] = []
    for slot, name, deployment in used_deployments():
        entry = {"slot": slot, "name": name, "deployment": deployment}
        try:
            registered = resolve(deployment)
        except ValueError:  # recorded, then unregistered: say what ran, claim nothing more
            out.append(entry)
            continue
        dist, version = distribution_version(registered)
        out.append({**entry, "distribution": dist, "version": version, **registered.detail(_vendor_id(slot, name))})
    return out


def read_footer_deployments(metadata: Mapping[bytes, bytes] | None) -> list[dict[str, str]]:
    """Decode the deployments key out of a Parquet footer; ``[]`` for a file without one or a malformed one."""
    raw = (metadata or {}).get(DEPLOYMENTS_KEY.encode())
    if raw is None:
        return []
    try:
        entries = json.loads(raw.decode())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []
    if not isinstance(entries, list):
        return []
    return [{str(k): str(v) for k, v in e.items()} for e in entries
            if isinstance(e, dict) and {"slot", "name", "deployment"} <= e.keys()]
