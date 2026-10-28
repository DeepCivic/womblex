"""Named deployment registry: where a vendor's models are served from, declared once.

A *deployment* is the thing a model is called on: a hosted API, a private
endpoint, a local server. It declares the environment variables it reads, whether
it needs the network and is billed, a free reachability check, an optional billed
smoke call, the models it serves and how its client is built. Models in any slot
name the deployments that may serve them (``ModelEntry.deployments``), so
credentials, reachability and the client are declared and checked once per
deployment rather than once per function.

Built-ins register when first looked up; installed packages add more through an
entry point in the group ``womblex.deployments`` whose name is the deployment's and
whose value is a :class:`Deployment`::

    [project.entry-points."womblex.deployments"]
    my-vendor = "my_pkg.deployment:DEPLOYMENT"

A declaration names variables, never a secret.
"""

from __future__ import annotations

import importlib
import logging
import os
import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from womblex.utils.model_registry import ModelEntry, plugin_objects

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "womblex.deployments"

#: The modules that register the built-in deployments at import.
_BUILTIN_MODULES = (
    "womblex.providers.bedrock.deployment",
    "womblex.providers.isaacus.deployment",
    "womblex.providers.ollama.deployment",
)


def serves_everything(model: str) -> bool:
    """The default ``Deployment.serves``: every model id."""
    return True


@dataclass(frozen=True)
class Deployment:
    """A registered deployment; see the module docstring."""

    name: str
    check: Callable[[Sequence[str]], None]
    """Free reachability check, called once with the vendor model ids a run uses on it: raises unless it answers and serves them.

    Every callable here takes vendor ids (``ModelEntry.vendor_model``), never Womblex's registered names.
    """
    needs_network: bool
    billed: bool
    env: tuple[tuple[str, ...], ...] = ()
    """Required variables: every group must have at least one of its alternatives set."""
    env_optional: tuple[str, ...] = ()
    """Variables read when set, with a default otherwise."""
    serves: Callable[[str], bool] = serves_everything
    served: Callable[[], str] = lambda: "every model"
    """What the deployment serves, in words, for a refusal that names it."""
    smoke: Callable[[str, str], None] | None = None
    """``smoke(slot, model)``: one billed minimal call that raises on failure; ``None`` where there is none."""
    make_client: Callable[[Iterable[str]], object] | None = None
    """Build the client for the model ids a caller will use; ``None`` where a provider builds its own."""
    source: str = "builtin"
    version: str = ""

    def unset(self) -> list[tuple[str, ...]]:
        """The required variable groups with none of their alternatives set."""
        return [g for g in self.env if not any((os.environ.get(v) or "").strip() for v in g)]


_deployments: dict[str, Deployment] = {}
_loaded = False
_clients: dict[str, object] = {}
_clients_lock = threading.Lock()


def _norm(name: str) -> str:
    return name.strip().lower()


def register(deployment: Deployment) -> None:
    """Register *deployment*; a name already held by a different source is refused, not shadowed."""
    key = _norm(deployment.name)
    existing = _deployments.get(key)
    if existing is not None and existing.source != deployment.source:
        raise ValueError(f"deployment name {key!r} is already registered by {existing.source!r}")
    _deployments[key] = deployment


def _load() -> None:
    """Register the built-ins and every installed entry point, once per process."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    for module in _BUILTIN_MODULES:
        importlib.import_module(module)
    for name, obj, dist, version in plugin_objects(ENTRY_POINT_GROUP):
        if not isinstance(obj, Deployment) or _norm(obj.name) != _norm(name):
            logger.warning("deployment plugin %s skipped: it must be a Deployment of the same name", name)
            continue
        try:
            register(replace(obj, source=dist, version=version))
        except ValueError as exc:
            logger.warning("deployment plugin %s skipped: %s", name, exc)


def known_deployments() -> list[str]:
    """Names registered, built-ins and plugins."""
    _load()
    return sorted(_deployments)


def resolve(name: str) -> Deployment:
    """The deployment registered as *name*, or raise listing the known names."""
    _load()
    try:
        return _deployments[_norm(name)]
    except KeyError:
        raise ValueError(f"unknown deployment: {name!r} (known: {sorted(_deployments)})") from None


def client_for(name: str, models: Iterable[str] = ()) -> object:
    """The one client this process holds for deployment *name*, built on first use.

    Every provider on a deployment is handed the same client. *models* are the ids the
    caller will use: the deployment's client is built to serve them, and a later caller's
    must be served by the deployment too, or this raises.
    """
    deployment = resolve(name)
    if deployment.make_client is None:
        raise ValueError(f"deployment {deployment.name!r} builds no client")
    wanted = list(models)
    with _clients_lock:
        client = _clients.get(deployment.name)
        if client is None:
            client = _clients[deployment.name] = deployment.make_client(wanted)
            return client
    unserved = [m for m in wanted if not deployment.serves(m)]
    if unserved:
        raise ValueError(f"deployment {deployment.name!r} does not serve {', '.join(unserved)}")
    return client


def reset_clients() -> None:
    """Forget the clients this process built. For tests, and after the environment changes."""
    with _clients_lock:
        _clients.clear()


#: What the ``isaacus`` model trait means: served by the Isaacus deployments.
ISAACUS_DEPLOYMENTS = ("isaacus-api", "isaacus-sagemaker")


def deployments_for(entry: ModelEntry) -> tuple[str, ...]:
    """The deployments that may serve *entry*: those it declares, else Isaacus's for the ``isaacus`` trait.

    Empty for a model that runs locally.
    """
    if entry.deployments:
        return entry.deployments
    return ISAACUS_DEPLOYMENTS if entry.traits.get("isaacus") else ()


def distribution_version(deployment: Deployment) -> tuple[str, str]:
    """The distribution and version that supplied *deployment*; Womblex's own for a built-in."""
    if deployment.source == "builtin":
        from womblex import __version__

        return "womblex", __version__
    return deployment.source, deployment.version


def deployment_for_pinging_model(slot: str, name: str, model: Any) -> Deployment:
    """A deployment built from a built model that has ``ping()`` and declared none.

    Its free check is the ping; ``ping_is_billed`` marks it billed, and only a billed
    one is taken to need the network. Such a reader keeps working unchanged, checked
    once per model.
    """
    billed = bool(getattr(model, "ping_is_billed", False))
    return Deployment(
        name=f"{slot}:{_norm(name)}", check=lambda _models: model.ping(), needs_network=billed, billed=billed,
    )
