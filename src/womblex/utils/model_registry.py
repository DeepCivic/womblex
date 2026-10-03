"""Named model registry: one place that maps a config name to a model factory.

A *slot* is a place in the pipeline that takes a swappable model (OCR engine,
layout analyser, ...). Built-ins register under their existing names and
aliases; installed packages add more through an entry point in the group
``womblex.models.<slot>`` whose value is the factory itself, e.g.::

    [project.entry-points."womblex.models.ocr"]
    my-ocr = "my_pkg.ocr:make_reader"

A factory may carry a ``womblex_traits`` dict (for example
``{"markdown": True}`` on an OCR factory whose reader returns page markdown).
Config names registered models only: an unknown name is an error listing the
known names, and anything shaped like an import path is refused outright.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any

logger = logging.getLogger(__name__)

SLOT_OCR = "ocr"
SLOT_LAYOUT = "layout"
SLOTS: tuple[str, ...] = (SLOT_OCR, SLOT_LAYOUT)

ENTRY_POINT_GROUP_PREFIX = "womblex.models."


@dataclass(frozen=True)
class ModelEntry:
    """A registered model: its canonical name, factory and declared traits."""

    slot: str
    name: str
    factory: Callable[..., Any]
    source: str = "builtin"
    traits: dict[str, Any] = field(default_factory=dict)


_entries: dict[str, dict[str, ModelEntry]] = {slot: {} for slot in SLOTS}
_aliases: dict[str, dict[str, str]] = {slot: {} for slot in SLOTS}
_loaded: set[str] = set()


def _check_slot(slot: str) -> None:
    if slot not in _entries:
        raise ValueError(f"unknown model slot: {slot!r} (known: {list(SLOTS)})")


def _norm(name: str) -> str:
    return name.strip().lower()


def register(
    slot: str,
    name: str,
    factory: Callable[..., Any],
    *,
    aliases: tuple[str, ...] = (),
    source: str = "builtin",
    traits: dict[str, Any] | None = None,
) -> None:
    """Register *factory* under *name* (and *aliases*) for *slot*.

    A name already held by a different model is refused rather than shadowed,
    so a plugin cannot silently replace a built-in.
    """
    _check_slot(slot)
    canonical = _norm(name)
    for key in (canonical, *(_norm(a) for a in aliases)):
        owner = _aliases[slot].get(key)
        if owner is not None and owner != canonical:
            raise ValueError(
                f"{slot} model name {key!r} is already registered to {owner!r}"
            )
    _entries[slot][canonical] = ModelEntry(
        slot, canonical, factory, source, dict(traits or {})
    )
    for key in (canonical, *(_norm(a) for a in aliases)):
        _aliases[slot][key] = canonical


def _load_plugins(slot: str) -> None:
    """Register every installed entry point for *slot* once per process."""
    if slot in _loaded:
        return
    _loaded.add(slot)
    for ep in entry_points(group=ENTRY_POINT_GROUP_PREFIX + slot):
        try:
            factory = ep.load()
        except Exception:
            logger.exception("model plugin %s (%s) failed to load", ep.name, slot)
            continue
        dist = ep.dist.name if ep.dist is not None else "unknown"
        try:
            register(
                slot, ep.name, factory, source=dist,
                traits=getattr(factory, "womblex_traits", None),
            )
        except ValueError as exc:
            logger.warning("model plugin %s (%s) skipped: %s", ep.name, slot, exc)


def known_names(slot: str) -> list[str]:
    """Canonical names registered for *slot*, built-ins and plugins."""
    _check_slot(slot)
    _load_plugins(slot)
    return sorted(_entries[slot])


def resolve(slot: str, name: str) -> ModelEntry:
    """Return the entry registered for *name*, or raise listing the known names."""
    _check_slot(slot)
    _load_plugins(slot)
    key = _norm(name)
    canonical = _aliases[slot].get(key)
    if canonical is None:
        hint = ""
        if ":" in key or "/" in key or "." in key:
            hint = " (import paths are not accepted; name a registered model)"
        raise ValueError(
            f"unknown {slot} model: {name!r}{hint} (known: {known_names(slot)})"
        )
    return _entries[slot][canonical]
