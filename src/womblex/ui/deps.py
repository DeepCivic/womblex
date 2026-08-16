"""Where the console reads run state from.

The sidecar reaches run state exactly the way a worker does
(docs/ui-plan.md §2), so it adds no configuration surface of its own: a
local deployment points at an ``output_root`` that is bind-mounted
read-only, and a cloud deployment sets ``WOMBLEX_STORE_URI`` — the same
variable ``womblex-cloud`` already reads.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from fastapi import Request


@dataclass(frozen=True)
class UISettings:
    """Where the console reads run state from. Exactly one of the two is set."""

    output_root: Path | None
    store_uri: str | None
    allow_execute: bool = False

    def __post_init__(self) -> None:
        if bool(self.output_root) == bool(self.store_uri):
            raise ValueError("UISettings needs exactly one of output_root or store_uri")

    @property
    def is_remote(self) -> bool:
        return self.store_uri is not None


def resolve_settings(
    output_root: Path | None,
    store_uri: str | None,
    *,
    allow_execute: bool = False,
) -> UISettings:
    """Resolve settings from explicit arguments, falling back to env vars.

    ``$WOMBLEX_UI_OUTPUT_ROOT`` names a local run root;
    ``$WOMBLEX_STORE_URI`` names an object store. Raises ``ValueError`` when
    both or neither resolve — the console reads exactly one run source, and
    silently preferring one over the other would hide a misconfigured
    deployment behind an empty run list.
    """
    root = output_root
    if root is None and "WOMBLEX_UI_OUTPUT_ROOT" in os.environ:
        root = Path(os.environ["WOMBLEX_UI_OUTPUT_ROOT"])
    store = store_uri or os.environ.get("WOMBLEX_STORE_URI")
    if root and store:
        raise ValueError("pass only one of --output-root / --store (or their env vars)")
    if not root and not store:
        raise ValueError(
            "no run source: pass --output-root or --store "
            "(or set $WOMBLEX_UI_OUTPUT_ROOT / $WOMBLEX_STORE_URI)"
        )
    return UISettings(output_root=root, store_uri=store, allow_execute=allow_execute)


def get_settings(request: Request) -> UISettings:
    """FastAPI dependency: the app-wide settings resolved at startup."""
    settings: UISettings = request.app.state.settings
    return settings
