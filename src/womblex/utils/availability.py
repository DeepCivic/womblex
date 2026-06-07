"""Service-availability checks used to gate optional pipeline stages.

The chunker sizes chunks with the Kanon-2 tokeniser, which is only
obtainable through the Isaacus API (Kanon-2 is API-only). Stages that
build a chunker gate on :func:`isaacus_available`: when the API isn't
configured, chunking is skipped rather than attempting a tokeniser load
that can't succeed.
"""

from __future__ import annotations

import importlib.util
import os


def isaacus_available() -> bool:
    """True when the Isaacus SDK is installed and an API key is set.

    Cheap and network-free — checks for the ``isaacus`` package and a
    non-empty ``ISAACUS_API_KEY``. The chunk stage uses this to decide
    whether the Kanon-2 tokeniser can be loaded at all.
    """
    if importlib.util.find_spec("isaacus") is None:
        return False
    return bool(os.environ.get("ISAACUS_API_KEY"))
