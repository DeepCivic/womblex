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

from womblex.utils.isaacus_client import API_KEY_ENV, sagemaker_configured


def isaacus_available() -> bool:
    """True when the Isaacus SDK is installed and a deployment is configured.

    Cheap and network-free — checks for the ``isaacus`` package plus either a
    non-empty ``ISAACUS_API_KEY`` (hosted API) or declared SageMaker endpoints
    (private deployment, which needs no key). The chunk stage uses this to
    decide whether the Kanon-2 tokeniser can be loaded at all. Propagates
    ``ValueError`` from a malformed endpoint spec rather than degrading to a
    silent skip.
    """
    if importlib.util.find_spec("isaacus") is None:
        return False
    if sagemaker_configured():
        return True
    return bool(os.environ.get(API_KEY_ENV))
