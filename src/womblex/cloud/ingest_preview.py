"""What a run over an ingest prefix would pick up, before anything is enqueued.

The count is :func:`womblex.cloud.dispatch.supported_keys`, the listing
`enqueue_extraction` itself uses, so a previewed prefix enqueues the count
shown. A nested layout, which a run refuses, reports the refusal and the
subdirectories that would each run, with a prefix ready to submit.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from womblex.cli._shared import NestedCorpusError
from womblex.cloud.dispatch import owned_prefix, supported_keys

if TYPE_CHECKING:
    from womblex.cloud.deployment import DeploymentSettings

logger = logging.getLogger(__name__)

#: Keys returned as a sample of what would run.
SAMPLE_SIZE = 5


def ingest_preflight(
    settings: DeploymentSettings, *, input_prefix: str | None = None, owner: str | None = None,
) -> dict[str, Any]:
    """Reachability and document count of the ingest location *input_prefix* names.

    An unreachable or unconfigured location is reported, not raised. Raises
    ``ValueError`` for a prefix that escapes the ingest root and
    `PrefixNotOwned` for one outside *owner*'s tree, as the enqueue does.
    """
    prefix = owned_prefix(owner, input_prefix)
    empty: dict[str, Any] = {
        "uri": settings.ingest_uri, "input_prefix": prefix, "kind": None, "reachable": False,
        "document_count": 0, "sample": [], "subdirectories": [], "error": None,
    }
    if not settings.ingest_uri:
        return {**empty, "error": "no ingest location configured"}
    from womblex.store.remote import RemoteStore, is_remote_uri

    kind = "remote" if is_remote_uri(settings.ingest_uri) else "local"
    try:
        _, keys = supported_keys(RemoteStore.from_uri(settings.ingest_uri), settings.ingest_uri, prefix)
    except NestedCorpusError as e:
        return {
            **empty, "kind": kind, "reachable": True, "error": str(e),
            "subdirectories": [
                {"name": name, "input_prefix": f"{prefix}/{name}" if prefix else name, "document_count": count}
                for name, count in sorted(e.nested.items())
            ],
        }
    except Exception as e:
        logger.warning("ingest preflight: %s unreachable: %s", settings.ingest_uri, e)
        return {**empty, "kind": kind, "error": str(e)}
    return {**empty, "kind": kind, "reachable": True, "document_count": len(keys), "sample": keys[:SAMPLE_SIZE]}
