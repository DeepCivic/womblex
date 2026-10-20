"""Copy a run's raw sources into an egress bundle, one outcome per ``source_hash``.

The status vocabulary is :mod:`womblex.store.source_resolver`'s four, plus
``upload_failed``. :class:`LocalSources` copies from a local corpus:
`SourceResolver` finds and checks the file, which is then uploaded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from womblex.store.egress_output import raw_key_for
from womblex.store.remote import RemoteStore
from womblex.store.source_resolver import NOT_FOUND, RESOLVED, SourceResolver

logger = logging.getLogger(__name__)

#: A source that resolved but whose copy to the destination raised. Not part
#: of :mod:`womblex.store.source_resolver`'s vocabulary — it happens at
#: upload time, after resolution has already succeeded.
UPLOAD_FAILED = "upload_failed"


@dataclass(frozen=True)
class SourceOutcome:
    """One ``source_hash``'s copy: its status, the bundle key it landed at (``None`` where no file
    was produced), and the key this export attempted, which stale-source cleanup leaves alone."""

    status: str
    raw_key: str | None = None
    attempted: str | None = None


class LocalSources:
    """Sources in a local corpus, found and checked by :class:`SourceResolver`."""

    def __init__(self, resolver: SourceResolver) -> None:
        self.resolver = resolver
        self._resolutions = {r.source_hash: r for r in resolver.resolve_all()}

    def copy(self, source_hash: str, store: RemoteStore, prefix: str) -> SourceOutcome:
        resolution = self._resolutions.get(source_hash)
        if resolution is None:
            return SourceOutcome(NOT_FOUND)
        if not resolution.ok or resolution.path is None:
            return SourceOutcome(resolution.status)
        key = raw_key_for(source_hash, resolution.path.suffix.lower())
        try:
            store.upload_file(resolution.path, f"{prefix}/{key}")
        except Exception:
            logger.warning("failed to upload source for source_hash=%s (doc_id=%s)",
                           source_hash, resolution.doc_id, exc_info=True)
            return SourceOutcome(UPLOAD_FAILED, attempted=key)
        return SourceOutcome(RESOLVED, key, key)


__all__ = ["UPLOAD_FAILED", "LocalSources", "SourceOutcome"]
