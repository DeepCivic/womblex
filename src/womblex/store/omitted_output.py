"""``omitted.parquet``: the files a run did not look at, and why.

A run ingests the supported documents directly under its ingest location. The rest of the listing
is passed over (`cli/_shared.select_files`); this records each such file, so a consumer can tell
"not looked for" from "looked for and found nothing". One file at the run root, written when the run
is enqueued or starts; a run that omitted nothing still writes it, with no rows.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

from womblex.store.output import _write_rows

if TYPE_CHECKING:  # pragma: no cover - typing only
    from womblex.cli._shared import Omitted
    from womblex.store.remote import RemoteStore

logger = logging.getLogger(__name__)

OMITTED_FILENAME = "omitted.parquet"
REASONS = ("unsupported_type", "nested")

OMITTED_SCHEMA = pa.schema([
    ("source_relpath", pa.string()),  # path under the ingest location
    ("reason", pa.string()),  # unsupported_type | nested
])


def write_omitted(omitted: Iterable[Omitted], run_root: Path) -> Path:
    """Write *omitted* to ``<run_root>/omitted.parquet``, replacing an earlier one."""
    target = run_root / OMITTED_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_rows([{"source_relpath": o.source_relpath, "reason": o.reason} for o in omitted],
                target, OMITTED_SCHEMA, role="omitted")
    return target


def publish_omitted(omitted: Iterable[Omitted], store: RemoteStore, output_prefix: str) -> None:
    """Write *omitted* to ``<output_prefix>/omitted.parquet`` in *store*. A failure is logged, not raised:
    the record is a label on the run, and the run is not held for it."""
    try:
        with tempfile.TemporaryDirectory(prefix="womblex-omitted-") as tmp:
            store.upload_file(write_omitted(omitted, Path(tmp)), f"{output_prefix.strip('/')}/{OMITTED_FILENAME}")
    except Exception:
        logger.warning("could not record the omitted files under %s", output_prefix, exc_info=True)


def read_omitted(path: Path) -> pa.Table:
    """The omitted files of the run whose ``omitted.parquet`` is *path*."""
    return pq.read_table(str(path))


__all__ = ["OMITTED_FILENAME", "OMITTED_SCHEMA", "REASONS", "publish_omitted", "read_omitted", "write_omitted"]
