"""Compare two Parquet files by content: rows, schema and footer without the write time.

Every pipeline Parquet records when it was written (`womblex.written_at`), so two writes of
the same rows differ in their bytes. A test that asserts two writes agree compares them here.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from womblex.store.contract import WRITTEN_AT_KEY


def footer_without_write_time(path: Path) -> dict[bytes, bytes]:
    meta = pq.read_schema(str(path)).metadata or {}
    return {k: v for k, v in meta.items() if k != WRITTEN_AT_KEY.encode()}


def assert_same_parquet(a: Path, b: Path) -> None:
    """Rows, schema and footer (less the write time) of *a* and *b* are equal."""
    assert pq.read_table(str(a)).equals(pq.read_table(str(b))), f"{a.name}: rows differ"
    assert footer_without_write_time(a) == footer_without_write_time(b), f"{a.name}: footers differ"
