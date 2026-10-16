"""Shared manifest rows and writers for the console tests (``tests/test_ui_*.py``).

The ``api_client`` fixture they use lives in ``tests/conftest.py``.
"""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from womblex.store.output import ELEMENTS_SUFFIX, MANIFEST_SCHEMA

REPO_ROOT = Path(__file__).resolve().parent.parent


_OK_ROW = {
    "source_hash": "hash-a", "collection_id": "test", "doc_id": "doc-a",
    "filename": "doc-a.pdf", "ext": ".pdf", "extraction_method": "native",
    "elements_count": 3, "table_cells_count": 0, "form_fields_count": 0,
    "status": "ok", "error": "", "extracted_at_iso": "2026-08-16T00:00:00Z",
    "parser_version": "2.0",
}
# A failed document, which the Corpus Inspector's failure filter needs to see:
# the manifest carries it as a row, not as an omission.
_FAILED_ROW = {
    **_OK_ROW,
    "source_hash": "hash-b", "doc_id": "doc-b", "filename": "doc-b.pdf",
    "status": "error", "error": "boom",
}
_ROWS = [_OK_ROW, _FAILED_ROW]


def _write_manifest_shard(shard_dir: Path, rows: list[dict]) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(rows, schema=MANIFEST_SCHEMA),
        str(shard_dir / "batch-0001._manifest.parquet"),
    )
    (shard_dir / f"batch-0001{ELEMENTS_SUFFIX}").write_bytes(b"stub")


def _write_consolidated_manifest(run_dir: Path, rows: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(rows, schema=MANIFEST_SCHEMA), str(run_dir / "manifest.parquet"))

