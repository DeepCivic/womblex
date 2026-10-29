"""The manifest row a batch writes for one document, and the hash that keys it."""

from __future__ import annotations

import hashlib
from pathlib import Path

from womblex.ingest.extract import ExtractionResult
from womblex.store.content_digest import content_digest

#: The `extraction_method` of a row whose source could not be read: its `source_hash` is not over bytes.
UNREADABLE_METHOD = "unreadable"


def _source_hash(source_path: str) -> str:
    """SHA-256 of the source bytes. Returns '' if the file is unreadable."""
    p = Path(source_path)
    if not p.exists() or not p.is_file():
        return ""
    h = hashlib.sha256()
    try:
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:  # present but not readable (permissions, a read error): unreadable, not a failed batch
        return ""
    return h.hexdigest()


def unreadable_hash(source_relpath: str, batch_num: int) -> str:
    """The `source_hash` of a source that could not be read: SHA-256 over `unreadable:<relpath>:<batch>`.

    Never empty, so the row joins; the batch number keeps an unreadable file in two firings of one
    collection two distinct versions in its chain.
    """
    return hashlib.sha256(f"unreadable:{source_relpath}:{batch_num}".encode()).hexdigest()


def manifest_row(
    doc_id: str, source_path: str, res: ExtractionResult, *, source_hash: str, collection_id: str,
    ingest_root: str, relpath: str, tc_count: int, ff_count: int, extracted_at: str, parser_version: str,
    unreadable: bool = False,
) -> dict:
    """One `MANIFEST_SCHEMA` row (the versioning columns stay null in a per-batch manifest).

    An *unreadable* source is a failed row whose method says so; any other failure keeps the file-bytes
    hash and the method it had, so `status` alone does not mark the case.
    """
    return {
        "source_hash": source_hash,
        "collection_id": collection_id,
        "doc_id": doc_id,
        "ingest_root": ingest_root,
        "source_relpath": relpath,
        "filename": Path(source_path).name,
        "ext": Path(source_path).suffix.lower(),
        "extraction_method": UNREADABLE_METHOD if unreadable else res.method,
        "elements_count": len(res.elements),
        "table_cells_count": tc_count,
        "form_fields_count": ff_count,
        "status": "error" if res.error or unreadable else "completed",
        "error": res.error or ("the source could not be read" if unreadable else ""),
        "extracted_at_iso": extracted_at,
        "parser_version": parser_version,
        "content_digest": content_digest(res.elements),
    }
