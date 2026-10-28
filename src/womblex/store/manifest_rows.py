"""The manifest row a batch writes for one document, and the hash that keys it."""

from __future__ import annotations

import hashlib
from pathlib import Path

from womblex.ingest.extract import ExtractionResult
from womblex.store.content_digest import content_digest


def _source_hash(source_path: str) -> str:
    """SHA-256 of the source bytes. Returns '' if the file is unreadable."""
    p = Path(source_path)
    if not p.exists() or not p.is_file():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_row(
    doc_id: str, source_path: str, res: ExtractionResult, *, source_hash: str, collection_id: str,
    ingest_root: str, relpath: str, tc_count: int, ff_count: int, extracted_at: str, parser_version: str,
) -> dict:
    """One `MANIFEST_SCHEMA` row (the versioning columns stay null in a per-batch manifest)."""
    return {
        "source_hash": source_hash,
        "collection_id": collection_id,
        "doc_id": doc_id,
        "ingest_root": ingest_root,
        "source_relpath": relpath,
        "filename": Path(source_path).name,
        "ext": Path(source_path).suffix.lower(),
        "extraction_method": res.method,
        "elements_count": len(res.elements),
        "table_cells_count": tc_count,
        "form_fields_count": ff_count,
        "status": "error" if res.error else "completed",
        "error": res.error or "",
        "extracted_at_iso": extracted_at,
        "parser_version": parser_version,
        "content_digest": content_digest(res.elements),
    }
