"""Parquet output writer for extraction results.

Per-batch output is split across four sibling parquet files:

- ``batch-NNNN.elements.parquet`` — one row per element (the canonical
  structural stream)
- ``batch-NNNN.table_cells.parquet`` — children of ``kind='table'``
  elements, one row per cell
- ``batch-NNNN.form_fields.parquet`` — children of ``kind='form'``
  elements, one row per field
- ``batch-NNNN._manifest.parquet`` — one row per source file

Sidecars are joinable via ``(source_hash, parent_elem_order)``. Text is
verbatim from extraction; no post-processing is applied at the schema
boundary.

The caller passes ``output_path`` as ``batch-NNNN.parquet``; the writer
derives the four sibling paths from its stem.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from womblex.ingest.extract import ExtractionResult

logger = logging.getLogger(__name__)

PARSER_VERSION = "2.0"

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

_BBOX_TYPE = pa.struct([
    ("x", pa.float32()),
    ("y", pa.float32()),
    ("width", pa.float32()),
    ("height", pa.float32()),
])

ELEMENT_SCHEMA = pa.schema([
    ("source_hash", pa.string()),
    ("collection_id", pa.string()),
    ("elem_order", pa.int32()),
    ("kind", pa.string()),
    ("extractor", pa.string()),
    ("confidence", pa.float32()),
    ("page", pa.int32()),
    ("bbox", _BBOX_TYPE),
    ("text", pa.string()),
    ("alt_text", pa.string()),
    ("header_rows", pa.list_(pa.int32())),
    ("sheet", pa.string()),
    ("row", pa.int32()),
    ("col", pa.int32()),
    ("value", pa.string()),
    ("value_type", pa.string()),
    ("formula", pa.string()),
    ("number_format", pa.string()),
    ("merge_range", pa.string()),
    ("meta", pa.map_(pa.string(), pa.string())),
])

TABLE_CELLS_SCHEMA = pa.schema([
    ("source_hash", pa.string()),
    ("parent_elem_order", pa.int32()),
    ("row", pa.int32()),
    ("col", pa.int32()),
    ("value", pa.string()),
    ("rowspan", pa.int32()),
    ("colspan", pa.int32()),
    ("value_type", pa.string()),
])

FORM_FIELDS_SCHEMA = pa.schema([
    ("source_hash", pa.string()),
    ("parent_elem_order", pa.int32()),
    ("field_index", pa.int32()),
    ("name", pa.string()),
    ("value", pa.string()),
    ("field_type", pa.string()),
])

MANIFEST_SCHEMA = pa.schema([
    ("source_hash", pa.string()),
    ("collection_id", pa.string()),
    ("filename", pa.string()),
    ("ext", pa.string()),
    ("extraction_method", pa.string()),
    ("elements_count", pa.int64()),
    ("table_cells_count", pa.int64()),
    ("form_fields_count", pa.int64()),
    ("status", pa.string()),
    ("error", pa.string()),
    ("extracted_at_iso", pa.string()),
    ("parser_version", pa.string()),
])


_SHARD_ROLES = ("elements", "table_cells", "form_fields", "manifest")
_SHARD_SUFFIX = {
    "elements": ".elements.parquet",
    "table_cells": ".table_cells.parquet",
    "form_fields": ".form_fields.parquet",
    "manifest": "._manifest.parquet",
}
_SHARD_SCHEMA = {
    "elements": ELEMENT_SCHEMA,
    "table_cells": TABLE_CELLS_SCHEMA,
    "form_fields": FORM_FIELDS_SCHEMA,
    "manifest": MANIFEST_SCHEMA,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _shard_paths(output_path: Path) -> dict[str, Path]:
    """Map shard role → sibling parquet path derived from ``output_path``.

    ``output_path`` is the legacy single-file path (e.g.
    ``batch-0001.parquet``). Sibling sidecar files share its stem.
    """
    base = output_path.parent / output_path.stem
    return {role: Path(f"{base}{_SHARD_SUFFIX[role]}") for role in _SHARD_ROLES}


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


def _bbox_dict(b: Any) -> dict[str, float] | None:
    if b is None:
        return None
    return {"x": float(b.x), "y": float(b.y), "width": float(b.width), "height": float(b.height)}


def _meta_pairs(meta: dict[str, str] | None) -> list[tuple[str, str]]:
    if not meta:
        return []
    return [(str(k), str(v)) for k, v in meta.items()]


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def write_results(
    results: list[tuple[str, str, ExtractionResult]],
    output_path: Path,
    *,
    collection_id: str = "",
) -> Path:
    """Write a batch's results to sibling element + sidecar + manifest parquets.

    ``output_path`` is the legacy shard path (``batch-NNNN.parquet``);
    the four sibling files are written next to it sharing its stem.
    Returns ``output_path`` for caller back-compat.
    """
    paths = _shard_paths(output_path)
    elements_rows: list[dict] = []
    table_cells_rows: list[dict] = []
    form_fields_rows: list[dict] = []
    manifest_rows: list[dict] = []
    extracted_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    for doc_id, source_path, res in results:
        src_hash = _source_hash(source_path)
        tc_count = 0
        ff_count = 0

        for e in res.elements:
            elements_rows.append({
                "source_hash": src_hash,
                "collection_id": collection_id,
                "elem_order": e.order,
                "kind": e.kind,
                "extractor": e.extractor,
                "confidence": float(e.confidence),
                "page": e.page,
                "bbox": _bbox_dict(e.bbox),
                "text": e.text,
                "alt_text": e.alt_text,
                "header_rows": e.header_rows,
                "sheet": e.sheet,
                "row": e.row,
                "col": e.col,
                "value": e.value,
                "value_type": e.value_type,
                "formula": e.formula,
                "number_format": e.number_format,
                "merge_range": e.merge_range,
                "meta": _meta_pairs(e.meta),
            })
            if e.kind == "table" and e.cells:
                for c in e.cells:
                    table_cells_rows.append({
                        "source_hash": src_hash,
                        "parent_elem_order": e.order,
                        "row": c.row,
                        "col": c.col,
                        "value": c.value,
                        "rowspan": c.rowspan,
                        "colspan": c.colspan,
                        "value_type": c.value_type,
                    })
                    tc_count += 1
            elif e.kind == "form" and e.fields:
                for i, f in enumerate(e.fields):
                    form_fields_rows.append({
                        "source_hash": src_hash,
                        "parent_elem_order": e.order,
                        "field_index": i,
                        "name": f.name,
                        "value": f.value,
                        "field_type": f.field_type,
                    })
                    ff_count += 1

        manifest_rows.append({
            "source_hash": src_hash,
            "collection_id": collection_id,
            "filename": Path(source_path).name,
            "ext": Path(source_path).suffix.lower(),
            "extraction_method": res.method,
            "elements_count": len(res.elements),
            "table_cells_count": tc_count,
            "form_fields_count": ff_count,
            "status": "error" if res.error else "completed",
            "error": res.error or "",
            "extracted_at_iso": extracted_at,
            "parser_version": PARSER_VERSION,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(elements_rows, paths["elements"], ELEMENT_SCHEMA)
    _write_rows(table_cells_rows, paths["table_cells"], TABLE_CELLS_SCHEMA)
    _write_rows(form_fields_rows, paths["form_fields"], FORM_FIELDS_SCHEMA)
    _write_rows(manifest_rows, paths["manifest"], MANIFEST_SCHEMA)

    logger.info(
        "Wrote shard %s: docs=%d elements=%d table_cells=%d form_fields=%d",
        output_path.stem,
        len(manifest_rows), len(elements_rows),
        len(table_cells_rows), len(form_fields_rows),
    )
    return output_path


def _write_rows(rows: list[dict], path: Path, schema: pa.Schema) -> None:
    if rows:
        table = pa.Table.from_pylist(rows, schema=schema)
    else:
        table = pa.table({f.name: pa.array([], type=f.type) for f in schema}, schema=schema)
    pq.write_table(table, str(path), compression="zstd", compression_level=3)


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def read_elements(path: Path) -> pa.Table:
    """Read elements from a single shard file or a shard-directory glob."""
    return _read_role(path, role="elements")


def read_table_cells(path: Path) -> pa.Table:
    return _read_role(path, role="table_cells")


def read_form_fields(path: Path) -> pa.Table:
    return _read_role(path, role="form_fields")


def read_manifest(path: Path) -> pa.Table:
    return _read_role(path, role="manifest")


def _read_role(path: Path, *, role: str) -> pa.Table:
    schema = _SHARD_SCHEMA[role]
    p = Path(path)
    if p.is_dir():
        shards = sorted(p.glob(f"*{_SHARD_SUFFIX[role]}"))
        if not shards:
            return pa.table({f.name: pa.array([], type=f.type) for f in schema}, schema=schema)
        return pa.concat_tables([pq.read_table(str(s), schema=schema) for s in shards])
    # Single-file path: interpret as shard base (drop .parquet, append role suffix)
    role_path = p.parent / f"{p.stem}{_SHARD_SUFFIX[role]}"
    return pq.read_table(str(role_path), schema=schema)


# Back-compat alias for callers that still expect read_results to return
# the element table.
read_results = read_elements


# ---------------------------------------------------------------------------
# Integrity verification
# ---------------------------------------------------------------------------


class ShardVerificationError(RuntimeError):
    """Raised when a per-batch parquet shard fails on-disk verification."""


def verify_shard_persistence(
    output_path: Path,
    expected_docs: int,
    prev_total_size: int,
) -> int:
    """Sanity-check shard files after a write.

    Checks: every shard file exists, none are empty, manifest row count
    matches ``expected_docs``, every (source_hash, parent_elem_order) in
    table_cells / form_fields references an element with the matching
    kind, and the cumulative on-disk size has not shrunk.

    Returns the new cumulative on-disk size of the shard directory.
    Raises ``ShardVerificationError`` on any anomaly.
    """
    paths = _shard_paths(output_path)
    for role, path in paths.items():
        if not path.exists():
            raise ShardVerificationError(f"shard missing after write: {path}")
        if path.stat().st_size == 0:
            raise ShardVerificationError(f"shard is zero bytes: {path}")
        try:
            pq.ParquetFile(str(path)).metadata  # smoke-test readability
        except Exception as e:
            raise ShardVerificationError(f"shard unreadable: {path}: {e}") from e

    manifest_rows = pq.ParquetFile(str(paths["manifest"])).metadata.num_rows
    if manifest_rows != expected_docs:
        raise ShardVerificationError(
            f"manifest row count mismatch: {paths['manifest']} has {manifest_rows} rows, "
            f"expected {expected_docs}"
        )

    _verify_sidecar_integrity(paths)

    shard_dir = output_path.parent
    cumulative_size = sum(p.stat().st_size for p in shard_dir.glob("*.parquet"))
    if cumulative_size < prev_total_size:
        raise ShardVerificationError(
            f"shard directory shrank: {shard_dir} was {prev_total_size}b, "
            f"now {cumulative_size}b — likely overwrite"
        )
    return cumulative_size


def _verify_sidecar_integrity(paths: dict[str, Path]) -> None:
    """Every (source_hash, parent_elem_order) in sidecars must map to a
    matching-kind element. Raises ShardVerificationError on orphan refs.
    """
    elems = pq.read_table(str(paths["elements"]), columns=["source_hash", "elem_order", "kind"])
    # Build {(source_hash, elem_order): kind}
    src = elems.column("source_hash").to_pylist()
    ord_ = elems.column("elem_order").to_pylist()
    kind = elems.column("kind").to_pylist()
    elem_index = {(s, o): k for s, o, k in zip(src, ord_, kind)}

    _check_sidecar(paths["table_cells"], elem_index, expected_kind="table", role="table_cells")
    _check_sidecar(paths["form_fields"], elem_index, expected_kind="form", role="form_fields")


def _check_sidecar(
    path: Path, elem_index: dict[tuple[str, int], str], *, expected_kind: str, role: str,
) -> None:
    t = pq.read_table(str(path), columns=["source_hash", "parent_elem_order"])
    src = t.column("source_hash").to_pylist()
    parent = t.column("parent_elem_order").to_pylist()
    for s, o in zip(src, parent):
        k = elem_index.get((s, o))
        if k is None:
            raise ShardVerificationError(
                f"{role} references missing element: source_hash={s} parent_elem_order={o}"
            )
        if k != expected_kind:
            raise ShardVerificationError(
                f"{role} references element with wrong kind: "
                f"source_hash={s} parent_elem_order={o} kind={k!r} (expected {expected_kind!r})"
            )
