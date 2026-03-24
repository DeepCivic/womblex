"""Parquet output writer for extraction results.

Writes one row per document with nested columns for tables, forms,
images, and text blocks. Uses PyArrow for schema enforcement and
efficient columnar storage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from womblex.ingest.extract import ExtractionResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parquet schema
# ---------------------------------------------------------------------------

_POSITION_TYPE = pa.struct([
    ("x", pa.float64()),
    ("y", pa.float64()),
    ("width", pa.float64()),
    ("height", pa.float64()),
])

EXTRACTION_SCHEMA = pa.schema([
    ("document_id", pa.string()),
    ("source_path", pa.string()),
    ("text", pa.string()),
    ("metadata", pa.struct([
        ("extraction_strategy", pa.string()),
        ("confidence", pa.float64()),
        ("processing_time", pa.float64()),
        ("page_count", pa.int32()),
        ("text_coverage", pa.float64()),
    ])),
    ("warnings", pa.list_(pa.string())),
    ("tables", pa.list_(pa.struct([
        ("headers", pa.list_(pa.string())),
        ("rows", pa.list_(pa.list_(pa.string()))),
        ("position", _POSITION_TYPE),
        ("confidence", pa.float64()),
    ]))),
    ("forms", pa.list_(pa.struct([
        ("field_name", pa.string()),
        ("value", pa.string()),
        ("position", _POSITION_TYPE),
        ("confidence", pa.float64()),
    ]))),
    ("images", pa.list_(pa.struct([
        ("alt_text", pa.string()),
        ("position", _POSITION_TYPE),
        ("confidence", pa.float64()),
    ]))),
    ("text_blocks", pa.list_(pa.struct([
        ("text", pa.string()),
        ("position", _POSITION_TYPE),
        ("block_type", pa.string()),
        ("confidence", pa.float64()),
    ]))),
])


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _position_dict(pos: Any) -> dict[str, float]:
    """Convert a Position dataclass to a plain dict."""
    return {"x": pos.x, "y": pos.y, "width": pos.width, "height": pos.height}


def _result_to_row(
    doc_id: str,
    source_path: str,
    result: ExtractionResult,
) -> dict[str, Any]:
    """Convert an ExtractionResult to a flat dict matching the Parquet schema."""
    meta = result.metadata
    meta_dict = {
        "extraction_strategy": meta.extraction_strategy if meta else result.method,
        "confidence": meta.confidence if meta else 0.0,
        "processing_time": meta.processing_time if meta else 0.0,
        "page_count": meta.page_count if meta else result.page_count,
        "text_coverage": meta.text_coverage if meta else 0.0,
    }

    tables = [
        {
            "headers": t.headers,
            "rows": t.rows,
            "position": _position_dict(t.position),
            "confidence": t.confidence,
        }
        for t in result.tables
    ]

    forms = [
        {
            "field_name": f.field_name,
            "value": f.value,
            "position": _position_dict(f.position),
            "confidence": f.confidence,
        }
        for f in result.forms
    ]

    images = [
        {
            "alt_text": i.alt_text,
            "position": _position_dict(i.position),
            "confidence": i.confidence,
        }
        for i in result.images
    ]

    text_blocks = [
        {
            "text": b.text,
            "position": _position_dict(b.position),
            "block_type": b.block_type,
            "confidence": b.confidence,
        }
        for b in result.text_blocks
    ]

    return {
        "document_id": doc_id,
        "source_path": source_path,
        "text": result.full_text,
        "metadata": meta_dict,
        "warnings": result.warnings,
        "tables": tables,
        "forms": forms,
        "images": images,
        "text_blocks": text_blocks,
    }


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def write_results(
    results: list[tuple[str, str, ExtractionResult]],
    output_path: Path,
) -> Path:
    """Write extraction results to a Parquet file.

    Args:
        results: List of (document_id, source_path, ExtractionResult) tuples.
        output_path: Destination Parquet file path.

    Returns:
        The output path written.
    """
    rows = [_result_to_row(doc_id, src, res) for doc_id, src, res in results]

    if not rows:
        # Write empty table with schema
        table = pa.table({f.name: pa.array([], type=f.type) for f in EXTRACTION_SCHEMA}, schema=EXTRACTION_SCHEMA)
    else:
        table = pa.Table.from_pylist(rows, schema=EXTRACTION_SCHEMA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(output_path))
    logger.info("Wrote %d documents to %s", len(rows), output_path)
    return output_path


def read_results(path: Path) -> pa.Table:
    """Read extraction results from a Parquet file."""
    return pq.read_table(str(path), schema=EXTRACTION_SCHEMA)
