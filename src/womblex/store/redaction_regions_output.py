"""``*.redaction_regions.parquet``: where a source withheld content, one row per region.

``*.redactions.parquet`` says which elements sit on a page with a redaction; this keeps the regions
themselves, so a consumer can tell a withheld region from a gap. One file per batch, joined to
``*.elements.parquet`` on ``(source_hash, page)``. Written whenever redaction detection ran over the
batch (empty when it found none), so a missing file means it did not run.

``bbox_pt`` is in PDF points (72 to the inch), top-left origin, converted from the pixels at the
detection dpi, and ``area`` is its width times height in square points. Unlike ``elements.bbox`` (hence the name) it
is not normalised: a withheld region is measured on the source page.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from womblex.store.output import _BBOX_TYPE, _write_rows

logger = logging.getLogger(__name__)

REDACTION_REGIONS_SUFFIX = ".redaction_regions.parquet"

REDACTION_REGIONS_SCHEMA = pa.schema([
    ("source_hash", pa.string()),
    ("page", pa.int32()),  # 0-based, as elements.page
    ("bbox_pt", _BBOX_TYPE),  # PDF points, top-left origin
    ("area", pa.float32()),  # square points
])


def redaction_regions_path_for(base_path: Path) -> Path:
    """The ``<base>.redaction_regions.parquet`` sibling of a shard base path."""
    return base_path.parent / f"{base_path.stem}{REDACTION_REGIONS_SUFFIX}"


def write_redaction_regions(
    rows: list[dict], base_path: Path, *, metadata: dict[bytes, bytes] | None = None,
) -> Path:
    """Write a batch's region rows (matching :data:`REDACTION_REGIONS_SCHEMA`); none writes an empty file."""
    target = redaction_regions_path_for(base_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(rows, target, REDACTION_REGIONS_SCHEMA, metadata=metadata, role="redaction_regions")
    logger.info("Wrote redaction_regions shard %s: rows=%d", target.name, len(rows))
    return target


def read_redaction_regions(path: Path) -> pa.Table:
    """Read region rows from one shard file or a shard directory."""
    p = Path(path)
    if p.is_dir():
        shards = sorted(p.glob(f"*{REDACTION_REGIONS_SUFFIX}"))
        if not shards:
            return pa.table({f.name: pa.array([], type=f.type) for f in REDACTION_REGIONS_SCHEMA},
                            schema=REDACTION_REGIONS_SCHEMA)
        return pa.concat_tables([pq.read_table(str(s)) for s in shards])
    target = p if p.name.endswith(REDACTION_REGIONS_SUFFIX) else redaction_regions_path_for(p)
    return pq.read_table(str(target))


__all__ = [
    "REDACTION_REGIONS_SCHEMA",
    "REDACTION_REGIONS_SUFFIX",
    "read_redaction_regions",
    "redaction_regions_path_for",
    "write_redaction_regions",
]
