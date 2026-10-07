"""Parquet IO for layout regions (``*.layout_regions.parquet``) and the layout fingerprint.

One row per detected region, keyed by ``(source_hash, page)``. ``page`` is
0-based, as on elements. ``bbox`` is normalised 0-1 with a top-left origin, the
element ``BBox`` convention, so a region and an element on the same page share
one coordinate space whatever the render dpi.

A page that was analysed gets at least one row. Where the model found nothing
the row has ``status = 'empty'``; where analysis failed it has
``status = 'error'`` and ``error`` says why. Both carry null geometry, so a page
that found nothing is never read as a page that could not be analysed. A page
the layout step did not select has no row at all, which is also how a page
outside the configured ``layout.page_scope`` reads.

The footer carries a :class:`LayoutFingerprint` (what produced the regions) and
whether redaction consumed them. The same fingerprint rides in the extraction
``*.elements.parquet`` footer, which records what extraction consumed; a
sidecar rerun cannot inherit that from the run stamp, so it carries its own.

No document text is stored; the contract sensitivity is ``none``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq

from womblex.store.output import _BBOX_TYPE, _write_rows
from womblex.store.source_provenance import NAMESPACE

if TYPE_CHECKING:
    from womblex.config import WomblexConfig
    from womblex.ingest.layout_step import PageLayout

logger = logging.getLogger(__name__)

LAYOUT_REGIONS_SUFFIX = ".layout_regions.parquet"

#: Bumped when the sidecar's columns or the box convention change, so a file
#: written under another schema never fingerprint-matches this one.
LAYOUT_SCHEMA_VERSION = "1"

FINGERPRINT_KEY = f"{NAMESPACE}.layout_fingerprint"
REDACTION_CONSUMED_KEY = f"{NAMESPACE}.layout_redaction_consumed"

STATUS_OK = "ok"
STATUS_EMPTY = "empty"
STATUS_ERROR = "error"

_CATEGORY = pa.dictionary(pa.int8(), pa.string())

LAYOUT_REGIONS_SCHEMA = pa.schema([
    ("source_hash", pa.string()),
    ("page", pa.int32()),
    ("region_order", pa.int32()),   # null on a status row
    ("bbox", _BBOX_TYPE),           # null on a status row
    ("label", _CATEGORY),           # the model's own class name
    ("block_type", _CATEGORY),      # the womblex vocabulary
    ("confidence", pa.float32()),   # always present on an ok row
    ("status", _CATEGORY),          # ok | empty | error
    ("error", pa.string()),
])


@dataclass(frozen=True)
class LayoutFingerprint:
    """What produced a set of layout regions.

    Equal fingerprints mean the same model, settings, render and page scope, so
    a rerun that would reproduce them can be skipped. ``model_digest`` is the
    content digest of the model's local files where it has them, and
    ``<distribution>==<version>`` for a plugin model, which Womblex cannot
    digest.
    """

    model: str
    model_digest: str
    options_digest: str
    dpi: int
    page_scope: str
    schema_version: str = LAYOUT_SCHEMA_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str) -> LayoutFingerprint | None:
        """Decode a footer value; ``None`` if it is not a fingerprint."""
        try:
            data = json.loads(raw)
            return cls(
                model=str(data["model"]),
                model_digest=str(data["model_digest"]),
                options_digest=str(data["options_digest"]),
                dpi=int(data["dpi"]),
                page_scope=str(data["page_scope"]),
                schema_version=str(data["schema_version"]),
            )
        except (ValueError, KeyError, TypeError):
            return None

    def footer_metadata(self) -> dict[bytes, bytes]:
        return {FINGERPRINT_KEY.encode(): self.to_json().encode()}


def _model_digest(name: str, options: Mapping[str, Any]) -> str:
    """Digest of the layout model's local files, or its plugin distribution and version."""
    import womblex.ingest.paddle_ocr  # noqa: F401  (registers the built-in)
    from womblex.utils.model_registry import SLOT_LAYOUT, distribution_version, resolve
    from womblex.utils.models import digest_model_path, resolve_local_model_path

    entry = resolve(SLOT_LAYOUT, name)
    if entry.source != "builtin":
        dist, version = distribution_version(entry)
        return f"{dist}=={version}"
    model_dir = options.get("model_dir")
    if model_dir is None:
        from womblex.ingest.layout_onnx import MODEL_DIR

        found = resolve_local_model_path(MODEL_DIR, record=False)
        model_dir = found if isinstance(found, Path) else None
    if model_dir is None or not Path(model_dir).exists():
        return ""
    return digest_model_path(Path(model_dir))


def layout_fingerprint(config: WomblexConfig) -> LayoutFingerprint:
    """The fingerprint *config* would produce: its layout model and settings at the OCR dpi."""
    layout = config.layout
    options = json.dumps(layout.options, sort_keys=True, separators=(",", ":"), default=str)
    return LayoutFingerprint(
        model=layout.model,
        model_digest=_model_digest(layout.model, layout.options),
        options_digest="sha256:" + hashlib.sha256(options.encode()).hexdigest(),
        dpi=config.extraction.ocr.dpi,
        page_scope=layout.page_scope,
    )


def read_footer_layout_fingerprint(
    metadata: Mapping[bytes, bytes] | None,
) -> LayoutFingerprint | None:
    """The fingerprint in a Parquet footer; ``None`` for a file that carries none."""
    raw = (metadata or {}).get(FINGERPRINT_KEY.encode())
    return LayoutFingerprint.from_json(raw.decode(errors="replace")) if raw is not None else None


def read_footer_redaction_consumed(metadata: Mapping[bytes, bytes] | None) -> bool | None:
    """Whether redaction consumed the regions; ``None`` where the footer does not say."""
    raw = (metadata or {}).get(REDACTION_CONSUMED_KEY.encode())
    return None if raw is None else raw == b"true"


def layout_rows(documents: Iterable[tuple[str, Iterable[PageLayout]]]) -> list[dict]:
    """Sidecar rows for ``(source_hash, pages)`` pairs, in page then region order."""
    rows: list[dict] = []
    for source_hash, pages in documents:
        for page in pages:
            base = {"source_hash": source_hash, "page": page.page, "status": page.status}
            if not page.regions:
                rows.append({**base, "error": page.error or None})
                continue
            for order, r in enumerate(page.regions):
                rows.append({
                    **base, "region_order": order,
                    "bbox": {"x": r.bbox.x, "y": r.bbox.y,
                             "width": r.bbox.width, "height": r.bbox.height},
                    "label": r.label, "block_type": r.block_type,
                    "confidence": r.confidence, "error": None,
                })
    return rows


def layout_regions_path_for(base_path: Path) -> Path:
    """The ``<base>.layout_regions.parquet`` sibling of a shard base path."""
    return base_path.parent / f"{base_path.stem}{LAYOUT_REGIONS_SUFFIX}"


def write_layout_regions(
    rows: list[dict],
    output_path: Path,
    fingerprint: LayoutFingerprint,
    *,
    redaction_consumed: bool,
    metadata: dict[bytes, bytes] | None = None,
) -> Path:
    """Write a batch's layout rows (matching :data:`LAYOUT_REGIONS_SCHEMA`).

    ``metadata`` is the shard's provenance and run footer, passed by the writer
    that owns them.
    """
    target = layout_regions_path_for(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    footer = {
        **(metadata or {}),
        **fingerprint.footer_metadata(),
        REDACTION_CONSUMED_KEY.encode(): b"true" if redaction_consumed else b"false",
    }
    _write_rows(rows, target, LAYOUT_REGIONS_SCHEMA, metadata=footer)
    logger.info("Wrote layout_regions shard %s: rows=%d", target.name, len(rows))
    return target


def read_layout_regions(path: Path) -> pa.Table:
    """Read layout regions from one shard file or a shard directory."""
    p = Path(path)
    if p.is_dir():
        shards = sorted(p.glob(f"*{LAYOUT_REGIONS_SUFFIX}"))
        if not shards:
            return pa.table(
                {f.name: pa.array([], type=f.type) for f in LAYOUT_REGIONS_SCHEMA},
                schema=LAYOUT_REGIONS_SCHEMA,
            )
        return pa.concat_tables([_read_shard(s) for s in shards])
    return _read_shard(p if p.name.endswith(LAYOUT_REGIONS_SUFFIX) else layout_regions_path_for(p))


def _read_shard(path: Path) -> pa.Table:
    raw = pq.read_table(str(path))
    missing = [f.name for f in LAYOUT_REGIONS_SCHEMA if f.name not in raw.schema.names]
    if missing:
        raise ValueError(
            f"layout shard {path} missing columns {missing}; schema bump without compat shim?"
        )
    return raw.select([f.name for f in LAYOUT_REGIONS_SCHEMA]).cast(LAYOUT_REGIONS_SCHEMA)


__all__ = [
    "FINGERPRINT_KEY",
    "LAYOUT_REGIONS_SCHEMA",
    "LAYOUT_REGIONS_SUFFIX",
    "LAYOUT_SCHEMA_VERSION",
    "REDACTION_CONSUMED_KEY",
    "STATUS_EMPTY",
    "STATUS_ERROR",
    "STATUS_OK",
    "LayoutFingerprint",
    "layout_fingerprint",
    "layout_regions_path_for",
    "layout_rows",
    "read_footer_layout_fingerprint",
    "read_footer_redaction_consumed",
    "read_layout_regions",
    "write_layout_regions",
]
