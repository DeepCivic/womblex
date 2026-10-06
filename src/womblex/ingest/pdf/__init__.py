"""The PDF seam: one entry point, one backend chosen at call time.

`types.py` holds the backend-neutral vocabulary the extractors are written
against. A backend lives in its own `_*.py` and is imported only when a
document is opened, so `import womblex` — and importing this package — loads no
PDF library.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from pathlib import Path

    from womblex.ingest.pdf.types import Document

#: Backend name to the module providing ``open_document(path)``.
_BACKENDS = {"fitz": "womblex.ingest.pdf._fitz"}

#: The backend used when a caller names none. F1 in
#: `docs/plan-permissive-deps.md` flips this to the permissive backend.
DEFAULT_BACKEND = "fitz"


def open_document(path: Path, *, backend: str | None = None) -> Document:
    """Open *path* as a `Document`, loading the backend on the way.

    An unknown name lists the known ones rather than failing on the import,
    the same way `utils/model_registry` reports an unknown model.
    """
    name = backend or DEFAULT_BACKEND
    module = _BACKENDS.get(name)
    if module is None:
        known = ", ".join(sorted(_BACKENDS))
        raise ValueError(f"unknown PDF backend {name!r}; known backends: {known}")
    return cast("Document", import_module(module).open_document(path))


def native(obj: object) -> Any:
    """The backend's own page or document, for callees not yet on the seam.

    Transitional: P3a and P3b port `page_profile`, `strategies_scanned`,
    `spreadsheet_print` and the rest, and this goes with the last of them.
    """
    return obj.native  # type: ignore[attr-defined]


__all__ = ["DEFAULT_BACKEND", "native", "open_document"]
