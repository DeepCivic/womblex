"""The PDF seam: one entry point, one backend chosen at call time.

`types.py` holds the backend-neutral vocabulary the extractors are written
against. A backend lives in its own `_*.py` and is imported only when a
document is opened, so `import womblex` — and importing this package — loads no
PDF library.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pathlib import Path

    from womblex.ingest.pdf.types import Document

#: Backend name to the module providing ``open_document(path)``.
_BACKENDS = {"fitz": "womblex.ingest.pdf._fitz", "pdfium": "womblex.ingest.pdf._pdfium_doc"}

#: The backend used when a caller names none: the permissive one. `fitz` stays
#: selectable until F2 in `docs/plan-permissive-deps.md` deletes it.
DEFAULT_BACKEND = "pdfium"


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


__all__ = ["DEFAULT_BACKEND", "open_document"]
