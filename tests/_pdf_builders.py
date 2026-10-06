"""Build test PDFs on reportlab and read them back through the PDF seam.

Coordinates are top-left in points, as in `ingest.pdf.types.Rect` and the fitz
calls these replace; the y-flip to reportlab's bottom-left happens here only.
Defaults follow fitz's: A4 pages and 11pt Helvetica. `open()` goes through
`open_document` without naming a backend, so the tests follow the default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

from womblex.ingest.pdf import open_document

if TYPE_CHECKING:
    from pathlib import Path

    from PIL.Image import Image

    from womblex.ingest.pdf.types import Document, Rect

Colour = tuple[float, float, float]


class PdfBuilder:
    """Accumulate pages and draws, then write the PDF once on `save()`."""

    def __init__(self, path: Path) -> None:
        self.path = path
        # invariant: no timestamp or random ID, so the bytes are reproducible.
        self._canvas = Canvas(str(path), invariant=1)
        self._height: float | None = None
        self._saved = False

    def page(self, width: float = 595, height: float = 842) -> Self:
        if self._height is not None:
            self._canvas.showPage()
        self._canvas.setPageSize((width, height))
        self._height = height
        return self

    def text(self, x: float, y: float, text: str, *, size: float = 11) -> Self:
        """Draw *text* with its baseline starting at (x, y)."""
        self._canvas.setFillColorRGB(0, 0, 0)
        self._canvas.setFont("Helvetica", size)
        self._canvas.drawString(x, self._flip(y), text)
        return self

    def rect(self, rect: Rect, colour: Colour = (0, 0, 0)) -> Self:
        """A rectangle filled and stroked (1pt) in *colour*."""
        self._canvas.setStrokeColorRGB(*colour)
        self._canvas.setFillColorRGB(*colour)
        self._canvas.rect(rect.x0, self._flip(rect.y1), rect.width, rect.height, stroke=1, fill=1)
        return self

    def image(self, rect: Rect, image: Image) -> Self:
        """Draw *image* inside *rect*, keeping its aspect ratio and centred."""
        self._canvas.drawImage(
            ImageReader(image), rect.x0, self._flip(rect.y1), rect.width, rect.height,
            preserveAspectRatio=True, anchor="c",
        )
        return self

    def save(self) -> Path:
        if not self._saved:
            if self._height is None:
                self.page()
            self._canvas.showPage()
            self._canvas.save()
            self._saved = True
        return self.path

    def open(self) -> Document:
        return open_document(self.save())

    def _flip(self, y: float) -> float:
        if self._height is None:
            raise RuntimeError("call page() before drawing")
        return self._height - y
