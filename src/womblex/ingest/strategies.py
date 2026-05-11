"""Re-export shim for the remaining (non-PDF) extraction strategies.

PDFs route through `extract_pdf_with_plan` (per-page profile + orchestrator)
rather than dedicated strategy classes. This shim exposes only the
non-PDF extractors and the legacy `ImageExtractor` for callers that still
import from `womblex.ingest.strategies`.
"""

from womblex.ingest.strategies_scanned import ImageExtractor  # noqa: F401
from womblex.ingest.strategies_file import (  # noqa: F401
    DocxExtractor,
    NonTextualExtractor,
    TextExtractor,
)

__all__ = [
    "ImageExtractor",
    "DocxExtractor",
    "TextExtractor",
    "NonTextualExtractor",
]
