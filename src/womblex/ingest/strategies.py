"""Re-export shim for the remaining (non-PDF) extraction strategies.

PDFs route through `extract_pdf_with_plan` (per-page profile + orchestrator)
rather than dedicated strategy classes. This shim exposes only the
non-PDF extractors and the legacy `ImageExtractor` for callers that still
import from `womblex.ingest.strategies`.
"""

from womblex.ingest.strategies_file import (
    DocxExtractor,
    NonTextualExtractor,
    TextExtractor,
)
from womblex.ingest.strategies_scanned import ImageExtractor

__all__ = [
    "DocxExtractor",
    "ImageExtractor",
    "NonTextualExtractor",
    "TextExtractor",
]
