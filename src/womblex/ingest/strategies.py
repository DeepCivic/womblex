"""Extraction strategy implementations for each document type.

Strategies are split across focused modules by document type family:

- ``strategies_native``  — native text-layer PDFs (narrative, structured)
- ``strategies_scanned`` — OCR-dependent types (scanned, hybrid, image)
- ``strategies_file``    — non-PDF formats (DOCX, plain text, non-textual)
- ``spreadsheet``        — CSV/XLSX (already separate)

This module re-exports all strategy classes so existing imports continue
to work unchanged.
"""

# Native text-layer strategies
from womblex.ingest.strategies_native import (  # noqa: F401
    NativeNarrativeExtractor,
    NativeWithStructuredExtractor,
    StructuredExtractor,
)

# OCR-dependent strategies
from womblex.ingest.strategies_scanned import (  # noqa: F401
    HybridExtractor,
    ImageExtractor,
    ScannedHandwrittenExtractor,
    ScannedMachinewrittenExtractor,
    ScannedMixedExtractor,
)

# File-based strategies
from womblex.ingest.strategies_file import (  # noqa: F401
    DocxExtractor,
    NonTextualExtractor,
    TextExtractor,
)

__all__ = [
    "NativeNarrativeExtractor",
    "NativeWithStructuredExtractor",
    "StructuredExtractor",
    "ScannedMachinewrittenExtractor",
    "ScannedHandwrittenExtractor",
    "ScannedMixedExtractor",
    "HybridExtractor",
    "ImageExtractor",
    "DocxExtractor",
    "TextExtractor",
    "NonTextualExtractor",
]
