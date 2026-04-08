"""Backend protocols for ingest operations.

Defines type-hinted Protocol classes for pluggable backends: OCR readers,
layout analysers, and image preprocessors.  Strategy implementations depend
on these protocols — not on concrete classes — so backends can be swapped
without touching extraction logic.
"""

from womblex.ingest.interfaces.protocols import (
    LayoutAnalyzer,
    LayoutRegionResult,
    OCRReader,
    OCRRegionResult,
    Preprocessor,
    PreprocessResult,
)

__all__ = [
    "LayoutAnalyzer",
    "LayoutRegionResult",
    "OCRReader",
    "OCRRegionResult",
    "Preprocessor",
    "PreprocessResult",
]
