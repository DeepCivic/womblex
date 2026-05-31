"""Result models shared across operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from womblex.ingest.detect import DocumentProfile
from womblex.ingest.extract import ExtractionResult
from womblex.process.chunker import TextChunk


@dataclass
class DocumentResult:
    """Processing result for a single document."""

    path: Path
    doc_id: str
    profile: DocumentProfile | None = None
    extraction: ExtractionResult | None = None
    chunks: list[TextChunk] = field(default_factory=list)
    enrichment: Any = None  # EnrichmentResult when isaacus extra installed
    graph: Any = None  # DocumentGraph when isaacus extra installed
    error: str | None = None
    status: str = "pending"


@dataclass
class BatchResult:
    """Processing result for a batch of documents."""

    results: list[DocumentResult] = field(default_factory=list)

    @property
    def succeeded(self) -> int:
        return sum(1 for r in self.results if r.status == "completed")

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == "error")

    @property
    def enriched(self) -> int:
        return sum(1 for r in self.results if r.enrichment is not None)
