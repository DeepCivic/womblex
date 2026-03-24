"""Tests for enrichment Parquet output.

Requires the isaacus extra: pip install womblex[isaacus]
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

# Skip all tests in this module if isaacus extra not installed
pytest.importorskip("isaacus", reason="isaacus extra not installed")

from womblex.analyse.graph import build_document_graph
from womblex.analyse.models import (
    EnrichmentResult,
    Location,
    Person,
    Segment,
    Span,
    Term,
)
from womblex.process.chunker import TextChunk
from womblex.store.enrichment_output import (
    ENRICHMENT_META_SCHEMA,
    ENTITY_SCHEMA,
    GRAPH_EDGE_SCHEMA,
    write_enrichment_metadata,
    write_entity_mentions,
    write_graph_edges,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_text() -> str:
    return "The Department of Home Affairs issued a compliance notice."


def _make_chunks(text: str) -> list[TextChunk]:
    return [TextChunk(text=text, start_char=0, end_char=len(text), chunk_index=0)]


def _make_enrichment(text: str) -> EnrichmentResult:
    return EnrichmentResult(
        text=text,
        type="other",
        jurisdiction="AU",
        title=Span(start=0, end=3),
        segments=[
            Segment(
                id="seg:0",
                kind="unit",
                type="paragraph",
                category="main",
                span=Span(start=0, end=len(text)),
                level=0,
            ),
        ],
        locations=[
            Location(
                id="loc:0",
                name=Span(start=4, end=14),  # "Department"
                type="city",
                mentions=[Span(start=4, end=14)],
            ),
        ],
        persons=[
            Person(
                id="per:0",
                name=Span(start=4, end=31),  # "Department of Home Affairs"
                type="politic",
                role="other",
                mentions=[Span(start=4, end=31)],
            ),
        ],
        terms=[
            Term(
                id="term:0",
                name=Span(start=40, end=50),  # "compliance"
                meaning=Span(start=40, end=57),  # "compliance notice"
                mentions=[Span(start=40, end=50)],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Tests: entity mentions
# ---------------------------------------------------------------------------


class TestWriteEntityMentions:
    def test_write_entities(self, tmp_path: Path) -> None:
        text = _make_text()
        enrichment = _make_enrichment(text)
        chunks = _make_chunks(text)
        output = tmp_path / "entities.parquet"

        write_entity_mentions([(
            "doc1", enrichment, chunks,  # type: ignore[arg-type]
        )], output)

        assert output.exists()
        table = pq.read_table(str(output))
        assert table.num_rows == 3  # 1 person + 1 location + 1 term mention

        # Check schema
        for field in ENTITY_SCHEMA:
            assert field.name in table.column_names

    def test_write_entities_empty(self, tmp_path: Path) -> None:
        output = tmp_path / "entities_empty.parquet"
        write_entity_mentions([], output)
        assert output.exists()
        table = pq.read_table(str(output))
        assert table.num_rows == 0

    def test_entity_content(self, tmp_path: Path) -> None:
        text = _make_text()
        enrichment = _make_enrichment(text)
        output = tmp_path / "entities.parquet"

        write_entity_mentions([(
            "doc1", enrichment, None,
        )], output)

        table = pq.read_table(str(output))
        doc_ids = table.column("document_id").to_pylist()
        assert all(d == "doc1" for d in doc_ids)

        labels = table.column("entity_label").to_pylist()
        assert "person" in labels
        assert "location" in labels
        assert "term" in labels

    def test_chunk_mapping(self, tmp_path: Path) -> None:
        text = _make_text()
        enrichment = _make_enrichment(text)
        chunks = _make_chunks(text)
        output = tmp_path / "entities.parquet"

        write_entity_mentions([(
            "doc1", enrichment, chunks,  # type: ignore[arg-type]
        )], output)

        table = pq.read_table(str(output))
        chunk_indices = table.column("chunk_index").to_pylist()
        # All mentions overlap with chunk 0
        assert all(ci == 0 for ci in chunk_indices)


# ---------------------------------------------------------------------------
# Tests: graph edges
# ---------------------------------------------------------------------------


class TestWriteGraphEdges:
    def test_write_edges(self, tmp_path: Path) -> None:
        text = _make_text()
        enrichment = _make_enrichment(text)
        graph = build_document_graph("doc1", enrichment)
        output = tmp_path / "edges.parquet"

        write_graph_edges([("doc1", graph)], output)

        assert output.exists()
        table = pq.read_table(str(output))
        assert table.num_rows > 0

        for field in GRAPH_EDGE_SCHEMA:
            assert field.name in table.column_names

    def test_write_edges_empty(self, tmp_path: Path) -> None:
        output = tmp_path / "edges_empty.parquet"
        write_graph_edges([], output)
        assert output.exists()
        table = pq.read_table(str(output))
        assert table.num_rows == 0

    def test_edge_content(self, tmp_path: Path) -> None:
        text = _make_text()
        enrichment = _make_enrichment(text)
        graph = build_document_graph("doc1", enrichment)
        output = tmp_path / "edges.parquet"

        write_graph_edges([("doc1", graph)], output)

        table = pq.read_table(str(output))
        relations = table.column("relation").to_pylist()
        assert "contains" in relations


# ---------------------------------------------------------------------------
# Tests: enrichment metadata
# ---------------------------------------------------------------------------


class TestWriteEnrichmentMetadata:
    def test_write_metadata(self, tmp_path: Path) -> None:
        text = _make_text()
        enrichment = _make_enrichment(text)
        output = tmp_path / "meta.parquet"

        write_enrichment_metadata([("doc1", enrichment)], output)

        assert output.exists()
        table = pq.read_table(str(output))
        assert table.num_rows == 1

        for field in ENRICHMENT_META_SCHEMA:
            assert field.name in table.column_names

    def test_metadata_values(self, tmp_path: Path) -> None:
        text = _make_text()
        enrichment = _make_enrichment(text)
        output = tmp_path / "meta.parquet"

        write_enrichment_metadata([("doc1", enrichment)], output)

        table = pq.read_table(str(output))
        row = table.to_pydict()
        assert row["document_id"][0] == "doc1"
        assert row["doc_type_enriched"][0] == "other"
        assert row["jurisdiction"][0] == "AU"
        assert row["segment_count"][0] == 1
        assert row["person_count"][0] == 1
        assert row["location_count"][0] == 1
        assert row["term_count"][0] == 1

    def test_write_metadata_empty(self, tmp_path: Path) -> None:
        output = tmp_path / "meta_empty.parquet"
        write_enrichment_metadata([], output)
        assert output.exists()
        table = pq.read_table(str(output))
        assert table.num_rows == 0
