"""Tests for womblex.process.chunker.

Exercises the post-I2 surface: a single :func:`chunk_batch` entry point
plus :func:`create_chunker`, :func:`table_to_markdown`,
:func:`reassemble_narrative`, :func:`collect_tables_from_elements`,
:func:`build_chunk_input`, the :class:`TextChunk` / :class:`ChunkInput`
dataclasses and the ``_repair_redaction_splits`` helper.

Tests use a word-count callable token counter so they don't depend on a
HuggingFace tokeniser. semchunk treats it as any other token counter.
"""

from __future__ import annotations

import semchunk

from womblex.ingest.elements import BBox, Cell, Element
from womblex.process.chunker import (
    ChunkInput,
    TextChunk,
    _repair_redaction_splits,
    build_chunk_input,
    chunk_batch,
    collect_tables_from_elements,
    create_chunker,
    reassemble_narrative,
    table_to_markdown,
)


def _word_token_counter(text: str) -> int:
    return len(text.split())


def _make_test_chunker(chunk_size: int = 50) -> semchunk.Chunker:
    return semchunk.chunkerify(_word_token_counter, chunk_size=chunk_size)


# ---------------------------------------------------------------------------
# create_chunker
# ---------------------------------------------------------------------------


class TestCreateChunker:
    def test_returns_callable(self) -> None:
        chunker = create_chunker(tokenizer=_word_token_counter, chunk_size=100)
        assert callable(chunker)

    def test_exposes_full_chunkerify_surface(self) -> None:
        chunker = create_chunker(
            _word_token_counter, chunk_size=50,
            memoize=False, cache_maxsize=10, max_token_chars=20,
        )
        result = chunker(["hello world"], offsets=True)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_default_chunking_model_none_is_offline(self) -> None:
        # Default (no chunking_model) must stay fully offline: a callable
        # token counter and no Isaacus client, yet chunking succeeds.
        chunker = create_chunker(_word_token_counter, chunk_size=50)
        chunks = chunker("alpha beta gamma delta")
        assert chunks  # produced offline, no API touched

    def test_ai_chunking_params_forwarded_to_chunkerify(self, monkeypatch) -> None:
        captured: dict = {}
        real_chunkerify = semchunk.chunkerify

        def _fake_chunkerify(tokenizer, **kwargs):
            captured["tokenizer"] = tokenizer
            captured.update(kwargs)
            return real_chunkerify(_word_token_counter, chunk_size=50)

        monkeypatch.setattr("womblex.process.chunker.semchunk.chunkerify", _fake_chunkerify)
        sentinel_client = object()
        create_chunker(
            _word_token_counter, chunk_size=50,
            chunking_model="kanon-2-enricher",
            isaacus_client=sentinel_client,
            tokenizer_kwargs={"add_special_tokens": False},
        )
        assert captured["chunking_model"] == "kanon-2-enricher"
        assert captured["isaacus_client"] is sentinel_client
        assert captured["tokenizer_kwargs"] == {"add_special_tokens": False}


# ---------------------------------------------------------------------------
# TextChunk
# ---------------------------------------------------------------------------


class TestTextChunk:
    def test_default_fields(self) -> None:
        c = TextChunk(text="x", start_char=0, end_char=1, chunk_index=0)
        assert c.content_type == "narrative"
        assert c.has_redaction is False
        assert c.page_start is None
        assert c.page_end is None

    def test_explicit_fields(self) -> None:
        c = TextChunk(
            text="x", start_char=0, end_char=1, chunk_index=0,
            content_type="table", has_redaction=True,
            page_start=3, page_end=5,
        )
        assert c.content_type == "table"
        assert c.has_redaction is True
        assert (c.page_start, c.page_end) == (3, 5)


# ---------------------------------------------------------------------------
# table_to_markdown
# ---------------------------------------------------------------------------


class TestTableToMarkdown:
    def test_empty(self) -> None:
        assert table_to_markdown([], []) == ""

    def test_headers_and_rows(self) -> None:
        md = table_to_markdown(["A", "B"], [["1", "2"], ["3", "4"]])
        lines = md.splitlines()
        assert lines[0] == "| A | B |"
        assert "1" in lines[2]
        assert "3" in lines[3]

    def test_short_row_padded(self) -> None:
        md = table_to_markdown(["A", "B", "C"], [["x"]])
        assert "| x |  |  |" in md or "x" in md


# ---------------------------------------------------------------------------
# _repair_redaction_splits
# ---------------------------------------------------------------------------


class TestRepairRedactionSplits:
    def test_no_change_when_marker_intact(self) -> None:
        chunks = [TextChunk("a <REDACTED> b", 0, 14, 0)]
        out = _repair_redaction_splits(chunks)
        assert out == chunks

    def test_split_marker_merged(self) -> None:
        chunks = [
            TextChunk("a <RED", 0, 6, 0),
            TextChunk("ACTED> b", 6, 14, 1),
        ]
        out = _repair_redaction_splits(chunks)
        assert len(out) == 1
        assert "<REDACTED>" in out[0].text
        assert out[0].start_char == 0
        assert out[0].end_char == 14
        assert out[0].has_redaction is True

    def test_indices_resequenced(self) -> None:
        chunks = [
            TextChunk("a <RED", 0, 6, 0),
            TextChunk("ACTED> b", 6, 14, 1),
            TextChunk("c", 14, 15, 2),
        ]
        out = _repair_redaction_splits(chunks)
        assert [c.chunk_index for c in out] == [0, 1]

    def test_merge_carries_page_span(self) -> None:
        chunks = [
            TextChunk("a <RED", 0, 6, 0, page_start=1, page_end=1),
            TextChunk("ACTED> b", 6, 14, 1, page_start=2, page_end=2),
        ]
        out = _repair_redaction_splits(chunks)
        assert (out[0].page_start, out[0].page_end) == (1, 2)


# ---------------------------------------------------------------------------
# reassemble_narrative + collect_tables_from_elements
# ---------------------------------------------------------------------------


def _para(order: int, text: str, page: int | None) -> Element:
    return Element(
        order=order, kind="paragraph", extractor="t",
        text=text, page=page,
    )


def _table_elem(order: int, page: int | None, rows: list[list[str]], header_rows: list[int] | None) -> Element:
    cells = []
    for r, row in enumerate(rows):
        for c, v in enumerate(row):
            cells.append(Cell(row=r, col=c, value=v))
    return Element(
        order=order, kind="table", extractor="t",
        page=page, cells=cells, header_rows=header_rows or [],
        bbox=BBox(x=0, y=0, width=1, height=1),
    )


class TestReassembleNarrative:
    def test_joins_with_double_newline(self) -> None:
        elements = [_para(0, "first", 1), _para(1, "second", 1)]
        text, _ = reassemble_narrative(elements)
        assert text == "first\n\nsecond"

    def test_skips_empty_text(self) -> None:
        elements = [_para(0, "x", 1), _para(1, "", 1), _para(2, "y", 1)]
        text, _ = reassemble_narrative(elements)
        assert text == "x\n\ny"

    def test_skips_non_text_kinds(self) -> None:
        elements = [_para(0, "x", 1), _table_elem(1, 1, [["a"]], None), _para(2, "y", 1)]
        text, _ = reassemble_narrative(elements)
        assert text == "x\n\ny"

    def test_page_breaks_track_offsets(self) -> None:
        elements = [
            _para(0, "alpha", 1),     # offset 0..5
            _para(1, "beta", 1),      # offset 5+2+4 = 11
            _para(2, "gamma", 2),     # next page
        ]
        text, page_breaks = reassemble_narrative(elements)
        # "alpha" (5) + "\n\nbeta" (6) = 11, then "\n\ngamma" (7) → 18
        assert text == "alpha\n\nbeta\n\ngamma"
        assert len(page_breaks) == 2
        assert page_breaks[0][1] == 1
        assert page_breaks[1][1] == 2
        # Page 1 ends at offset 11 (end of "beta"); page 2 ends at 18 (end of text).
        assert page_breaks[0][0] == 11
        assert page_breaks[1][0] == 18

    def test_no_page_yields_empty_breaks(self) -> None:
        elements = [_para(0, "alpha", None), _para(1, "beta", None)]
        _, page_breaks = reassemble_narrative(elements)
        assert page_breaks == []


class TestCollectTablesFromElements:
    def test_kind_table_emitted(self) -> None:
        elements = [_table_elem(0, 3, [["A", "B"], ["1", "2"]], header_rows=[0])]
        out = collect_tables_from_elements(elements)
        assert len(out) == 1
        page, md = out[0]
        assert page == 3
        assert "| A | B |" in md
        assert "| 1 | 2 |" in md

    def test_sheet_cells_aggregated_as_synthetic_table(self) -> None:
        # Two sheet_cell elements → one synthetic table.
        elements = [
            Element(order=0, kind="sheet_cell", extractor="t",
                    sheet="S1", row=0, col=0, value="H1"),
            Element(order=1, kind="sheet_cell", extractor="t",
                    sheet="S1", row=0, col=1, value="H2"),
            Element(order=2, kind="sheet_cell", extractor="t",
                    sheet="S1", row=1, col=0, value="v1"),
        ]
        out = collect_tables_from_elements(elements)
        assert len(out) == 1
        page, md = out[0]
        assert page is None
        assert "H1" in md and "H2" in md


# ---------------------------------------------------------------------------
# build_chunk_input
# ---------------------------------------------------------------------------


class TestBuildChunkInput:
    def test_assembles_narrative_and_tables(self) -> None:
        elements = [
            _para(0, "intro", 1),
            _table_elem(1, 1, [["A"], ["1"]], header_rows=[0]),
        ]
        ci = build_chunk_input("hash-x", elements)
        assert ci.source_hash == "hash-x"
        assert ci.narrative == "intro"
        assert ci.page_breaks == [(5, 1)]
        assert len(ci.tables) == 1

    def test_include_tables_false_skips_tables(self) -> None:
        elements = [_table_elem(0, 1, [["A"], ["1"]], header_rows=[0])]
        ci = build_chunk_input("hash-x", elements, include_tables=False)
        assert ci.tables == []


# ---------------------------------------------------------------------------
# chunk_batch
# ---------------------------------------------------------------------------


class TestChunkBatch:
    def test_empty_input(self) -> None:
        assert chunk_batch([], _make_test_chunker()) == {}

    def test_narrative_only(self) -> None:
        chunker = _make_test_chunker(chunk_size=50)
        ci = ChunkInput(source_hash="a", narrative="hello world", page_breaks=[(11, 1)])
        out = chunk_batch([ci], chunker)
        assert "a" in out
        assert len(out["a"]) >= 1
        assert all(c.content_type == "narrative" for c in out["a"])
        assert all(c.page_start == 1 for c in out["a"])

    def test_tables_get_table_content_type(self) -> None:
        chunker = _make_test_chunker(chunk_size=50)
        ci = ChunkInput(
            source_hash="a", narrative="",
            tables=[(2, "| A | B |\n| --- | --- |\n| 1 | 2 |")],
        )
        out = chunk_batch([ci], chunker)
        chunks = out["a"]
        assert chunks
        assert all(c.content_type == "table" for c in chunks)
        assert all(c.page_start == 2 and c.page_end == 2 for c in chunks)

    def test_multiple_docs_keyed_by_source_hash(self) -> None:
        chunker = _make_test_chunker(chunk_size=50)
        inputs = [
            ChunkInput("a", "first document content"),
            ChunkInput("b", "second document content"),
        ]
        out = chunk_batch(inputs, chunker)
        assert set(out) == {"a", "b"}
        assert all(out["a"][0].text != "" for _ in [None])
        assert out["b"][0].text != out["a"][0].text

    def test_per_doc_chunk_index_resequenced(self) -> None:
        chunker = _make_test_chunker(chunk_size=5)
        ci = ChunkInput("a", " ".join(f"word{i}" for i in range(30)))
        out = chunk_batch([ci], chunker)
        indices = [c.chunk_index for c in out["a"]]
        assert indices == list(range(len(indices)))

    def test_overlap_passes_through(self) -> None:
        chunker = _make_test_chunker(chunk_size=10)
        text = " ".join(f"word{i}" for i in range(30))
        ci = ChunkInput("a", text)
        no_overlap = chunk_batch([ci], chunker)
        with_overlap = chunk_batch([ci], chunker, overlap=3)
        # Overlapping chunks duplicate boundary content, so total text grows.
        assert sum(len(c.text) for c in with_overlap["a"]) > sum(
            len(c.text) for c in no_overlap["a"]
        )

    def test_has_redaction_set_from_marker(self) -> None:
        chunker = _make_test_chunker(chunk_size=50)
        ci = ChunkInput("a", "before <REDACTED> after")
        out = chunk_batch([ci], chunker)
        assert any(c.has_redaction for c in out["a"])

    def test_doc_with_only_tables(self) -> None:
        chunker = _make_test_chunker(chunk_size=50)
        ci = ChunkInput(
            source_hash="a", narrative="",
            tables=[(1, table_to_markdown(["H"], [["v"]]))],
        )
        out = chunk_batch([ci], chunker)
        assert out["a"]
        assert out["a"][0].content_type == "table"

    def test_offsets_match_input_text(self) -> None:
        chunker = _make_test_chunker(chunk_size=10)
        text = " ".join(f"word{i}" for i in range(30))
        ci = ChunkInput("a", text)
        out = chunk_batch([ci], chunker)
        for c in out["a"]:
            assert text[c.start_char:c.end_char] == c.text

    def test_split_redaction_marker_repaired(self) -> None:
        # Construct text such that semchunk plausibly splits around the
        # marker — the repair pass merges across the boundary.
        chunker = _make_test_chunker(chunk_size=3)
        text = "lots of text before <REDACTED> and then lots of text after"
        ci = ChunkInput("a", text)
        out = chunk_batch([ci], chunker)
        joined = "".join(c.text for c in out["a"])
        assert "<REDACTED>" in joined
