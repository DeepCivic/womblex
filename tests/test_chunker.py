"""Tests for womblex.process.chunker — text chunking.

Tests exercise semchunk directly (no mocks). Because the production
tokeniser (isaacus/kanon-2-tokenizer) requires network access and
the ``transformers`` library, tests use a simple word-based token counter
passed as a callable — which is a supported semchunk interface.

The chunking logic itself is identical regardless of token counter.
"""

import pytest
import semchunk

from womblex.process.chunker import (
    TextChunk,
    _repair_redaction_splits,
    chunk_document,
    chunk_text,
    chunk_texts_batch,
    create_chunker,
    table_to_markdown,
)


def _word_token_counter(text: str) -> int:
    """Simple word-count token counter for testing."""
    return len(text.split())


def _make_test_chunker(chunk_size: int = 50) -> semchunk.Chunker:
    """Create a chunker using a word-based token counter (no network needed)."""
    return semchunk.chunkerify(_word_token_counter, chunk_size=chunk_size)


# ---------------------------------------------------------------------------
# create_chunker
# ---------------------------------------------------------------------------


class TestCreateChunker:
    def test_returns_callable_with_function(self) -> None:
        chunker = create_chunker(tokenizer=_word_token_counter, chunk_size=100)
        assert callable(chunker)

    def test_custom_chunk_size(self) -> None:
        chunker = _make_test_chunker(chunk_size=20)
        assert callable(chunker)

    def test_chunker_produces_output(self) -> None:
        chunker = _make_test_chunker(chunk_size=10)
        result = chunker("Hello world this is a test sentence.")
        assert isinstance(result, list)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------


class TestChunkText:
    @pytest.fixture(autouse=True)
    def _setup_chunker(self) -> None:
        self.chunker = _make_test_chunker(chunk_size=50)

    def test_empty_text_returns_empty(self) -> None:
        result = chunk_text("", self.chunker)
        assert result == []

    def test_whitespace_only_returns_empty(self) -> None:
        result = chunk_text("   \n\n  ", self.chunker)
        assert result == []

    def test_short_text_single_chunk(self) -> None:
        text = "Hello world."
        result = chunk_text(text, self.chunker)
        assert len(result) == 1
        assert result[0].text == text
        assert result[0].chunk_index == 0

    def test_long_text_multiple_chunks(self) -> None:
        text = " ".join(["The quick brown fox jumps over the lazy dog."] * 50)
        result = chunk_text(text, self.chunker)
        assert len(result) > 1

    def test_chunks_are_text_chunk_instances(self) -> None:
        text = "A simple test sentence."
        result = chunk_text(text, self.chunker)
        for chunk in result:
            assert isinstance(chunk, TextChunk)

    def test_chunk_indices_are_sequential(self) -> None:
        text = " ".join(["Word"] * 200)
        result = chunk_text(text, self.chunker)
        indices = [c.chunk_index for c in result]
        assert indices == list(range(len(result)))

    def test_chunk_offsets_are_non_negative(self) -> None:
        text = " ".join(["Testing offset tracking."] * 30)
        result = chunk_text(text, self.chunker)
        for chunk in result:
            assert chunk.start_char >= 0
            assert chunk.end_char >= chunk.start_char

    def test_chunk_text_matches_offset(self) -> None:
        text = " ".join(["Offset validation sentence."] * 30)
        result = chunk_text(text, self.chunker)
        for chunk in result:
            extracted = text[chunk.start_char : chunk.end_char]
            assert extracted == chunk.text

    def test_all_text_is_covered(self) -> None:
        """Concatenating all chunks should reconstruct the original text (approximately)."""
        text = "First sentence here. Second sentence here. Third sentence here."
        result = chunk_text(text, self.chunker)
        reconstructed = "".join(c.text for c in result)
        for word in text.split():
            assert word in reconstructed or word.rstrip(".") in reconstructed

    def test_small_chunk_size(self) -> None:
        """With very small chunk size, text should split into many pieces."""
        chunker = _make_test_chunker(chunk_size=5)
        text = "One two three four five six seven eight nine ten eleven twelve."
        result = chunk_text(text, chunker)
        assert len(result) >= 2
        for chunk in result:
            word_count = len(chunk.text.split())
            assert word_count <= 6  # allow a bit of slack

    def test_default_content_type_is_narrative(self) -> None:
        result = chunk_text("Some text.", self.chunker)
        assert all(c.content_type == "narrative" for c in result)

    def test_custom_content_type(self) -> None:
        result = chunk_text("Table data here.", self.chunker, content_type="table")
        assert all(c.content_type == "table" for c in result)


# ---------------------------------------------------------------------------
# TextChunk dataclass
# ---------------------------------------------------------------------------


class TestTextChunk:
    def test_fields(self) -> None:
        chunk = TextChunk(text="hello", start_char=0, end_char=5, chunk_index=0)
        assert chunk.text == "hello"
        assert chunk.start_char == 0
        assert chunk.end_char == 5
        assert chunk.chunk_index == 0

    def test_default_content_type(self) -> None:
        chunk = TextChunk(text="x", start_char=0, end_char=1, chunk_index=0)
        assert chunk.content_type == "narrative"

    def test_explicit_content_type(self) -> None:
        chunk = TextChunk(
            text="x", start_char=0, end_char=1, chunk_index=0, content_type="table"
        )
        assert chunk.content_type == "table"


# ---------------------------------------------------------------------------
# table_to_markdown
# ---------------------------------------------------------------------------


class TestTableToMarkdown:
    def test_empty_table(self) -> None:
        assert table_to_markdown([], []) == ""

    def test_headers_only(self) -> None:
        md = table_to_markdown(["Name", "Age"], [])
        assert "| Name | Age |" in md
        assert "| --- | --- |" in md

    def test_headers_and_rows(self) -> None:
        md = table_to_markdown(["Name", "Age"], [["Alice", "30"], ["Bob", "25"]])
        lines = md.strip().split("\n")
        assert len(lines) == 4  # header + separator + 2 rows
        assert "Alice" in lines[2]
        assert "Bob" in lines[3]

    def test_short_row_is_padded(self) -> None:
        md = table_to_markdown(["A", "B", "C"], [["only_one"]])
        # Should not raise; short row gets padded
        assert "only_one" in md

    def test_no_headers_with_rows(self) -> None:
        md = table_to_markdown([], [["x", "y"]])
        assert md != ""
        assert "| x | y |" in md


# ---------------------------------------------------------------------------
# _repair_redaction_splits
# ---------------------------------------------------------------------------


class TestRepairRedactionSplits:
    def test_no_split_no_change(self) -> None:
        chunks = [
            TextChunk(text="before [REDACTED] after", start_char=0, end_char=23, chunk_index=0),
        ]
        result = _repair_redaction_splits(chunks)
        assert len(result) == 1
        assert result[0].text == chunks[0].text

    def test_split_marker_is_merged(self) -> None:
        chunks = [
            TextChunk(text="some text [REDAC", start_char=0, end_char=16, chunk_index=0),
            TextChunk(text="TED] more text", start_char=16, end_char=30, chunk_index=1),
        ]
        result = _repair_redaction_splits(chunks)
        assert len(result) == 1
        assert "[REDACTED]" in result[0].text
        assert result[0].start_char == 0
        assert result[0].end_char == 30

    def test_single_char_split(self) -> None:
        chunks = [
            TextChunk(text="text [", start_char=0, end_char=6, chunk_index=0),
            TextChunk(text="REDACTED] end", start_char=6, end_char=19, chunk_index=1),
        ]
        result = _repair_redaction_splits(chunks)
        assert len(result) == 1
        assert "[REDACTED]" in result[0].text

    def test_no_redaction_unchanged(self) -> None:
        chunks = [
            TextChunk(text="chunk one", start_char=0, end_char=9, chunk_index=0),
            TextChunk(text="chunk two", start_char=10, end_char=19, chunk_index=1),
        ]
        result = _repair_redaction_splits(chunks)
        assert len(result) == 2

    def test_empty_list(self) -> None:
        assert _repair_redaction_splits([]) == []

    def test_indices_resequenced_after_merge(self) -> None:
        chunks = [
            TextChunk(text="a [REDAC", start_char=0, end_char=8, chunk_index=0),
            TextChunk(text="TED] b", start_char=8, end_char=14, chunk_index=1),
            TextChunk(text="c", start_char=15, end_char=16, chunk_index=2),
        ]
        result = _repair_redaction_splits(chunks)
        assert len(result) == 2
        assert result[0].chunk_index == 0
        assert result[1].chunk_index == 1


# ---------------------------------------------------------------------------
# chunk_document
# ---------------------------------------------------------------------------


class _FakeTable:
    """Minimal stand-in for TableData with headers and rows attributes."""

    def __init__(self, headers: list[str], rows: list[list[str]]) -> None:
        self.headers = headers
        self.rows = rows


class TestChunkDocument:
    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.chunker = _make_test_chunker(chunk_size=50)

    def test_narrative_only(self) -> None:
        text = " ".join(["Government document narrative text."] * 30)
        result = chunk_document(text, self.chunker)
        assert len(result) > 0
        assert all(c.content_type == "narrative" for c in result)

    def test_with_tables(self) -> None:
        text = "Some narrative text here."
        tables = [_FakeTable(["Name", "Age"], [["Alice", "30"], ["Bob", "25"]])]
        result = chunk_document(text, self.chunker, tables=tables)

        narrative = [c for c in result if c.content_type == "narrative"]
        table_chunks = [c for c in result if c.content_type == "table"]

        assert len(narrative) >= 1
        assert len(table_chunks) >= 1
        assert "Alice" in table_chunks[0].text

    def test_empty_text_no_tables(self) -> None:
        result = chunk_document("", self.chunker)
        assert result == []

    def test_empty_text_with_table(self) -> None:
        tables = [_FakeTable(["H"], [["val"]])]
        result = chunk_document("   ", self.chunker, tables=tables)
        assert len(result) >= 1
        assert result[0].content_type == "table"

    def test_indices_are_sequential(self) -> None:
        text = " ".join(["Word"] * 200)
        tables = [_FakeTable(["A", "B"], [["x", "y"]])]
        result = chunk_document(text, self.chunker, tables=tables)

        indices = [c.chunk_index for c in result]
        assert indices == list(range(len(result)))

    def test_redaction_preserved(self) -> None:
        text = (
            "The officer noted that [REDACTED] was present at the facility. "
            "No further action was taken regarding [REDACTED] involvement."
        )
        result = chunk_document(text, self.chunker)
        combined = " ".join(c.text for c in result)
        assert combined.count("[REDACTED]") == 2

    def test_empty_table_skipped(self) -> None:
        tables = [_FakeTable([], [])]
        result = chunk_document("Some text.", self.chunker, tables=tables)
        assert all(c.content_type == "narrative" for c in result)

    def test_none_tables_handled(self) -> None:
        result = chunk_document("Some text.", self.chunker, tables=None)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# New semchunk v3+ parameter tests
# ---------------------------------------------------------------------------


class TestOverlap:
    """Tests for overlap parameter."""

    def test_overlap_produces_overlapping_chunks(self) -> None:
        chunker = _make_test_chunker(chunk_size=10)
        text = " ".join(f"word{i}" for i in range(30))
        chunks_no_overlap = chunk_text(text, chunker)
        chunks_with_overlap = chunk_text(text, chunker, overlap=3)

        # With overlap, chunks share boundary text, so total text coverage
        # exceeds the source length.
        total_no = sum(len(c.text) for c in chunks_no_overlap)
        total_with = sum(len(c.text) for c in chunks_with_overlap)
        assert total_with > total_no

    def test_overlap_chunks_have_valid_offsets(self) -> None:
        chunker = _make_test_chunker(chunk_size=10)
        text = " ".join(f"word{i}" for i in range(30))
        chunks = chunk_text(text, chunker, overlap=2)
        for chunk in chunks:
            assert chunk.start_char >= 0
            assert chunk.end_char >= chunk.start_char
            assert text[chunk.start_char:chunk.end_char] == chunk.text

    def test_overlap_float_proportion(self) -> None:
        chunker = _make_test_chunker(chunk_size=10)
        text = " ".join(f"word{i}" for i in range(30))
        # 0.2 = 20% of chunk_size = 2 tokens overlap
        chunks = chunk_text(text, chunker, overlap=0.2)
        assert len(chunks) >= 2

    def test_no_overlap_default(self) -> None:
        chunker = _make_test_chunker(chunk_size=10)
        text = " ".join(f"word{i}" for i in range(20))
        chunks = chunk_text(text, chunker)
        # Without overlap, chunks should not share text.
        for i in range(len(chunks) - 1):
            assert chunks[i].end_char <= chunks[i + 1].start_char


class TestMemoizeAndMaxTokenChars:
    """Tests for creation-time parameters."""

    def test_memoize_false(self) -> None:
        chunker = create_chunker(_word_token_counter, chunk_size=50, memoize=False)
        result = chunker("hello world", offsets=True)
        assert len(result[0]) >= 1

    def test_max_token_chars(self) -> None:
        chunker = create_chunker(_word_token_counter, chunk_size=50, max_token_chars=10)
        result = chunker("hello world", offsets=True)
        assert len(result[0]) >= 1


class TestRedactionRepairWithOverlap:
    """Redaction repair remains safe when overlap is active."""

    def test_complete_marker_in_overlap_no_merge(self) -> None:
        """If [REDACTED] is complete in both chunks (overlap), no merge needed."""
        chunks = [
            TextChunk(text="text [REDACTED] more", start_char=0, end_char=20, chunk_index=0),
            TextChunk(text="[REDACTED] more next", start_char=10, end_char=30, chunk_index=1),
        ]
        result = _repair_redaction_splits(chunks)
        assert len(result) == 2  # No merge — marker is complete in both

    def test_split_marker_still_repaired_with_overlap(self) -> None:
        """Split marker at overlap boundary is still repaired."""
        chunks = [
            TextChunk(text="text [REDAC", start_char=0, end_char=11, chunk_index=0),
            TextChunk(text="TED] more", start_char=8, end_char=17, chunk_index=1),
        ]
        result = _repair_redaction_splits(chunks)
        assert len(result) == 1
        assert "[REDACTED]" in result[0].text


class TestChunkDocumentWithOverlap:
    """chunk_document passes overlap through correctly."""

    def test_overlap_passed_to_narrative(self) -> None:
        chunker = _make_test_chunker(chunk_size=10)
        text = " ".join(f"word{i}" for i in range(30))
        chunks = chunk_document(text, chunker, overlap=3)
        # Should produce more chunks than without overlap due to shared text
        chunks_no_overlap = chunk_document(text, chunker)
        assert len(chunks) >= len(chunks_no_overlap)

    def test_tables_not_overlapped(self) -> None:
        """Table chunks don't get overlap (they're self-contained)."""
        chunker = _make_test_chunker(chunk_size=10)

        class FakeTable:
            headers = ["A", "B"]
            rows = [["1", "2"], ["3", "4"]]

        chunks = chunk_document("short text", chunker, tables=[FakeTable()], overlap=5)
        table_chunks = [c for c in chunks if c.content_type == "table"]
        # Table chunks exist and have valid offsets
        assert len(table_chunks) >= 1
        for tc in table_chunks:
            assert tc.start_char >= 0


# ---------------------------------------------------------------------------
# chunk_texts_batch
# ---------------------------------------------------------------------------


class TestChunkTextsBatch:
    def test_empty_list(self) -> None:
        assert chunk_texts_batch([], _make_test_chunker()) == []

    def test_single_text(self) -> None:
        chunker = _make_test_chunker(chunk_size=50)
        result = chunk_texts_batch(["Hello world."], chunker)
        assert len(result) == 1
        assert len(result[0]) >= 1
        assert result[0][0].text == "Hello world."

    def test_multiple_texts(self) -> None:
        chunker = _make_test_chunker(chunk_size=50)
        texts = ["First document.", "Second document here."]
        result = chunk_texts_batch(texts, chunker)
        assert len(result) == 2
        assert result[0][0].text == "First document."
        assert result[1][0].text == "Second document here."

    def test_content_types_assigned(self) -> None:
        chunker = _make_test_chunker(chunk_size=50)
        texts = ["Narrative text.", "| A | B |"]
        result = chunk_texts_batch(texts, chunker, content_types=["narrative", "table"])
        assert result[0][0].content_type == "narrative"
        assert result[1][0].content_type == "table"

    def test_default_content_type_is_narrative(self) -> None:
        chunker = _make_test_chunker(chunk_size=50)
        result = chunk_texts_batch(["Some text."], chunker)
        assert result[0][0].content_type == "narrative"

    def test_offsets_valid(self) -> None:
        chunker = _make_test_chunker(chunk_size=10)
        text = " ".join(f"word{i}" for i in range(30))
        result = chunk_texts_batch([text], chunker)
        for chunk in result[0]:
            assert text[chunk.start_char:chunk.end_char] == chunk.text


# ---------------------------------------------------------------------------
# Batch mode via chunk_document
# ---------------------------------------------------------------------------


class TestChunkDocumentBatch:
    def test_batch_produces_same_content(self) -> None:
        """Batch and sequential modes produce chunks with the same text content."""
        chunker = _make_test_chunker(chunk_size=50)
        text = " ".join(["Government document narrative text."] * 30)
        tables = [_FakeTable(["Name", "Age"], [["Alice", "30"]])]

        sequential = chunk_document(text, chunker, tables=tables, batch=False)
        batched = chunk_document(text, chunker, tables=tables, batch=True)

        seq_texts = [c.text for c in sequential]
        bat_texts = [c.text for c in batched]
        assert seq_texts == bat_texts

    def test_batch_content_types_preserved(self) -> None:
        chunker = _make_test_chunker(chunk_size=50)
        text = "Some narrative."
        tables = [_FakeTable(["H"], [["v"]])]
        result = chunk_document(text, chunker, tables=tables, batch=True)
        types = {c.content_type for c in result}
        assert "narrative" in types
        assert "table" in types

    def test_batch_empty_text_with_table(self) -> None:
        chunker = _make_test_chunker(chunk_size=50)
        tables = [_FakeTable(["H"], [["val"]])]
        result = chunk_document("   ", chunker, tables=tables, batch=True)
        assert len(result) >= 1
        assert result[0].content_type == "table"

    def test_batch_empty_everything(self) -> None:
        chunker = _make_test_chunker(chunk_size=50)
        result = chunk_document("", chunker, batch=True)
        assert result == []

    def test_batch_indices_sequential(self) -> None:
        chunker = _make_test_chunker(chunk_size=10)
        text = " ".join(["Word"] * 100)
        tables = [_FakeTable(["A", "B"], [["x", "y"]])]
        result = chunk_document(text, chunker, tables=tables, batch=True)
        indices = [c.chunk_index for c in result]
        assert indices == list(range(len(result)))

    def test_batch_progress_does_not_error(self) -> None:
        chunker = _make_test_chunker(chunk_size=50)
        result = chunk_document("Some text.", chunker, batch=True, progress=True)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# create_chunker with new parameters
# ---------------------------------------------------------------------------


class TestCreateChunkerExtended:
    def test_cache_maxsize(self) -> None:
        chunker = create_chunker(_word_token_counter, chunk_size=50, cache_maxsize=100)
        result = chunker("hello world", offsets=True)
        assert len(result[0]) >= 1
