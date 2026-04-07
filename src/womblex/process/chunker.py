"""Text chunking using semchunk.

Wraps semchunk v3+ to split extracted text into semantically coherent
chunks. Exposes all semchunk parameters (overlap, memoize, max_token_chars,
processes) through config. Uses semchunk's native offset tracking.

Handles tables (converted to markdown) and preserves ``[REDACTED]``
markers across chunk boundaries.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import semchunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TextChunk:
    """A single chunk of text with offset and content-type metadata."""

    text: str
    start_char: int
    end_char: int
    chunk_index: int
    content_type: str = "narrative"  # "narrative" | "table"
    has_redaction: bool = False


# ---------------------------------------------------------------------------
# Chunker factory
# ---------------------------------------------------------------------------


def create_chunker(
    tokenizer: str | Callable[[str], int],
    chunk_size: int,
    *,
    memoize: bool = True,
    max_token_chars: int | None = None,
) -> semchunk.Chunker:
    """Create a semchunk chunker.

    Args:
        tokenizer: HuggingFace tokeniser identifier string, or a callable
            token counter ``(str) -> int``.
        chunk_size: Maximum tokens per chunk.
        memoize: Cache token counts for repeated substrings.
        max_token_chars: Max chars per token estimate for optimisation.

    Returns:
        A semchunk Chunker instance.
    """
    return semchunk.chunkerify(
        tokenizer,
        chunk_size=chunk_size,
        memoize=memoize,
        max_token_chars=max_token_chars,
    )


# ---------------------------------------------------------------------------
# Table to markdown
# ---------------------------------------------------------------------------


def table_to_markdown(headers: list[str], rows: list[list[str]]) -> str:
    """Convert table data to a markdown table string."""
    if not headers and not rows:
        return ""

    cols = headers if headers else ([""] * len(rows[0]) if rows else [])
    lines: list[str] = []

    # Header row
    lines.append("| " + " | ".join(cols) + " |")
    # Separator
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    # Data rows
    for row in rows:
        padded = list(row) + [""] * (len(cols) - len(row))
        lines.append("| " + " | ".join(padded[: len(cols)]) + " |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core chunking — uses semchunk native offsets
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    chunker: semchunk.Chunker,
    content_type: str = "narrative",
    *,
    overlap: int | float | None = None,
    processes: int = 1,
) -> list[TextChunk]:
    """Split text into chunks using semchunk's native offset tracking.

    Args:
        text: The full text to chunk.
        chunker: A semchunk Chunker instance.
        content_type: Label applied to every chunk produced.
        overlap: Token overlap between adjacent chunks.
        processes: Number of parallel workers (1 = single-threaded).

    Returns:
        List of TextChunk objects with character offsets.
    """
    if not text.strip():
        return []

    result_tuple = chunker(
        text,
        offsets=True,
        overlap=overlap,
        processes=processes,
    )
    # semchunk returns (list[str], list[tuple[int, int]]) when offsets=True
    raw_chunks: list[str] = result_tuple[0]
    raw_offsets: list[tuple[int, int]] = result_tuple[1]

    return [
        TextChunk(
            text=chunk_str,
            start_char=start,
            end_char=end,
            chunk_index=i,
            content_type=content_type,
        )
        for i, (chunk_str, (start, end)) in enumerate(zip(raw_chunks, raw_offsets))
    ]


def _repair_redaction_splits(chunks: list[TextChunk]) -> list[TextChunk]:
    """Merge chunks where a ``[REDACTED]`` marker was split across a boundary.

    Safe with overlap: if the marker is already complete in both chunks
    (due to overlap), the repair won't trigger. Only fires when one chunk
    ends with a prefix of ``[REDACTED]`` and the next starts with the suffix.
    """
    if not chunks:
        return chunks

    marker = "[REDACTED]"
    repaired: list[TextChunk] = []

    i = 0
    while i < len(chunks):
        chunk = chunks[i]

        needs_merge = False
        if i + 1 < len(chunks):
            for length in range(1, len(marker)):
                suffix = marker[:length]
                if chunk.text.endswith(suffix) and chunks[i + 1].text.startswith(
                    marker[length:]
                ):
                    needs_merge = True
                    break

        if needs_merge:
            merged_text = chunk.text + chunks[i + 1].text
            repaired.append(
                TextChunk(
                    text=merged_text,
                    start_char=chunk.start_char,
                    end_char=chunks[i + 1].end_char,
                    chunk_index=len(repaired),
                    content_type=chunk.content_type,
                )
            )
            i += 2
        else:
            repaired.append(
                TextChunk(
                    text=chunk.text,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    chunk_index=len(repaired),
                    content_type=chunk.content_type,
                )
            )
            i += 1

    return repaired


# ---------------------------------------------------------------------------
# Document-level chunking
# ---------------------------------------------------------------------------


def chunk_document(
    full_text: str,
    chunker: semchunk.Chunker,
    tables: list[object] | None = None,
    *,
    overlap: int | float | None = None,
    processes: int = 1,
) -> list[TextChunk]:
    """Chunk extracted document content, handling tables and redactions.

    1. Chunks narrative text using semchunk native offsets.
    2. Converts ``TableData`` objects to markdown and chunks separately.
    3. Appends table chunks after narrative chunks (stable ordering).
    4. Repairs any ``[REDACTED]`` markers split across chunk boundaries.

    Args:
        full_text: Concatenated page text from ``ExtractionResult.full_text``.
        chunker: A semchunk Chunker instance.
        tables: Optional list of ``TableData`` objects (from extraction).
        overlap: Token overlap between adjacent chunks.
        processes: Number of parallel workers.

    Returns:
        Ordered list of TextChunk objects.
    """
    all_chunks: list[TextChunk] = []

    # Narrative chunks
    narrative_chunks = chunk_text(
        full_text, chunker, content_type="narrative",
        overlap=overlap, processes=processes,
    )
    all_chunks.extend(narrative_chunks)

    # Table chunks (no overlap — tables are self-contained)
    if tables:
        for tbl in tables:
            md = table_to_markdown(tbl.headers, tbl.rows)  # type: ignore[attr-defined]
            if not md.strip():
                continue
            tbl_chunks = chunk_text(md, chunker, content_type="table")
            all_chunks.extend(tbl_chunks)

    # Re-index after merging
    for idx, chunk in enumerate(all_chunks):
        chunk.chunk_index = idx

    # Repair split [REDACTED] markers
    all_chunks = _repair_redaction_splits(all_chunks)

    return all_chunks
