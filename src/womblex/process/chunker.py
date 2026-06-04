"""Text chunking using semchunk.

Adapter boundary (audited I5, 2026-05-30). semchunk 3.x owns all
chunking; Womblex is a thin adapter handling only what semchunk can't
know about — parquet I/O, element-stream → ``ChunkInput`` projection,
source-hash plumbing, and ``<REDACTED>`` cross-boundary repair.
semchunk's own parameters *are* the feature surface; this module adds
no toggle that re-exposes a semchunk feature under a Womblex name (the
dead ``ChunkingConfig.batch`` flag was removed in I5 — chunk_batch
always batches across the whole input list).

Pass-through to semchunk:

- :func:`create_chunker` wraps ``semchunk.chunkerify`` and exposes
  every creation-time parameter (``tokenizer`` → ``tokenizer_or_token_counter``,
  ``chunk_size``, ``memoize``, ``cache_maxsize``, ``max_token_chars``).
- :func:`chunk_batch` is the single entry point used by every caller
  (E2E :func:`womblex.operations.run_chunking`, per-stage
  :mod:`womblex.process.chunk_stage`). It flattens narratives across
  every input document into one semchunk call, and table markdowns into
  another, so ``processes``, ``overlap`` and ``progress`` pass straight
  through and parallelise across the whole batch instead of being thrown
  away per-document. ``offsets=True`` is pinned (Womblex needs char
  offsets for page mapping); every other ``__call__`` argument is the
  caller's via ``ChunkingConfig``.

Womblex-only surface (no semchunk equivalent):

- :class:`TextChunk` — Womblex's chunk schema (offsets, content type,
  page span, redaction flag).
- :func:`table_to_markdown` — TableData → markdown projection.
- :func:`reassemble_narrative` / :func:`collect_tables_from_elements` /
  :func:`build_chunk_input` — element-stream → ``ChunkInput`` projection.
- :func:`_repair_redaction_splits` — heal ``<REDACTED>`` markers split
  across a chunk boundary; semchunk has no opinion about our marker.
"""

from __future__ import annotations

import bisect
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast

import semchunk

from womblex.ingest.elements import Element, TEXT_KINDS
from womblex.ingest.views import _element_to_table_data, _sheets_to_table_data

logger = logging.getLogger(__name__)

NARRATIVE_JOIN = "\n\n"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TextChunk:
    """A single chunk of text with offset, content-type and locality metadata."""

    text: str
    start_char: int
    end_char: int
    chunk_index: int
    content_type: str = "narrative"  # "narrative" | "table"
    has_redaction: bool = False
    page_start: int | None = None
    page_end: int | None = None


@dataclass
class ChunkInput:
    """One document's input to :func:`chunk_batch`.

    ``narrative`` is the concatenated text of the doc's text-bearing
    elements in ``elem_order``. ``page_breaks`` is a sorted list of
    ``(end_char_exclusive, page_number)`` pairs covering ``narrative``;
    empty for sources without page semantics (DOCX, spreadsheets).
    ``tables`` is one ``(page, markdown)`` per table — ``page`` may be
    ``None``.
    """

    source_hash: str
    narrative: str
    page_breaks: list[tuple[int, int]] = field(default_factory=list)
    tables: list[tuple[int | None, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Chunker factory
# ---------------------------------------------------------------------------


def create_chunker(
    tokenizer: str | Callable[[str], int],
    chunk_size: int | None = None,
    *,
    memoize: bool = True,
    cache_maxsize: int | None = None,
    max_token_chars: int | None = None,
) -> semchunk.Chunker:
    """Construct a semchunk chunker.

    Args:
        tokenizer: HuggingFace tokeniser identifier string, or a callable
            token counter ``(str) -> int``.
        chunk_size: Maximum tokens per chunk. ``None`` passes through to
            semchunk, which derives the size from the tokeniser's
            ``model_max_length`` (only valid for a real tokeniser, not a
            callable token counter).
        memoize: Cache token counts for repeated substrings.
        cache_maxsize: Upper bound on memoization cache entries.
            ``None`` = unbounded.
        max_token_chars: Max chars per token estimate for optimisation.
    """
    return semchunk.chunkerify(
        tokenizer,
        chunk_size=chunk_size,
        memoize=memoize,
        cache_maxsize=cache_maxsize,
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

    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for row in rows:
        padded = list(row) + [""] * (len(cols) - len(row))
        lines.append("| " + " | ".join(padded[: len(cols)]) + " |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Redaction-split repair
# ---------------------------------------------------------------------------


def _repair_redaction_splits(chunks: list[TextChunk]) -> list[TextChunk]:
    """Merge chunks where a ``<REDACTED>`` marker was split across a boundary.

    Safe with overlap: if the marker is already complete in both chunks
    (because overlap duplicated it), the repair won't trigger. Only
    fires when one chunk ends with a prefix of ``<REDACTED>`` and the
    next starts with the suffix.
    """
    if not chunks:
        return chunks

    # Cross-file coupling: this literal MUST match the blackout marker emitted
    # by redact/stage.py and stay coherent with the PII tag style in
    # pii/cleaner.py. See docs/decisions.md "PII / redaction marker
    # convention" before changing.
    marker = "<REDACTED>"
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
            nxt = chunks[i + 1]
            merged_text = chunk.text + nxt.text
            repaired.append(
                TextChunk(
                    text=merged_text,
                    start_char=chunk.start_char,
                    end_char=nxt.end_char,
                    chunk_index=len(repaired),
                    content_type=chunk.content_type,
                    has_redaction=True,
                    page_start=chunk.page_start,
                    page_end=nxt.page_end if nxt.page_end is not None else chunk.page_end,
                )
            )
            i += 2
        else:
            chunk.chunk_index = len(repaired)
            repaired.append(chunk)
            i += 1

    return repaired


# ---------------------------------------------------------------------------
# Page lookup
# ---------------------------------------------------------------------------


def _page_for_offset(
    page_breaks: list[tuple[int, int]], offset: int,
) -> int | None:
    """Return the page number covering ``offset``, or ``None`` if past end.

    ``page_breaks`` is sorted by ``end_char_exclusive``; an offset falls
    on the first page whose end exceeds it.
    """
    if not page_breaks:
        return None
    ends = [end for end, _ in page_breaks]
    idx = bisect.bisect_right(ends, offset)
    if idx >= len(page_breaks):
        return None
    return page_breaks[idx][1]


# ---------------------------------------------------------------------------
# Batch chunking — single entry point
# ---------------------------------------------------------------------------


_REDACTED_MARKER = "<REDACTED>"


def chunk_batch(
    inputs: list[ChunkInput],
    chunker: semchunk.Chunker,
    *,
    overlap: int | float | None = None,
    processes: int = 1,
    progress: bool = False,
) -> dict[str, list[TextChunk]]:
    """Chunk every doc's narrative + tables in two semchunk calls.

    All narrative texts across ``inputs`` are flattened into one
    ``chunker(...)`` invocation (with ``overlap``); all table markdowns
    into another (no overlap — tables are self-contained). semchunk's
    ``processes`` and ``progress`` arguments therefore parallelise
    across the entire batch, not per document.

    Returns ``{source_hash: list[TextChunk]}`` with ``chunk_index``
    re-sequenced per doc, ``has_redaction`` populated from the chunk
    text, and ``page_start`` / ``page_end`` resolved from per-doc page
    breaks (or the table's page for table chunks).
    """
    if not inputs:
        return {}

    narrative_texts: list[str] = []
    narrative_owners: list[int] = []
    table_texts: list[str] = []
    table_owners: list[tuple[int, int | None]] = []

    for i, doc in enumerate(inputs):
        if doc.narrative.strip():
            narrative_texts.append(doc.narrative)
            narrative_owners.append(i)
        for page, md in doc.tables:
            if md.strip():
                table_texts.append(md)
                table_owners.append((i, page))

    out: dict[str, list[TextChunk]] = {doc.source_hash: [] for doc in inputs}

    if narrative_texts:
        n_chunks_all, n_offsets_all = _chunker_batch(
            chunker, narrative_texts,
            overlap=overlap, processes=processes, progress=progress,
        )
        for owner_idx, chunks, offsets in zip(
            narrative_owners, n_chunks_all, n_offsets_all,
        ):
            doc = inputs[owner_idx]
            for text, (start, end) in zip(chunks, offsets):
                out[doc.source_hash].append(
                    _build_narrative_chunk(
                        text=text, start=start, end=end,
                        page_breaks=doc.page_breaks,
                    )
                )

    if table_texts:
        t_chunks_all, t_offsets_all = _chunker_batch(
            chunker, table_texts,
            overlap=None, processes=processes, progress=False,
        )
        for (owner_idx, table_page), chunks, offsets in zip(
            table_owners, t_chunks_all, t_offsets_all,
        ):
            doc = inputs[owner_idx]
            for text, (start, end) in zip(chunks, offsets):
                out[doc.source_hash].append(
                    TextChunk(
                        text=text,
                        start_char=start,
                        end_char=end,
                        chunk_index=0,
                        content_type="table",
                        has_redaction=_REDACTED_MARKER in text,
                        page_start=table_page,
                        page_end=table_page,
                    )
                )

    for src, doc_chunks in out.items():
        out[src] = _repair_redaction_splits(doc_chunks)

    return out


def _chunker_batch(
    chunker: semchunk.Chunker,
    texts: list[str],
    *,
    overlap: int | float | None,
    processes: int,
    progress: bool,
) -> tuple[list[list[str]], list[list[tuple[int, int]]]]:
    """Call semchunk on a list of texts; always returns ``(chunks, offsets)``."""
    chunks, offsets = chunker(
        texts,
        offsets=True,
        overlap=overlap,
        processes=processes,
        progress=progress,
    )
    return cast(
        "tuple[list[list[str]], list[list[tuple[int, int]]]]", (chunks, offsets),
    )


def _build_narrative_chunk(
    *, text: str, start: int, end: int,
    page_breaks: list[tuple[int, int]],
) -> TextChunk:
    page_start = _page_for_offset(page_breaks, start)
    page_end = _page_for_offset(page_breaks, max(start, end - 1))
    return TextChunk(
        text=text,
        start_char=start,
        end_char=end,
        chunk_index=0,
        content_type="narrative",
        has_redaction=_REDACTED_MARKER in text,
        page_start=page_start,
        page_end=page_end,
    )


# ---------------------------------------------------------------------------
# Element-stream reassembly
# ---------------------------------------------------------------------------


def reassemble_narrative(
    elements: list[Element],
) -> tuple[str, list[tuple[int, int]]]:
    """Concatenate TEXT_KINDS element texts and emit page-break offsets.

    Returns ``(text, [(end_offset_exclusive, page_number), ...])``. An
    offset in ``[prev_end, curr_end)`` resolves to ``curr_page``; offsets
    past the final entry have no page. Elements with no ``page``
    contribute to the text but not to any page span.

    Elements are assumed already sorted by ``order``.
    """
    parts: list[str] = []
    spans: list[list] = []
    cursor = 0

    for e in elements:
        if e.kind not in TEXT_KINDS or not e.text:
            continue
        piece = e.text if not parts else NARRATIVE_JOIN + e.text
        parts.append(piece)
        next_cursor = cursor + len(piece)
        if e.page is not None:
            if spans and spans[-1][2] == e.page:
                spans[-1][1] = next_cursor
            else:
                spans.append([cursor, next_cursor, e.page])
        cursor = next_cursor

    page_breaks = [(end, page) for _, end, page in spans]
    return "".join(parts), page_breaks


def collect_tables_from_elements(
    elements: list[Element],
) -> list[tuple[int | None, str]]:
    """Materialise ``(page, markdown)`` per table for :func:`chunk_batch`.

    Mirrors :pyattr:`ExtractionResult.tables`: one entry per
    ``kind='table'`` element followed by one synthetic entry per
    spreadsheet sheet (page is ``None`` for sheets).
    """
    out: list[tuple[int | None, str]] = []
    for e in elements:
        if e.kind != "table":
            continue
        td = _element_to_table_data(e)
        md = table_to_markdown(td.headers, td.rows)
        if md.strip():
            out.append((e.page, md))

    for sheet_td in _sheets_to_table_data(elements):
        md = table_to_markdown(sheet_td.headers, sheet_td.rows)
        if md.strip():
            out.append((None, md))

    return out


def build_chunk_input(
    source_hash: str,
    elements: list[Element],
    *,
    include_tables: bool = True,
) -> ChunkInput:
    """Convenience: build a :class:`ChunkInput` from an ordered element list."""
    narrative, page_breaks = reassemble_narrative(elements)
    tables = collect_tables_from_elements(elements) if include_tables else []
    return ChunkInput(
        source_hash=source_hash,
        narrative=narrative,
        page_breaks=page_breaks,
        tables=tables,
    )
