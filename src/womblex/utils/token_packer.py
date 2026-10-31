"""Token-budget request packer for Isaacus API calls.

Isaacus rate limits bind on **tokens per request/window**, not request count
— feeding whole ~400-page documents (~150-200 K tokens) triggers 429s. So
requests to the enrichment / classification endpoints are packed by *exact*
local token counts, never naive doc-count batching. The counts come from the
``isaacus/kanon-2-tokenizer`` (free on Hugging Face, exact client-side), so
pre-flight budgeting is exact, not estimated.

Two concerns, one module:

- :class:`TokenCounter` — a thin cached wrapper over the kanon-2 tokenizer
  (fast tokenizer, no truncation, no network at call time once downloaded).
- :func:`pack_by_tokens` — greedily groups ``(key, text)`` items into
  requests of ``min(max_items, token_budget)``; an item over budget goes
  **solo**. :func:`split_on_boundaries` splits a single over-ceiling document
  on structural (blank-line) boundaries into offset-tagged segments so the
  caller can enrich each separately and merge results back on offsets.

Shared utility: consumed by :mod:`womblex.analyse.enrich_stage` (8-doc,
token-aware batching + long-doc split) and the landtitle sim's ``annotate``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_TOKENIZER = "isaacus/kanon-2-tokenizer"


class TokenCounter:
    """Exact local token counting via the kanon-2 (or any HF) tokenizer.

    The tokenizer is loaded lazily on first count so importing this module
    (and constructing a counter) stays cheap and offline-safe; the model is
    fetched/cached by ``transformers`` on first use.
    """

    def __init__(self, tokenizer: str = DEFAULT_TOKENIZER, **from_pretrained_kwargs: object):
        self._name = tokenizer
        self._kwargs = from_pretrained_kwargs
        self._tok: object | None = None

    def _tokenizer(self) -> object:
        if self._tok is None:
            from transformers import AutoTokenizer

            from womblex.utils.models import resolve_local_model_path

            # Prefer a vendored copy under models/<name> so token counting is
            # fully offline (no Hugging Face round-trip per run); fall back to
            # the hub id only when it isn't vendored.
            local = resolve_local_model_path(self._name.split("/")[-1])
            source = str(local) if isinstance(local, Path) else self._name
            self._tok = AutoTokenizer.from_pretrained(source, **self._kwargs)
        return self._tok

    def count(self, text: str) -> int:
        if not text:
            return 0
        tok = self._tokenizer()
        # verbose=False suppresses the ">model_max_length" warning; the count
        # is the full, untruncated token length either way.
        return len(tok(text, add_special_tokens=False, verbose=False)["input_ids"])  # type: ignore[operator,index]

    def count_batch(self, texts: list[str]) -> list[int]:
        if not texts:
            return []
        tok = self._tokenizer()
        encoded = tok(texts, add_special_tokens=False, verbose=False)["input_ids"]  # type: ignore[operator,index]
        return [len(ids) for ids in encoded]


@dataclass
class Packable:
    """One item to pack: an id ``key``, its ``text`` and its exact ``tokens``."""

    key: str
    text: str
    tokens: int


@dataclass
class RequestGroup:
    """A packed request: the items to send together and their token total.

    ``solo`` is True when the group is a single item that met or exceeded the
    token budget on its own — the caller may want to send it alone and, if it
    is also over the split ceiling, split it via :func:`split_on_boundaries`.
    """

    items: list[Packable]
    total_tokens: int
    solo: bool


@dataclass
class TextSegment:
    """A sub-document of a split: its ``text`` and char offsets into the parent.

    ``start_char`` is the offset of this segment's first character in the
    original text, so entity/segment spans returned for the segment merge
    back into the parent's coordinate space by adding ``start_char``.
    """

    text: str
    start_char: int
    end_char: int
    tokens: int


def pack_by_tokens(
    items: Iterable[tuple[str, str]],
    count_fn: Callable[[list[str]], list[int]],
    *,
    max_items: int,
    token_budget: int,
) -> Iterator[RequestGroup]:
    """Greedily group ``(key, text)`` items into token-budgeted requests.

    Each yielded :class:`RequestGroup` has at most ``max_items`` items and a
    token total at most ``token_budget`` — *except* a single item whose own
    token count meets or exceeds ``token_budget``, which is yielded solo
    (``solo=True``) so the caller can send it alone / split it. ``count_fn``
    maps a list of texts to their exact token counts (e.g.
    ``TokenCounter().count_batch``); it is called once per input chunk, so
    counting is batched, not per-item.

    Order is preserved. Items are consumed streaming, but ``count_fn`` is
    applied in windows of ``max_items`` to keep counting batched.
    """
    if max_items < 1:
        raise ValueError(f"max_items must be >= 1, got {max_items}")
    if token_budget < 1:
        raise ValueError(f"token_budget must be >= 1, got {token_budget}")

    current: list[Packable] = []
    current_tokens = 0

    for window in _windows(items, max_items):
        texts = [t for _, t in window]
        counts = count_fn(texts)
        for (key, text), tokens in zip(window, counts):
            item = Packable(key=key, text=text, tokens=tokens)
            if tokens >= token_budget:
                # Over budget on its own: flush what's buffered, then send solo.
                if current:
                    yield RequestGroup(current, current_tokens, solo=False)
                    current, current_tokens = [], 0
                yield RequestGroup([item], tokens, solo=True)
                continue
            if current and (
                len(current) >= max_items or current_tokens + tokens > token_budget
            ):
                yield RequestGroup(current, current_tokens, solo=False)
                current, current_tokens = [], 0
            current.append(item)
            current_tokens += tokens

    if current:
        yield RequestGroup(current, current_tokens, solo=False)


def split_on_boundaries(
    text: str,
    count_fn: Callable[[list[str]], list[int]],
    *,
    max_tokens: int,
) -> list[TextSegment]:
    """Split ``text`` into ``<= max_tokens`` segments on blank-line boundaries.

    Structural split for documents past the observed rate-limit ceiling
    (~100 K tokens): blocks (blank-line-delimited paragraphs — the same
    boundaries :mod:`womblex.ingest.records` emits) are accumulated greedily
    until the next would breach ``max_tokens``. Char offsets are into the
    original ``text`` so the caller merges each segment's entity/segment
    spans back by adding ``start_char`` (the same offset stitch the enricher
    does internally for >16 K inputs, applied one level up).

    A single block that alone exceeds ``max_tokens`` becomes its own segment
    (the enricher's ``overflow_strategy`` still chunks it internally) — we do
    not sub-split within a paragraph, keeping offsets on real boundaries.
    Returns a single whole-text segment when the text is already under budget.
    """
    total = count_fn([text])[0] if text else 0
    if total <= max_tokens:
        return [TextSegment(text=text, start_char=0, end_char=len(text), tokens=total)]

    blocks = _split_blocks_with_offsets(text)
    block_tokens = count_fn([b for b, _, _ in blocks]) if blocks else []

    segments: list[TextSegment] = []
    cur_start: int | None = None
    cur_end = 0
    cur_tokens = 0

    def flush() -> None:
        nonlocal cur_start, cur_end, cur_tokens
        if cur_start is not None:
            segments.append(TextSegment(
                text=text[cur_start:cur_end], start_char=cur_start,
                end_char=cur_end, tokens=cur_tokens,
            ))
        cur_start, cur_end, cur_tokens = None, 0, 0

    for (_block, start, end), tokens in zip(blocks, block_tokens):
        if cur_start is not None and cur_tokens + tokens > max_tokens:
            flush()
        if cur_start is None:
            cur_start = start
        cur_end = end
        cur_tokens += tokens
    flush()
    return segments


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _windows(items: Iterable[tuple[str, str]], size: int) -> Iterator[list[tuple[str, str]]]:
    buf: list[tuple[str, str]] = []
    for it in items:
        buf.append(it)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def _split_blocks_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """Return ``(block, start_char, end_char)`` for each blank-line block.

    Offsets index ``text``; the separators between blocks are excluded from
    the segments but their span is implicitly the gap between consecutive
    ``end`` and ``start`` — a segment reconstructed as ``text[start:end]``
    over a *run* of blocks includes the in-run separators verbatim.
    """
    import re

    blocks: list[tuple[str, int, int]] = []
    pos = 0
    for m in re.finditer(r"\n[ \t]*\n+", text):
        block = text[pos:m.start()]
        if block.strip():
            blocks.append((block, pos, m.start()))
        pos = m.end()
    tail = text[pos:]
    if tail.strip():
        blocks.append((tail, pos, len(text)))
    return blocks


__all__ = [
    "DEFAULT_TOKENIZER",
    "Packable",
    "RequestGroup",
    "TextSegment",
    "TokenCounter",
    "pack_by_tokens",
    "split_on_boundaries",
]
