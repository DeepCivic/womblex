"""Tests for the token-budget request packer.

Offline: a whitespace token counter (``count = word count``) stands in for the
kanon-2 tokenizer so packing/splitting logic is tested deterministically with
no network. One live test exercises the real :class:`TokenCounter` when
transformers can fetch the tokenizer.
"""

from __future__ import annotations

import pytest

from womblex.utils.token_packer import (
    pack_by_tokens,
    split_on_boundaries,
)


def _wc(texts: list[str]) -> list[int]:
    """Fake token counter: token count == whitespace word count."""
    return [len(t.split()) for t in texts]


class TestPackByTokens:
    def test_packs_to_max_items(self):
        items = [(f"k{i}", "a") for i in range(5)]  # 1 token each
        groups = list(pack_by_tokens(items, _wc, max_items=2, token_budget=1000))
        assert [len(g.items) for g in groups] == [2, 2, 1]
        assert all(not g.solo for g in groups)

    def test_packs_to_token_budget(self):
        # each text is 3 tokens; budget 8 → 2 per group (6 <= 8, 9 > 8)
        items = [(f"k{i}", "a b c") for i in range(5)]
        groups = list(pack_by_tokens(items, _wc, max_items=100, token_budget=8))
        assert [g.total_tokens for g in groups] == [6, 6, 3]

    def test_over_budget_item_goes_solo_and_flushes_buffer(self):
        items = [
            ("small1", "a"),
            ("big", "a b c d e"),   # 5 tokens >= budget 5
            ("small2", "a"),
        ]
        groups = list(pack_by_tokens(items, _wc, max_items=10, token_budget=5))
        # buffered small1 flushes, big goes solo, small2 buffers to the end
        assert [g.solo for g in groups] == [False, True, False]
        assert [ [i.key for i in g.items] for g in groups] == [["small1"], ["big"], ["small2"]]

    def test_order_preserved(self):
        items = [(f"k{i}", "a b") for i in range(6)]
        groups = list(pack_by_tokens(items, _wc, max_items=2, token_budget=1000))
        keys = [i.key for g in groups for i in g.items]
        assert keys == [f"k{i}" for i in range(6)]

    def test_validates_params(self):
        with pytest.raises(ValueError):
            list(pack_by_tokens([], _wc, max_items=0, token_budget=10))
        with pytest.raises(ValueError):
            list(pack_by_tokens([], _wc, max_items=1, token_budget=0))

    def test_empty_input(self):
        assert list(pack_by_tokens([], _wc, max_items=8, token_budget=100)) == []


class TestSplitOnBoundaries:
    def test_under_budget_is_single_segment(self):
        text = "a b\n\nc d"
        segs = split_on_boundaries(text, _wc, max_tokens=100)
        assert len(segs) == 1
        assert segs[0].text == text
        assert segs[0].start_char == 0 and segs[0].end_char == len(text)

    def test_splits_on_blank_lines_within_budget(self):
        # three 2-token blocks, budget 3 → each block its own segment
        text = "a b\n\nc d\n\ne f"
        segs = split_on_boundaries(text, _wc, max_tokens=3)
        assert len(segs) == 3
        assert [s.text for s in segs] == ["a b", "c d", "e f"]

    def test_offsets_index_original_text(self):
        text = "a b\n\nc d\n\ne f"
        segs = split_on_boundaries(text, _wc, max_tokens=5)  # 2+2=4 fits, +2=6 no
        # first segment groups blocks 1+2 (includes the separator verbatim)
        for s in segs:
            assert text[s.start_char:s.end_char] == s.text
        assert segs[0].text == "a b\n\nc d"
        assert segs[1].text == "e f"

    def test_single_huge_block_is_its_own_segment(self):
        text = "a b c d e f g h"  # 8 tokens, one block, budget 3
        segs = split_on_boundaries(text, _wc, max_tokens=3)
        assert len(segs) == 1
        assert segs[0].text == text


class TestTokenCounterLive:
    def test_real_kanon_tokenizer_counts(self):
        pytest.importorskip("transformers")
        from womblex.utils.token_packer import TokenCounter

        try:
            counter = TokenCounter()
            n = counter.count("The quick brown fox.")
        except Exception as e:  # no network / model unavailable
            pytest.skip(f"kanon-2 tokenizer unavailable: {e}")
        assert n > 0
        assert counter.count_batch(["a", "a b c"]) == [
            counter.count("a"), counter.count("a b c"),
        ]
