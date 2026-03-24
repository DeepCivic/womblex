"""Tests for CER and WER metric utilities."""

from __future__ import annotations

import pytest

from womblex.utils.metrics import cer, wer


class TestCER:
    def test_identical_strings(self) -> None:
        assert cer("hello world", "hello world") == 0.0

    def test_complete_mismatch(self) -> None:
        # hypothesis is entirely wrong — distance equals len(ref)
        result = cer("abc", "xyz")
        assert result == pytest.approx(1.0)

    def test_one_substitution(self) -> None:
        # "hxllo" vs "hello" — 1 substitution in 5 chars
        assert cer("hello", "hxllo") == pytest.approx(1 / 5)

    def test_insertion(self) -> None:
        # hypothesis has an extra char
        assert cer("cat", "cats") == pytest.approx(1 / 3)

    def test_deletion(self) -> None:
        assert cer("cats", "cat") == pytest.approx(1 / 4)

    def test_empty_reference_empty_hypothesis(self) -> None:
        assert cer("", "") == 0.0

    def test_empty_reference_nonempty_hypothesis(self) -> None:
        assert cer("", "something") == 1.0

    def test_normalise_case(self) -> None:
        assert cer("Hello", "hello") == 0.0

    def test_normalise_whitespace(self) -> None:
        assert cer("foo  bar", "foo bar") == 0.0

    def test_normalise_disabled(self) -> None:
        # With normalise=False, case difference counts as edits
        assert cer("Hello", "hello", normalise=False) > 0.0

    def test_unicode_normalisation(self) -> None:
        # NFC-normalised equivalents should match
        nfc = "\u00e9"   # é as single codepoint
        nfd = "e\u0301"  # é as e + combining accent
        assert cer(nfc, nfd) == 0.0


class TestWER:
    def test_identical(self) -> None:
        assert wer("the cat sat", "the cat sat") == 0.0

    def test_one_substitution(self) -> None:
        # 1 word wrong out of 3
        assert wer("the cat sat", "the dog sat") == pytest.approx(1 / 3)

    def test_one_deletion(self) -> None:
        # hypothesis missing one word
        assert wer("the cat sat", "the sat") == pytest.approx(1 / 3)

    def test_one_insertion(self) -> None:
        assert wer("the cat sat", "the big cat sat") == pytest.approx(1 / 3)

    def test_empty_reference_empty_hypothesis(self) -> None:
        assert wer("", "") == 0.0

    def test_empty_reference_nonempty_hypothesis(self) -> None:
        assert wer("", "extra words") == 1.0

    def test_normalise_case(self) -> None:
        assert wer("The Cat Sat", "the cat sat") == 0.0

    def test_normalise_whitespace(self) -> None:
        assert wer("the  cat\tsat", "the cat sat") == 0.0

    def test_perfect_ocr_on_iam_style_sentence(self) -> None:
        ref = "the quick brown fox jumps over the lazy dog"
        assert wer(ref, ref) == 0.0

    def test_cer_wer_both_detect_error(self) -> None:
        # Both metrics should be non-zero when there is an error
        ref = "a b c"
        hyp = "a x c"
        assert cer(ref, hyp) > 0.0
        assert wer(ref, hyp) > 0.0
        # CER and WER use different denominators so their values differ
        assert cer(ref, hyp) != pytest.approx(wer(ref, hyp))
