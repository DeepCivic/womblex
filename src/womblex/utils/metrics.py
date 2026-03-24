"""Text extraction accuracy metrics: CER and WER.

Both metrics use Levenshtein edit distance to compare a hypothesis (extracted
text) against a reference (ground truth). Lower values are better.
"""

from __future__ import annotations

import unicodedata


def _normalise(text: str) -> str:
    """Lowercase, NFC-normalise, and collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    # Collapse all whitespace runs to a single space and strip edges
    return " ".join(text.split())


def _levenshtein(seq_a: list | str, seq_b: list | str) -> int:
    """Compute Levenshtein edit distance between two sequences.

    Works on both strings (character-level) and lists of strings (word-level).
    Uses an O(min(m,n)) space DP implementation.
    """
    if len(seq_a) < len(seq_b):
        seq_a, seq_b = seq_b, seq_a

    prev = list(range(len(seq_b) + 1))
    for i, ca in enumerate(seq_a, 1):
        curr = [i]
        for j, cb in enumerate(seq_b, 1):
            curr.append(min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + (ca != cb),  # substitution
            ))
        prev = curr

    return prev[-1]


def cer(reference: str, hypothesis: str, *, normalise: bool = True) -> float:
    """Character Error Rate: edit distance at character level divided by len(reference).

    Args:
        reference: Ground truth text.
        hypothesis: Text produced by the extraction process.
        normalise: Apply lowercasing, NFC and whitespace normalisation before
            comparison. Defaults to True.

    Returns:
        CER as a float ≥ 0.0. Values > 1.0 are possible when the hypothesis is
        much longer than the reference. Returns 0.0 when both strings are empty.

    Raises:
        ValueError: If reference is non-empty after normalisation and hypothesis
            is also non-empty, but reference normalises to an empty string
            (guards against accidental empty-reference comparisons).
    """
    ref = _normalise(reference) if normalise else reference
    hyp = _normalise(hypothesis) if normalise else hypothesis

    if not ref:
        return 0.0 if not hyp else 1.0

    return _levenshtein(hyp, ref) / len(ref)


def wer(reference: str, hypothesis: str, *, normalise: bool = True) -> float:
    """Word Error Rate: edit distance at word level divided by word count of reference.

    Args:
        reference: Ground truth text.
        hypothesis: Text produced by the extraction process.
        normalise: Apply lowercasing, NFC and whitespace normalisation before
            tokenising. Defaults to True.

    Returns:
        WER as a float ≥ 0.0. Values > 1.0 are possible. Returns 0.0 when
        both inputs produce no words after normalisation.
    """
    ref = _normalise(reference) if normalise else reference
    hyp = _normalise(hypothesis) if normalise else hypothesis

    ref_words = ref.split()
    hyp_words = hyp.split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    return _levenshtein(hyp_words, ref_words) / len(ref_words)
