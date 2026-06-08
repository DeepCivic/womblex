"""Tests for the dictionary-gated OCR character-confusion repair op.

Unit tests for the pure :func:`womblex.process.spellfix.repair_text` corrector,
plus a per-stage test that builds a minimal chunks shard directly and asserts
the ``*.chunks_repaired.parquet`` + ``*.spellfix_corrections.parquet`` siblings.
Runs offline against the bundled en_AU Hunspell dictionary (needs ``spylls``).
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("spylls")

from womblex.config import SpellfixConfig  # noqa: E402
from womblex.process.spellfix import repair_text  # noqa: E402
from womblex.process.spellfix_stage import spellfix_shards  # noqa: E402
from womblex.store.output import CHUNKS_SCHEMA, MANIFEST_SCHEMA  # noqa: E402
from womblex.store.spellfix_output import (  # noqa: E402
    chunks_repaired_path_for,
    read_repaired_chunks,
    read_spellfix_corrections,
    spellfix_corrections_path_for,
)

DOC = "doc1"


# ---------------------------------------------------------------------------
# Corrector (Tier A homoglyph, default)
# ---------------------------------------------------------------------------


def test_homoglyph_fixes_digit_for_letter():
    fixed, corr = repair_text("The chi1d went home.")
    assert fixed == "The child went home."
    assert len(corr) == 1
    assert (corr[0].original, corr[0].corrected, corr[0].method) == ("chi1d", "child", "homoglyph")


def test_homoglyph_is_length_preserving():
    text = "A good p1an indeed."
    fixed, _ = repair_text(text)
    assert fixed == "A good plan indeed."
    assert len(fixed) == len(text)


def test_valid_word_is_never_touched():
    # Australian spelling is in the dictionary — must pass through untouched.
    fixed, corr = repair_text("The colour of our neighbourhood.")
    assert fixed == "The colour of our neighbourhood."
    assert corr == []


def test_codes_and_ids_left_alone():
    # No single homoglyph swap yields a dictionary word -> no change.
    for text in ("Section3 of the Act.", "FY2024 budget B2B."):
        fixed, corr = repair_text(text)
        assert fixed == text
        assert corr == []


def test_all_caps_acronym_skipped():
    fixed, corr = repair_text("The CH1LD record.")  # all-caps token guarded
    assert fixed == "The CH1LD record."
    assert corr == []


def test_no_digit_no_homoglyph_change():
    # Pure letter typo is out of Tier A scope (no confusable char).
    fixed, corr = repair_text("teh cat sat.")
    assert fixed == "teh cat sat."
    assert corr == []


# ---------------------------------------------------------------------------
# Tier B (general edit-distance-1, opt-in) + ambiguity gate
# ---------------------------------------------------------------------------


def test_ambiguous_token_left_verbatim_even_in_tier_b():
    # "teh" is one edit from several dictionary words (the/ten/tea/...);
    # the unambiguity gate must leave it unchanged.
    fixed, corr = repair_text("teh cat sat.", general_edits=True)
    assert fixed == "teh cat sat."
    assert corr == []


def test_in_dictionary_proper_noun_untouched_under_tier_b():
    fixed, _ = repair_text("Standish attended.", general_edits=True)
    assert fixed == "Standish attended."


# ---------------------------------------------------------------------------
# Per-stage driver
# ---------------------------------------------------------------------------


def _build_chunks_shard(d: Path, chunk_texts: list[str]) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    base = d / "batch-0001.parquet"
    man = {f.name: None for f in MANIFEST_SCHEMA}
    man.update({"source_hash": DOC, "doc_id": DOC, "filename": "doc1.pdf", "status": "ok"})
    pq.write_table(
        pa.Table.from_pylist([man], schema=MANIFEST_SCHEMA),
        str(d / "batch-0001._manifest.parquet"),
    )
    rows = []
    for i, text in enumerate(chunk_texts):
        row = {f.name: None for f in CHUNKS_SCHEMA}
        row.update({
            "source_hash": DOC, "chunk_index": i, "text": text,
            "start_char": 0, "end_char": len(text), "content_type": "narrative",
            "has_redaction": False, "page_start": 1, "page_end": 1,
        })
        rows.append(row)
    pq.write_table(
        pa.Table.from_pylist(rows, schema=CHUNKS_SCHEMA),
        str(d / "batch-0001.chunks.parquet"),
    )
    return base


def test_spellfix_shards_writes_repaired_and_audit(tmp_path: Path):
    base = _build_chunks_shard(tmp_path, ["The chi1d is fine.", "All good here."])
    result = spellfix_shards(tmp_path, SpellfixConfig(enabled=True))

    assert result.batches_written == 1
    assert result.chunks_repaired == 1          # only the first chunk changed
    assert result.corrections_applied == 1

    repaired = read_repaired_chunks(chunks_repaired_path_for(base))
    texts = repaired.column("text").to_pylist()
    assert "The child is fine." in texts
    assert "All good here." in texts            # passthrough chunk retained

    audit = read_spellfix_corrections(spellfix_corrections_path_for(base))
    assert audit.num_rows == 1
    assert audit.column("original").to_pylist() == ["chi1d"]
    assert audit.column("corrected").to_pylist() == ["child"]


def test_spellfix_shards_leaves_raw_chunks_untouched(tmp_path: Path):
    _build_chunks_shard(tmp_path, ["The chi1d is fine."])
    spellfix_shards(tmp_path, SpellfixConfig(enabled=True))

    raw = pq.read_table(str(tmp_path / "batch-0001.chunks.parquet"))
    assert raw.column("text").to_pylist() == ["The chi1d is fine."]
