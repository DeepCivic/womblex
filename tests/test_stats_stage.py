"""Tests for the per-stage stats op over a shard directory.

Builds minimal shards (elements + table_cells + manifest) directly and asserts
the sidecar, both loci and their anchors, heading-carried context, checkpoint
resume and the refusals that happen before any batch is written.
"""

from __future__ import annotations

import argparse
from datetime import date
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from womblex.config import StatsConfig
from womblex.process.evidence import verify_shards
from womblex.process.stats_stage import stats_shards
from womblex.process.text_overlay import MissingOverlayError
from womblex.store.checkpoint import CheckpointManager
from womblex.store.output import ELEMENT_SCHEMA, MANIFEST_SCHEMA, TABLE_CELLS_SCHEMA
from womblex.store.stats_output import (
    STAT_SPANS_SCHEMA,
    STAT_SPANS_SUFFIX,
    VOCABULARY_KEY,
    read_stat_spans,
    stat_spans_path_for,
)

DOC = "doc1"
MEASURES = [
    {"id": "median_income", "label": "Median income", "terms": ["median income"]},
    {"id": "population", "terms": ["population"], "unit": "persons"},
]
CONFIG = StatsConfig(enabled=True, measures=MEASURES)


def _element(order: int, kind: str, **kwargs) -> dict:
    row = {f.name: None for f in ELEMENT_SCHEMA}
    row.update({
        "source_hash": DOC, "collection_id": "c", "elem_order": order,
        "kind": kind, "extractor": "test", "confidence": 1.0, "page": 1,
    })
    row.update(kwargs)
    return row


def _cell(parent: int, row_i: int, col: int, value: str) -> dict:
    return {
        "source_hash": DOC, "parent_elem_order": parent, "row": row_i, "col": col,
        "value": value, "rowspan": 1, "colspan": 1, "value_type": "text",
    }


def _build_shard(d: Path, elements: list[dict], cells: list[dict] | None = None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    man = {f.name: None for f in MANIFEST_SCHEMA}
    man.update({"source_hash": DOC, "doc_id": DOC, "filename": "doc1.pdf", "status": "ok"})
    pq.write_table(pa.Table.from_pylist([man], schema=MANIFEST_SCHEMA),
                   str(d / "batch-0001._manifest.parquet"))
    pq.write_table(pa.Table.from_pylist(elements, schema=ELEMENT_SCHEMA),
                   str(d / "batch-0001.elements.parquet"))
    pq.write_table(pa.Table.from_pylist(cells or [], schema=TABLE_CELLS_SCHEMA),
                   str(d / "batch-0001.table_cells.parquet"))
    return d / "batch-0001.parquet"


def _rows(base: Path, kind: str) -> list[dict]:
    return [r for r in read_stat_spans(base).to_pylist() if r["kind"] == kind]


def _profile(tmp_path: Path) -> Path:
    return _build_shard(tmp_path, [
        _element(0, "heading", text="Regional profile: Dubbo"),
        _element(1, "heading", text="2021 Census"),
        _element(2, "paragraph", text="Introduction to the region."),
        _element(3, "heading", text="Income"),
        _element(4, "paragraph", text="Median income $1,242/week. The population was 43,516."),
        _element(5, "footer", text="Population 9,999 page 3"),
    ])


def test_a_figure_is_anchored_and_carries_its_headings(tmp_path: Path):
    base = _profile(tmp_path)

    result = stats_shards(tmp_path, CONFIG)

    assert (result.batches_written, result.figures_written) == (1, 2)
    income, population = _rows(base, "figure")
    assert (income["measure_id"], income["measure_label"]) == ("median_income", "Median income")
    assert (income["value"], income["unit"], income["rate_per"]) == (
        Decimal("1242.0000"), "AUD", "week")
    assert (income["locus"], income["text_layer"], income["elem_order"]) == (
        "narrative", "elements", 4)
    assert income["evidence_check"] == "holds" and income["anchor_level"] == "span"
    assert "Median income" in income["context"]
    # The title run and the current section, outermost first; the earlier section is gone.
    assert income["heading_path"] == ["Regional profile: Dubbo", "2021 Census", "Income"]
    assert income["heading_elem_orders"] == [0, 1, 3]
    assert (population["value"], population["unit"]) == (Decimal("43516.0000"), "persons")


def test_a_figure_inherits_the_period_its_headings_state(tmp_path: Path):
    base = _profile(tmp_path)
    stats_shards(tmp_path, CONFIG)

    for figure in _rows(base, "figure"):
        assert (figure["period"], figure["period_kind"], figure["period_source"]) == (
            "2021", "year", "heading")
        assert (figure["period_start"], figure["period_end"]) == (
            date(2021, 1, 1), date(2021, 12, 31))
    (stated,) = _rows(base, "period")
    assert (stated["text"], stated["elem_order"], stated["period_source"]) == ("2021", 1, "stated")


def test_offsets_index_the_reassembled_narrative(tmp_path: Path):
    from womblex.process.chunk_stage import _load_elements
    from womblex.process.chunker import reassemble_narrative

    base = _profile(tmp_path)
    stats_shards(tmp_path, CONFIG)

    narrative, _ = reassemble_narrative(_load_elements(base)[DOC])
    rows = read_stat_spans(base).to_pylist()
    assert rows and all(narrative[r["char_start"]:r["char_end"]] == r["text"] for r in rows)
    assert verify_shards(tmp_path).ok


def test_page_furniture_is_not_scanned(tmp_path: Path):
    base = _profile(tmp_path)
    stats_shards(tmp_path, CONFIG)
    assert all(r["elem_order"] != 5 for r in read_stat_spans(base).to_pylist())


def _table_shard(d: Path, cells: list[dict], header_rows: list[int]) -> Path:
    return _build_shard(d, [
        _element(0, "heading", text="Dubbo, 2016 Census"),
        _element(1, "table", header_rows=header_rows),
    ], cells)


def test_a_table_row_that_names_a_measure(tmp_path: Path):
    """The column header states each figure's period; the headings only where it does not."""
    base = _table_shard(tmp_path, [
        _cell(1, 0, 0, "Measure"), _cell(1, 0, 1, "2019–20"), _cell(1, 0, 2, "Change"),
        _cell(1, 1, 0, "Median income ($/week)"), _cell(1, 1, 1, "$1,242"), _cell(1, 1, 2, "4.5%"),
        _cell(1, 2, 0, "Population"), _cell(1, 2, 1, "n.p."), _cell(1, 2, 2, "-1.2%"),
        _cell(1, 3, 0, "Dwellings"), _cell(1, 3, 1, "18,000"), _cell(1, 3, 2, "2%"),
    ], header_rows=[0])

    stats_shards(tmp_path, CONFIG)

    figures = {(r["cell_row"], r["cell_col"]): r for r in _rows(base, "figure")}
    assert set(figures) == {(1, 1), (1, 2), (2, 2)}, "a row naming no measure yields nothing"
    assert (figures[1, 1]["value"], figures[1, 1]["unit"]) == (Decimal("1242.0000"), "AUD")
    assert (figures[1, 1]["period"], figures[1, 1]["period_source"]) == ("2019-20", "column")
    assert (figures[1, 2]["period"], figures[1, 2]["period_source"]) == ("2016", "heading")
    assert (figures[2, 2]["value"], figures[2, 2]["unit"]) == (Decimal("-1.2000"), "percent")
    row = figures[1, 1]
    assert (row["locus"], row["text_layer"], row["elem_order"]) == ("table_cell", "cell", 1)
    assert (row["char_start"], row["char_end"], row["evidence_check"]) == (0, 6, "holds")
    (gap,) = _rows(base, "cue")
    assert (gap["cue_type"], gap["cue"], gap["measure_id"]) == ("gap", "not_published", "population")
    assert (gap["cell_row"], gap["cell_col"]) == (2, 1)


def test_a_table_column_that_names_a_measure(tmp_path: Path):
    """Read the other way: the header names the measure and the row label the period."""
    base = _table_shard(tmp_path, [
        _cell(1, 0, 0, "Year"), _cell(1, 0, 1, "Population"), _cell(1, 0, 2, "Notes"),
        _cell(1, 1, 0, "2020-21"), _cell(1, 1, 1, "43,516"), _cell(1, 1, 2, "12"),
        _cell(1, 2, 0, "2021-22"), _cell(1, 2, 1, "44,010"), _cell(1, 2, 2, "13"),
    ], header_rows=[0])

    stats_shards(tmp_path, CONFIG)

    figures = _rows(base, "figure")
    assert [(r["cell_row"], r["cell_col"], r["value"], r["period"], r["period_source"])
            for r in figures] == [
        (1, 1, Decimal("43516.0000"), "2020-21", "row"),
        (2, 1, Decimal("44010.0000"), "2021-22", "row"),
    ]
    assert {r["unit"] for r in figures} == {"persons"}


def test_tables_and_narrative_can_each_be_switched_off(tmp_path: Path):
    base = _build_shard(tmp_path, [
        _element(0, "paragraph", text="Population 43,516."),
        _element(1, "table", header_rows=[0]),
    ], [_cell(1, 0, 0, "Measure"), _cell(1, 0, 1, "2021"),
        _cell(1, 1, 0, "Population"), _cell(1, 1, 1, "43,516")])

    stats_shards(tmp_path, CONFIG.model_copy(update={"tables": False}))
    assert {r["locus"] for r in read_stat_spans(base).to_pylist()} == {"narrative"}
    stats_shards(tmp_path, CONFIG.model_copy(update={"narrative": False}))
    assert {r["locus"] for r in read_stat_spans(base).to_pylist()} == {"table_cell"}


def test_the_footer_names_the_vocabulary_that_produced_the_file(tmp_path: Path):
    base = _profile(tmp_path)
    stats_shards(tmp_path, CONFIG)
    first = pq.read_schema(str(stat_spans_path_for(base))).metadata[VOCABULARY_KEY]

    stats_shards(tmp_path, StatsConfig(enabled=True, measures=MEASURES[:1]))
    second = pq.read_schema(str(stat_spans_path_for(base))).metadata[VOCABULARY_KEY]

    assert first.startswith(b"sha256:") and first != second


def test_no_measures_writes_periods_and_cues_only(tmp_path: Path, caplog):
    base = _profile(tmp_path)
    with caplog.at_level("WARNING"):
        stats_shards(tmp_path, StatsConfig(enabled=True))
    assert "no measures declared" in caplog.text
    assert {r["kind"] for r in read_stat_spans(base).to_pylist()} == {"period"}


def test_an_empty_batch_writes_a_schema_correct_sidecar(tmp_path: Path):
    base = _build_shard(tmp_path, [_element(0, "paragraph", text="nothing to see")])
    stats_shards(tmp_path, CONFIG)
    spans = read_stat_spans(base)
    assert spans.num_rows == 0 and spans.schema.names == STAT_SPANS_SCHEMA.names


def test_a_missing_declared_overlay_refuses_before_writing(tmp_path: Path):
    base = _profile(tmp_path)
    with pytest.raises(MissingOverlayError, match="batch-0001"):
        stats_shards(tmp_path, CONFIG, text_source="normalised")
    assert not stat_spans_path_for(base).exists()


def test_a_missing_vocabulary_refuses_before_writing(tmp_path: Path):
    base = _profile(tmp_path)
    with pytest.raises(FileNotFoundError):
        stats_shards(tmp_path, StatsConfig(enabled=True, vocabulary=tmp_path / "absent.yaml"))
    assert not stat_spans_path_for(base).exists()


def test_checkpoint_skips_a_second_run_and_heals_a_corrupt_sidecar(tmp_path: Path):
    from womblex.store.shard_audit import reconcile_stage_checkpoint_with_shards

    _profile(tmp_path)
    ckpt = CheckpointManager(tmp_path / ".stats-checkpoint", "ds_stats")
    assert stats_shards(tmp_path, CONFIG, checkpoint_mgr=ckpt).batches_written == 1
    assert stats_shards(tmp_path, CONFIG, checkpoint_mgr=ckpt).batches_written == 0

    (tmp_path / f"batch-0001{STAT_SPANS_SUFFIX}").write_bytes(b"")
    assert reconcile_stage_checkpoint_with_shards(ckpt, tmp_path, suffix=STAT_SPANS_SUFFIX) == [DOC]
    assert stats_shards(tmp_path, CONFIG, checkpoint_mgr=ckpt).batches_written == 1


def _cli(*argv: str) -> int:
    from womblex.cli.stats import _register_stats, cmd_stats

    parser = argparse.ArgumentParser()
    _register_stats(parser)
    return cmd_stats(parser.parse_args(list(argv)))


def test_cli_runs_with_a_vocabulary_file_and_refuses_a_missing_one(tmp_path: Path):
    shards = tmp_path / "documents"
    base = _profile(shards)
    vocab = tmp_path / "vocab.yaml"
    vocab.write_text("measures:\n  - id: population\n    terms: [population]\n")

    assert _cli("--shards", str(shards), "--vocabulary", str(tmp_path / "absent.yaml")) == 1
    assert not stat_spans_path_for(base).exists()
    assert _cli("--shards", str(shards), "--vocabulary", str(vocab)) == 0
    assert [r["measure_id"] for r in _rows(base, "figure")] == ["population"]
    assert _cli("--shards", str(shards), "--text-source", "normalised", "--no-resume") == 1
