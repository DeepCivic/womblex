"""Tests for the link stage beyond one small CSV: Parquet registers, an entity
under many names, candidate narrowing, several registers, and candidates read
from the text with no enrichment.

The registers here are invented: animals for the entities, and for the places a
few real Australian town names that are also surnames or common words, which is
the case the one-word rules exist for.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from womblex.config import LinkingConfig, ReferenceConfig, ReferenceNamesConfig
from womblex.link.blocking import NameIndex
from womblex.link.matcher import Candidate, Matcher, resolve
from womblex.link.reference import load_reference
from womblex.link.stage import link_shards
from womblex.link.text_candidates import PhraseBook
from womblex.store.entity_links_output import entity_links_path_for, read_entity_links
from womblex.store.output import ELEMENT_SCHEMA, MANIFEST_SCHEMA, TABLE_CELLS_SCHEMA

DOC = "doc1"


def _parquet(path: Path, rows: list[dict]) -> Path:
    pq.write_table(pa.Table.from_pylist(rows), str(path))
    return path


@pytest.fixture
def register(tmp_path: Path) -> ReferenceConfig:
    """Entities in Parquet (with a payload column the loader must not need) + a names table."""
    entities = _parquet(tmp_path / "entities.parquet", [
        {"abn": 11000000001, "legal_name": "WOMBAT HOLDINGS PTY LTD", "geometry": b"\x00\x01"},
        {"abn": 11000000002, "legal_name": "QUOKKA LOGISTICS PTY LTD", "geometry": b"\x00"},
        {"abn": 11000000003, "legal_name": "BILBY EARLY LEARNING LIMITED", "geometry": None},
    ])
    names = _parquet(tmp_path / "names.parquet", [
        {"abn": 11000000001, "name": "Burrow & Co", "name_type": "TRD"},
        {"abn": 11000000001, "name": "Wombat Earthworks", "name_type": "BN"},
        {"abn": 11000000002, "name": "Quokka Couriers", "name_type": "TRD"},
        {"abn": 99999999999, "name": "Nobody Pty Ltd", "name_type": "TRD"},
    ])
    return ReferenceConfig(
        path=entities, id_col="abn", name_col="legal_name", entity_type="organisation",
        match_fuzzy_cols=["legal_name"],
        names=ReferenceNamesConfig(path=names, id_col="abn", name_col="name"),
    )


@pytest.fixture
def gazetteer(tmp_path: Path) -> ReferenceConfig:
    places = _parquet(tmp_path / "places.parquet", [
        {"code": "SAL14494", "name": "Young"}, {"code": "SAL11710", "name": "Grant"},
        {"code": "SAL11329", "name": "Dubbo"}, {"code": "SAL12746", "name": "Wagga Wagga"},
        {"code": "SAL13376", "name": "Richmond"}, {"code": "SAL22169", "name": "Richmond"},
    ])
    return ReferenceConfig(
        path=places, id_col="code", name_col="name", entity_type="place",
        match_fuzzy_cols=["name"], single_word="cued", candidate_kinds=["location"],
    )


# --- one entity, many names --------------------------------------------------


def test_a_parquet_register_reads_only_its_declared_columns(register):
    table = load_reference(register)
    assert [e.entity_id for e in table.entities] == [
        "11000000001", "11000000002", "11000000003"]
    assert table.entities[0].name == "WOMBAT HOLDINGS PTY LTD"


def test_an_entity_matches_under_every_name_it_goes_by(register, caplog):
    with caplog.at_level("WARNING"):
        table = load_reference(register)
    assert table.entity_by_id("11000000001").fuzzy_keys == [
        "wombat holdings pty ltd", "burrow co", "wombat earthworks"]
    assert "1 name row(s)" in caplog.text, "a name for an id the register lacks is counted"

    for mention in ("Wombat Holdings Pty Ltd", "Burrow & Co", "WOMBAT EARTHWORKS"):
        (link,) = resolve([Candidate(mention, "corporate", DOC)], table)
        assert link.matched and link.entity.entity_id == "11000000001", mention


def test_rows_that_repeat_an_id_are_one_entity(tmp_path: Path):
    path = tmp_path / "long.csv"
    path.write_text("id,name\nE1,Numbat Pty Ltd\nE1,Numbat Trading\nE2,Echidna Co\n")
    table = load_reference(ReferenceConfig(
        path=path, id_col="id", name_col="name", match_fuzzy_cols=["name"]))
    assert len(table.entities) == 2
    assert table.entity_by_id("E1").fuzzy_keys == ["numbat pty ltd", "numbat trading"]
    assert table.entity_by_id("E1").name == "Numbat Pty Ltd", "the first row names the entity"


def test_a_declared_column_the_file_lacks_is_refused(tmp_path: Path, register):
    with pytest.raises(ValueError, match=r"no column\(s\) \['trading_name'\]"):
        load_reference(register.model_copy(update={"match_fuzzy_cols": ["trading_name"]}))


# --- narrowing ---------------------------------------------------------------


def _big_register(tmp_path: Path, n: int = 3000) -> ReferenceConfig:
    animals = ["wombat", "quokka", "bilby", "numbat", "echidna", "platypus", "dingo", "koala"]
    trades = ["plumbing", "haulage", "bakery", "surveying", "joinery", "catering", "fencing"]
    rows = [{"id": f"E{i:05d}",
             "name": f"{animals[i % 8]} {trades[i % 7]} {i:05d} pty ltd"} for i in range(n)]
    rows.append({"id": "TARGET", "name": "Bandicoot Early Learning Centre Pty Ltd"})
    return ReferenceConfig(
        path=_parquet(tmp_path / "big.parquet", rows), id_col="id", name_col="name",
        match_fuzzy_cols=["name"])


def test_a_large_register_is_narrowed_and_agrees_with_the_full_scan(tmp_path: Path):
    table = load_reference(_big_register(tmp_path))
    mentions = [
        "Bandicoot Early Learning Centre",          # a name with its suffix dropped
        "Bandicoot Earty Leaming Centre Pty Ltd",   # OCR noise in two words
        "wombat plumbing 00000 pty ltd",            # exact
        "Koala Catering 0O123",                     # no match in either
        "Pty Ltd",                                  # only the words every name shares
    ]
    candidates = [Candidate(m, "corporate", DOC) for m in mentions]
    narrowed = Matcher(table, narrow_above=100)
    full = Matcher(table, narrow_above=10**9)
    assert narrowed._index is not None and full._index is None
    for a, b in zip(narrowed.resolve(candidates), full.resolve(candidates), strict=True):
        assert (a.method, a.entity and a.entity.entity_id, round(a.confidence, 6)) == (
            b.method, b.entity and b.entity.entity_id, round(b.confidence, 6)), a.candidate.text
    assert narrowed.resolve(candidates)[1].entity.entity_id == "TARGET"


def test_narrowing_scores_a_handful_of_names_not_the_register(tmp_path: Path):
    table = load_reference(_big_register(tmp_path))
    index = NameIndex(table.entities)
    near = index.nearest("bandicoot earty leaming centre", 16)
    assert len(index) == 3001 and 0 < len(near) <= 16
    assert near[0][1].entity_id == "TARGET"
    assert index.nearest("zzzz qqqq", 16) == []


# --- names read from the text ------------------------------------------------


def _book(*registers: ReferenceConfig) -> PhraseBook:
    return PhraseBook([(r, load_reference(r)) for r in registers])


def _found(book: PhraseBook, text: str, **kwargs) -> list[tuple[str, str, str | None, str | None]]:
    return [(lk.candidate.text, lk.method, lk.entity and lk.entity.entity_id, lk.detail)
            for lk in book.find(text, DOC, text_layer="elements", **kwargs)]


@pytest.mark.parametrize(("text", "detail"), [
    ("Services in Young were reviewed.", "cue:in"),
    ("Young NSW 2594 was surveyed.", "cue:state"),
    ("The Young district grew.", "cue:district"),
    ("The City of Young met.", "cue:of"),
])
def test_a_one_word_place_links_beside_a_cue(gazetteer, text, detail):
    assert _found(_book(gazetteer), text) == [("Young", "name_exact", "SAL14494", detail)]


@pytest.mark.parametrize(("text", "detail"), [
    ("Mr Young gave evidence.", "person:honorific"),
    ("Peter Young gave evidence.", "person:name"),
    ("A report by Grant Thornton was tabled in Grant Thornton offices.", "person:name"),
    ("Young people were surveyed.", "uncued"),
    ("The use of Grant funds was audited.", "uncued"),
])
def test_a_one_word_place_that_reads_as_a_person_or_a_word_is_not_linked(gazetteer, text, detail):
    found = _found(_book(gazetteer), text)
    assert found and all(f[1:] == ("ambiguous", None, detail) for f in found), found


def test_lowercase_words_are_never_mentions(gazetteer):
    assert _found(_book(gazetteer), "The young were given a grant in dubbo.") == []


def test_a_one_word_place_in_a_heading_links(gazetteer):
    text = "Regional profile: Dubbo\n\nIt has a zoo."
    assert _found(_book(gazetteer), text, heading_spans=[(0, 23)]) == [
        ("Dubbo", "name_exact", "SAL11329", "cue:heading")]
    assert _found(_book(gazetteer), text)[0][1:] == ("ambiguous", None, "uncued")


def test_a_longer_name_needs_no_cue_and_may_wrap_a_line(gazetteer):
    assert _found(_book(gazetteer), "Flooding reached Wagga\nWagga overnight.") == [
        ("Wagga\nWagga", "name_exact", "SAL12746", None)]
    assert _found(_book(gazetteer), "Wagga\n\nWagga") == [], "never across a paragraph break"


def test_a_name_two_entities_share_is_not_linked(gazetteer):
    assert _found(_book(gazetteer), "Flooding in Richmond.") == [
        ("Richmond", "ambiguous", None, "entities:2")]


def test_single_word_policy_is_per_register(gazetteer):
    text = "Young people met."
    accept = gazetteer.model_copy(update={"single_word": "accept"})
    assert _found(_book(accept), text) == [("Young", "name_exact", "SAL14494", None)]
    assert _found(_book(gazetteer.model_copy(update={"single_word": "skip"})), text) == []
    cued = gazetteer.model_copy(update={"cue_words": ["visiting"]})
    assert _found(_book(cued), "While visiting Young they met.")[0][3] == "cue:visiting"


def test_every_name_of_an_entity_is_found_longest_first(register):
    text = "Burrow & Co and Wombat Earthworks are trading names of Wombat Holdings Pty Ltd."
    assert [(t, e) for t, _m, e, _d in _found(_book(register), text)] == [
        ("Burrow & Co", "11000000001"), ("Wombat Earthworks", "11000000001"),
        ("Wombat Holdings Pty Ltd", "11000000001")]


# --- the stage ---------------------------------------------------------------


def _shard(d: Path, texts: list[tuple[str, str]]) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    elements = []
    for order, (kind, text) in enumerate(texts):
        row = {f.name: None for f in ELEMENT_SCHEMA}
        row.update({"source_hash": DOC, "collection_id": "c", "elem_order": order, "kind": kind,
                    "extractor": "test", "confidence": 1.0, "page": 1, "text": text})
        elements.append(row)
    man = {f.name: None for f in MANIFEST_SCHEMA}
    man.update({"source_hash": DOC, "doc_id": DOC, "filename": "doc1.pdf", "status": "ok"})
    pq.write_table(pa.Table.from_pylist([man], schema=MANIFEST_SCHEMA),
                   str(d / "batch-0001._manifest.parquet"))
    pq.write_table(pa.Table.from_pylist(elements, schema=ELEMENT_SCHEMA),
                   str(d / "batch-0001.elements.parquet"))
    pq.write_table(pa.Table.from_pylist([], schema=TABLE_CELLS_SCHEMA),
                   str(d / "batch-0001.table_cells.parquet"))
    return d / "batch-0001.parquet"


def test_text_candidates_link_two_registers_with_no_enrichment(tmp_path, register, gazetteer):
    shards = tmp_path / "documents"
    base = _shard(shards, [
        ("heading", "Regional profile: Dubbo"),
        ("paragraph", "Wombat Earthworks operates in Young. Mr Grant chairs it."),
    ])
    config = LinkingConfig(reference=register, references=[gazetteer], candidate_source="text")

    result = link_shards(shards, config)

    assert (result.total_links, result.matched_links) == (4, 3)
    rows = {r["candidate_text"]: r for r in read_entity_links(base).to_pylist()}
    assert (rows["Dubbo"]["entity_type"], rows["Dubbo"]["match_detail"]) == ("place", "cue:heading")
    assert (rows["Young"]["entity_id"], rows["Young"]["match_detail"]) == ("SAL14494", "cue:in")
    assert rows["Wombat Earthworks"]["entity_type"] == "organisation"
    assert rows["Wombat Earthworks"]["canonical_name"] == "WOMBAT HOLDINGS PTY LTD"
    grant = rows["Grant"]
    assert (grant["matched"], grant["match_method"], grant["match_detail"]) == (
        False, "ambiguous", "person:honorific")
    # Every mention is located to its span in the narrative, and the receipt holds.
    assert {r["anchor_level"] for r in rows.values()} == {"span"}
    assert {r["evidence_check"] for r in rows.values()} == {"holds"}
    assert rows["Dubbo"]["elem_order"] == 0 and rows["Young"]["elem_order"] == 1


def test_enrichment_candidates_go_to_the_register_that_takes_their_kind(
    tmp_path, register, gazetteer,
):
    from womblex.store.enrichment_output import ENTITY_SCHEMA, enrichment_entities_path_for

    shards = tmp_path / "documents"
    base = _shard(shards, [("paragraph", "Quokka Couriers delivers to Dubbo and to Burrow.")])

    def entity(name: str, kind: str) -> dict:
        row = {f.name: None for f in ENTITY_SCHEMA}
        row.update({"source_hash": DOC, "entity_id": name, "name": name, "entity_type": kind})
        return row

    pq.write_table(pa.Table.from_pylist([
        entity("Quokka Couriers", "corporate"), entity("Dubbo", "location"),
        entity("Burrow", "location"),
    ], schema=ENTITY_SCHEMA), str(enrichment_entities_path_for(base)))
    config = LinkingConfig(reference=register, references=[gazetteer],
                           candidate_kinds=["corporate"])

    link_shards(shards, config)

    rows = {r["candidate_text"]: r for r in read_entity_links(base).to_pylist()}
    assert rows["Quokka Couriers"]["entity_id"] == "11000000002"
    assert (rows["Dubbo"]["entity_id"], rows["Dubbo"]["entity_type"]) == ("SAL11329", "place")
    assert rows["Burrow"]["matched"] is False, "a location is never tried against the entity register"
    assert rows["Dubbo"]["match_detail"] is None


def test_a_register_that_cannot_be_read_refuses_before_writing(tmp_path, register):
    shards = tmp_path / "documents"
    base = _shard(shards, [("paragraph", "Wombat Earthworks.")])
    broken = register.model_copy(update={"match_fuzzy_cols": ["absent"]})
    with pytest.raises(ValueError, match="no column"):
        link_shards(shards, LinkingConfig(reference=broken, candidate_source="text"))
    assert not entity_links_path_for(base).exists()


def test_cli_runs_text_candidates_without_an_enrichment_sidecar(tmp_path, register):
    import argparse

    import yaml

    from womblex.cli.link import _register_link, cmd_link

    shards = tmp_path / "run" / "documents"
    base = _shard(shards, [("paragraph", "Wombat Earthworks is here.")])
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "dataset": {"name": "t"},
        "paths": {"input_root": str(tmp_path), "output_root": str(tmp_path),
                  "checkpoint_dir": str(tmp_path / "ck")},
        "linking": {"enabled": True, "candidate_source": "text",
                    "reference": register.model_dump(mode="json")},
    }))
    parser = argparse.ArgumentParser()
    _register_link(parser)

    assert cmd_link(parser.parse_args(["--shards", str(shards), "--config", str(config)])) == 0
    assert [r["entity_id"] for r in read_entity_links(base).to_pylist()] == ["11000000001"]
