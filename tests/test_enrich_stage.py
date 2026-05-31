"""Tests for the per-stage enrich + link wiring over a shard directory.

Builds a real extraction shard from the budget-statement DOCX fixture,
mocks the Kanon-2 call with a canned enrichment result (so no API/key is
needed in CI), runs ``enrich_shards`` to produce the entities sidecar, then
runs ``link_shards`` against a real-valued Artemis register to confirm the
two stages compose: a corporate name + service address resolve to the
canonical SE-/PR- ids. Also covers checkpoint skip-on-resume.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from womblex.analyse.enrich_stage import enrich_shards
from womblex.analyse.models import EnrichmentResult, Location, Person, Span
from womblex.config import EnrichmentConfig, LinkingConfig, ReferenceConfig
from womblex.ingest.strategies_file import DocxExtractor
from womblex.link.stage import link_shards
from womblex.store.checkpoint import CheckpointManager
from womblex.store.enrichment_output import (
    enrichment_entities_path_for,
    read_enrichment_entities,
)
from womblex.store.output import read_entity_links, write_results

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "fixtures"
_BUDGET_DOCX = (
    _FIXTURES / "womblex-collection" / "_documents"
    / "foreign-affairs-and-trade-2025-26-portfolio-budget-statements.docx"
)

_REGISTER_CSV = (
    "ServiceApprovalNumber,Provider Approval Number,ServiceName,ProviderLegalName,"
    "ServiceAddress,Suburb,Postcode\n"
    "SE-40002132,PR-40030037,Artemis Early Learning,ARTEMIS EDUCATION PTY LTD,"
    "11 Cessnock St,FYSHWICK,2609\n"
)


def _canned_enrichment() -> EnrichmentResult:
    legal = "ARTEMIS EDUCATION PTY LTD"
    addr = "11 Cessnock St Fyshwick 2609"
    text = f"Notice issued to {legal}, trading as Artemis Early Learning, at {addr}."
    li, le = text.index(legal), text.index(legal) + len(legal)
    ai, ae = text.index(addr), text.index(addr) + len(addr)
    return EnrichmentResult(
        text=text, type="other",
        persons=[Person(id="p1", name=Span(li, le), type="corporate",
                        role="respondent", mentions=[Span(li, le)])],
        locations=[Location(id="l1", name=Span(ai, ae), type="address",
                            mentions=[Span(ai, ae)])],
    )


@pytest.fixture
def shard_dir(tmp_path) -> Path:
    if not _BUDGET_DOCX.exists():
        pytest.skip(f"fixture not present: {_BUDGET_DOCX}")
    d = tmp_path / "documents"
    d.mkdir()
    extraction = DocxExtractor().extract_path(_BUDGET_DOCX)
    write_results([("budget", str(_BUDGET_DOCX), extraction)], d / "batch-0001.parquet",
                  collection_id="test")
    return d


@pytest.fixture
def reference_config(tmp_path) -> ReferenceConfig:
    csv_path = tmp_path / "services.csv"
    csv_path.write_text(_REGISTER_CSV, encoding="utf-8")
    return ReferenceConfig(
        path=csv_path, id_col="ServiceApprovalNumber", name_col="ServiceName",
        entity_type="service", parent_id_col="Provider Approval Number",
        match_exact_cols=["ServiceAddress", "Suburb", "Postcode"],
        match_fuzzy_cols=["ServiceName", "ProviderLegalName"],
    )


class TestEnrichShards:
    def test_writes_entities_sidecar(self, shard_dir, monkeypatch):
        monkeypatch.setattr(
            "womblex.analyse.enrich_stage.enrich_document",
            lambda *a, **k: _canned_enrichment(),
        )
        result = enrich_shards(shard_dir, EnrichmentConfig(), client=object())
        assert result.docs_enriched == 1
        base = shard_dir / "batch-0001.parquet"
        assert enrichment_entities_path_for(base).exists()
        rows = read_enrichment_entities(base).to_pylist()
        kinds = {r["entity_type"] for r in rows}
        assert "corporate" in kinds and "address" in kinds

    def test_checkpoint_skips_on_resume(self, shard_dir, monkeypatch, tmp_path):
        calls = {"n": 0}

        def _fake(*a, **k):
            calls["n"] += 1
            return _canned_enrichment()

        monkeypatch.setattr("womblex.analyse.enrich_stage.enrich_document", _fake)
        ckpt = CheckpointManager(tmp_path / ".enrich-ckpt", "t_enrich")
        ckpt.load()
        enrich_shards(shard_dir, EnrichmentConfig(), client=object(), checkpoint_mgr=ckpt)
        first = calls["n"]
        assert first == 1
        # Second pass: sidecar exists + doc checkpointed -> no new enrich calls.
        enrich_shards(shard_dir, EnrichmentConfig(), client=object(), checkpoint_mgr=ckpt)
        assert calls["n"] == first

    def test_transient_failure_not_checkpointed(self, shard_dir, monkeypatch, tmp_path):
        # A connection-style failure must leave the doc unprocessed so a
        # resume retries it (not silently skipped forever).
        def _boom(*a, **k):
            raise RuntimeError("Enrichment failed: Connection error.")

        monkeypatch.setattr("womblex.analyse.enrich_stage.enrich_document", _boom)
        ckpt = CheckpointManager(tmp_path / ".enrich-ckpt", "t_enrich")
        ckpt.load()
        enrich_shards(shard_dir, EnrichmentConfig(), client=object(), checkpoint_mgr=ckpt)
        # the single budget doc errored -> nothing checkpointed
        assert "budget" not in ckpt.state.processed_ids


class TestEnrichThenLink:
    def test_full_chain_resolves_artemis(self, shard_dir, reference_config, monkeypatch):
        monkeypatch.setattr(
            "womblex.analyse.enrich_stage.enrich_document",
            lambda *a, **k: _canned_enrichment(),
        )
        enrich_shards(shard_dir, EnrichmentConfig(), client=object())

        cfg = LinkingConfig(enabled=True, reference=reference_config)
        result = link_shards(shard_dir, cfg)
        assert result.docs_linked == 1
        assert result.matched_links >= 1

        doc = read_entity_links(shard_dir, grain="doc").to_pylist()
        assert len(doc) == 1
        assert doc[0]["entity_id"] == "SE-40002132"
        assert doc[0]["parent_entity_id"] == "PR-40030037"
        # both the corporate-name and address mentions resolved to the same entity
        assert doc[0]["mention_count"] == 2
