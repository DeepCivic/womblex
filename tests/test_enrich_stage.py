"""Tests for the per-stage enrich + link wiring over a shard directory.

Builds a real extraction shard from the **Throsby** ACT FOI childcare notice
(small, native, ~5k chars), runs ``enrich_shards`` against the **live** Isaacus
Kanon-2 enricher (no mocks — real for local validation per CLAUDE.md; skips
cleanly without ``ISAACUS_API_KEY``), then ``link_shards`` against Throsby's
real Education-services register row to confirm the two stages compose: the
provider legal name resolves to the canonical SE-/PR- ids. Also covers
checkpoint skip-on-resume and no-checkpoint-on-failure (the latter via a real
invalid-key client, not a stubbed exception).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from womblex.analyse.enrich_stage import enrich_shards
from womblex.config import EnrichmentConfig, LinkingConfig, ReferenceConfig
from womblex.ingest.detect import DetectionConfig, detect_file_type
from womblex.ingest.extract import extract_text
from womblex.link.stage import link_shards
from womblex.store.checkpoint import CheckpointManager
from womblex.store.enrichment_output import (
    enrichment_entities_path_for,
    read_enrichment_entities,
)
from womblex.store.output import read_entity_links, write_results

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "fixtures"
_THROSBY_PDF = (
    _FIXTURES / "womblex-collection" / "_documents"
    / "00768-213A-270825-Throsby-Out-of-School-Care-"
      "Administrative-Decision-Other-Notice-and-Direction_Redacted.pdf"
)

# Throsby's real row from the ACT Education-services register. Real enrichment
# extracts the provider legal name ("Community Services #1 Incorporated"), which
# fuzzy-resolves to this SE-/PR- pair.
_REGISTER_CSV = (
    "ServiceApprovalNumber,Provider Approval Number,ServiceName,ProviderLegalName,"
    "ServiceAddress,Suburb,Postcode\n"
    "SE-40022307,PR-00005865,Throsby Out of School Hours Care,"
    "Community Services #1 Incorporated,1 Freshwater Street,THROSBY,2914\n"
)


@pytest.fixture
def shard_dir(tmp_path) -> Path:
    if not _THROSBY_PDF.exists():
        pytest.skip(f"fixture not present: {_THROSBY_PDF}")
    d = tmp_path / "documents"
    d.mkdir()
    extraction = extract_text(_THROSBY_PDF, detect_file_type(_THROSBY_PDF, DetectionConfig()))[0]
    write_results([("throsby", str(_THROSBY_PDF), extraction)], d / "batch-0001.parquet",
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
    def test_writes_entities_sidecar(self, shard_dir, isaacus_client):
        result = enrich_shards(shard_dir, EnrichmentConfig(), client=isaacus_client)
        assert result.docs_enriched == 1
        base = shard_dir / "batch-0001.parquet"
        assert enrichment_entities_path_for(base).exists()
        rows = read_enrichment_entities(base).to_pylist()
        assert rows, "real enrichment produced no entities"
        kinds = {r["entity_type"] for r in rows}
        # Throsby notice really contains a corporate provider + a postal address.
        assert "corporate" in kinds and "address" in kinds

    def test_checkpoint_skips_on_resume(self, shard_dir, isaacus_client, tmp_path):
        ckpt = CheckpointManager(tmp_path / ".enrich-ckpt", "t_enrich")
        ckpt.load()
        first = enrich_shards(shard_dir, EnrichmentConfig(), client=isaacus_client,
                              checkpoint_mgr=ckpt)
        assert first.docs_enriched == 1
        # Resume: doc checkpointed → no new enrich calls.
        second = enrich_shards(shard_dir, EnrichmentConfig(), client=isaacus_client,
                               checkpoint_mgr=ckpt)
        assert second.docs_enriched == 0

    def test_transient_failure_not_checkpointed(self, shard_dir, bad_isaacus_client, tmp_path):
        # A real API failure (invalid key) must leave the doc unprocessed so a
        # resume retries it rather than skipping it forever.
        ckpt = CheckpointManager(tmp_path / ".enrich-ckpt", "t_enrich")
        ckpt.load()
        enrich_shards(shard_dir, EnrichmentConfig(), client=bad_isaacus_client,
                      checkpoint_mgr=ckpt)
        assert "throsby" not in ckpt.state.processed_ids


class TestPersistDocumentReuse:
    """Live: the persisted Document round-trips and byte-matches the narrative.

    This is the runtime form of verification gate 1 (docs/decisions.md): the
    chunk stage's reuse guard accepts a Document only when ``document.text``
    equals the reassembled narrative, so the persisted-then-rehydrated text
    must be byte-identical to what enrich/chunk reassemble.
    """

    def test_doc_sidecar_round_trips_and_matches_narrative(self, shard_dir, isaacus_client):
        from womblex.analyse.enrich_stage import _load_narratives
        from womblex.store.enrichment_doc import (
            enrichment_doc_path_for,
            read_enrichment_docs,
        )

        enrich_shards(shard_dir, EnrichmentConfig(), client=isaacus_client,
                      persist_document=True)
        base = shard_dir / "batch-0001.parquet"
        assert enrichment_doc_path_for(base).exists()

        stored = read_enrichment_docs(base)
        assert stored, "persist_document=True produced no doc sidecar rows"

        from isaacus.types.ilgs.v1.document import Document

        narratives, _ = _load_narratives(base, "elements")
        for source_hash, (stamp, doc_json) in stored.items():
            assert stamp == "elements"
            doc = Document.model_validate_json(doc_json)
            assert doc.text == narratives[source_hash], (
                "rehydrated Document.text must byte-match the narrative the "
                "chunk-stage reuse guard reassembles"
            )

    def test_default_writes_no_doc_sidecar(self, shard_dir, isaacus_client):
        from womblex.store.enrichment_doc import enrichment_doc_path_for

        enrich_shards(shard_dir, EnrichmentConfig(), client=isaacus_client)
        assert not enrichment_doc_path_for(shard_dir / "batch-0001.parquet").exists()


class TestEnrichThenLink:
    def test_full_chain_resolves_throsby(self, shard_dir, reference_config, isaacus_client):
        enrich_shards(shard_dir, EnrichmentConfig(), client=isaacus_client)

        cfg = LinkingConfig(enabled=True, reference=reference_config)
        result = link_shards(shard_dir, cfg)
        assert result.docs_linked == 1
        assert result.matched_links >= 1

        doc = read_entity_links(shard_dir, grain="doc").to_pylist()
        assert len(doc) == 1
        # provider legal name resolved to Throsby's canonical service/provider ids
        assert doc[0]["entity_id"] == "SE-40022307"
        assert doc[0]["parent_entity_id"] == "PR-00005865"
