"""`womblex plan`: what a workflow would run and spend, offline, the shard directory untouched."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pytest

from tests.test_workflow_runner import _PRESET, _resolve, _stored, shards, write_workflow  # noqa: F401
from womblex.analyse.embed_stage import plan_embedding
from womblex.analyse.enrich_stage import _load_narratives
from womblex.cloud.stage_contracts import STAGE_CONTRACTS
from womblex.cloud.workflow_plan import plan_workflow
from womblex.config import EnrichmentConfig
from womblex.process.chunk_stage import _batch_bases
from womblex.utils.token_packer import TokenCounter

_NORMALISED = {"processing": {"text_source": "normalised"}}
_STEPS = ["normalise", "chunk", "enrich", "embed"]


@pytest.fixture(autouse=True)
def _no_paid_path(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a, **_k):
        raise AssertionError("plan reached a paid or model-probing path")

    monkeypatch.setattr("womblex.providers.isaacus.client.make_isaacus_client", boom)
    monkeypatch.setattr("womblex.utils.model_check.check_models", boom)


def _snapshot(d: Path) -> dict[str, int]:
    return {p.name: p.stat().st_mtime_ns for p in d.iterdir()}


def test_counts_are_the_stages_own_and_the_directory_is_untouched(shards: Path) -> None:  # noqa: F811
    before = _snapshot(shards)
    plan = plan_workflow(_stored(_STEPS, config=_NORMALISED), shards, resolve_preset=_resolve)
    assert _snapshot(shards) == before  # no normalised overlay, no chunks written
    assert [s.action for s in plan.steps] == ["run"] * 4
    enrich, embed = plan.steps[2].tokens, plan.steps[3].tokens
    assert enrich is not None and embed is not None
    assert enrich.basis == embed.basis == "simulated" and enrich.tokens > 0 and embed.tokens > 0

    counter = TokenCounter(EnrichmentConfig().tokenizer)  # the oracle: the same text counted directly
    narrative, _ = _load_narratives(_batch_bases(shards)[0], "elements")  # normalise is a passthrough here
    assert enrich.detail["narrative_tokens"] == sum(counter.count_batch(list(narrative.values())))


def test_embed_count_equals_a_real_chunk_run(shards: Path, tmp_path: Path) -> None:  # noqa: F811
    from womblex.cloud.stage_contracts import RunContext
    from womblex.cloud.workflow_runner import resolve_workflow_config

    stored = _stored(["chunk", "embed"])
    plan = plan_workflow(stored, shards, resolve_preset=_resolve)
    STAGE_CONTRACTS["chunk"].run(shards, resolve_workflow_config(stored, _resolve), RunContext())
    real = plan_embedding(shards)
    got = plan.steps[1].tokens
    assert got is not None and (got.tokens, got.requests, got.texts) == (real.tokens, real.requests, real.chunks)


def test_a_step_that_cannot_run_offline_leaves_the_count_pending(
    shards: Path, monkeypatch: pytest.MonkeyPatch,  # noqa: F811
) -> None:
    monkeypatch.setitem(STAGE_CONTRACTS, "chunk", replace(
        STAGE_CONTRACTS["chunk"], needs_isaacus_api_for=lambda _c: True))
    plan = plan_workflow(_stored(["chunk", "embed"]), shards, resolve_preset=_resolve, prices={"embed": 2.0})
    t = plan.steps[1].tokens
    assert t is not None and t.basis == "pending" and "'chunk'" in t.note and t.tokens == 0
    assert t.assumed_cost is None  # no count, so no $0.00


def test_price_appears_only_when_assumed(shards: Path) -> None:  # noqa: F811
    stored = _stored(["chunk", "embed"])
    assert plan_workflow(stored, shards, resolve_preset=_resolve).steps[1].tokens.assumed_cost is None  # type: ignore[union-attr]
    priced = plan_workflow(stored, shards, resolve_preset=_resolve, prices={"embed": 2.0}).steps[1].tokens
    assert priced is not None and priced.assumed_cost == pytest.approx(priced.tokens / 1e6 * 2.0)
    assert priced.assumed_price_per_million == 2.0


def test_cli_json_prices_and_exit_codes(shards: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:  # noqa: F811
    from womblex.cli.workflow import cmd_plan
    from womblex.presets import build_preset_record, preset_filename, serialise_preset_record

    write_workflow(tmp_path, "w", ["chunk", "embed"])
    record, _ = build_preset_record(name=_PRESET.name, description="", formats=(), config=_PRESET.config)
    (tmp_path / preset_filename(_PRESET.name)).write_text(serialise_preset_record(record))

    def args(**kw) -> argparse.Namespace:
        base = {"workflow": tmp_path / "w.yaml", "shards": shards, "presets_dir": tmp_path,
                "models_check": None, "assume_price": [], "json": True}
        return argparse.Namespace(**{**base, **kw})

    assert cmd_plan(args()) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["steps"][1]["tokens"]["basis"] == "simulated"
    assert cmd_plan(args(json=False, assume_price=["embed=3"])) == 0
    assert "ASSUMPTION" in capsys.readouterr().out
    assert cmd_plan(args(assume_price=["money=1"])) == 1
    write_workflow(tmp_path, "refused", ["embed"])  # chunks are not on disk
    for name, code in (("absent", "workflow_unavailable"), ("refused", "step_order")):  # still one JSON object
        capsys.readouterr()
        assert cmd_plan(args(workflow=tmp_path / f"{name}.yaml")) == 1 and json.loads(capsys.readouterr().out)["code"] == code


def test_layout_skips_only_when_every_sidecar_carries_the_fingerprint(shards: Path) -> None:  # noqa: F811
    from womblex.cloud.workflow_runner import resolve_workflow_config
    from womblex.store.layout_output import layout_fingerprint, write_layout_regions

    stored = _stored(["layout"])
    fp = layout_fingerprint(resolve_workflow_config(stored, _resolve))
    for dpi, action in ((fp.dpi + 1, "run"), (fp.dpi, "skip")):
        for base in _batch_bases(shards):
            write_layout_regions([], base, replace(fp, dpi=dpi), redaction_consumed=False)
        assert plan_workflow(stored, shards, resolve_preset=_resolve).steps[0].action == action


def test_local_steps_are_free_and_paid_ones_are_counted(shards: Path) -> None:  # noqa: F811
    plan = plan_workflow(_stored(["normalise", "chunk", "embed"], config=_NORMALISED), shards, resolve_preset=_resolve)
    assert [(s.stage, s.paid) for s in plan.steps] == [("normalise", False), ("chunk", False), ("embed", True)]
    assert plan.steps[1].tokens is None and plan.steps[2].tokens is not None


def _ai_chunking(chunk_size: int) -> object:
    return _stored(["chunk"], config={"chunking": {"chunking_model": "kanon-2-enricher", "chunk_size": chunk_size},
                                      "enrichment": {"enabled": True}})


@pytest.fixture
def docx_shards(tmp_path: Path) -> Path:
    """One batch holding a document with a narrative and tables (the register CSV has no narrative)."""
    from tests._synthetic import BUDGET_DOCX
    from womblex.batch import process_batch
    from womblex.config import DatasetConfig, PathsConfig, WomblexConfig

    out = tmp_path / "docx"
    out.mkdir()
    process_batch([BUDGET_DOCX], WomblexConfig(dataset=DatasetConfig(name="t"), paths=PathsConfig(
        input_root=tmp_path, output_root=tmp_path, checkpoint_dir=tmp_path)), batch_num=1, shard_dir=out)
    return out


def _narratives(shard_dir: Path) -> dict[str, str]:
    """Each document's non-blank narrative, as the chunk stage builds it."""
    from womblex.process.chunk_stage import _build_inputs_for_batch

    return {doc.source_hash: doc.narrative for base in _batch_bases(shard_dir)
            for doc in _build_inputs_for_batch(base, True)[0] if doc.narrative.strip()}


def test_ai_chunking_counts_each_narrative_over_the_chunk_size_in_the_enrichers_own_tokens(docx_shards: Path) -> None:
    """semchunk enriches a narrative only when it exceeds the chunk size; tables chunk in token mode (Phase R, D12)."""
    texts = list(_narratives(docx_shards).values())
    (step,) = plan_workflow(_ai_chunking(1), docx_shards, resolve_preset=_resolve, prices={"chunk": 4.0}).steps
    t = step.tokens
    assert step.paid and t is not None and t.basis == "on_disk" and texts
    assert t.requests == len(texts) and t.detail["reused_documents"] == 0  # every narrative is over one token
    assert t.tokens == sum(TokenCounter(EnrichmentConfig().tokenizer).count_batch(texts))
    assert t.assumed_cost == pytest.approx(t.tokens / 1e6 * 4.0)
    (step,) = plan_workflow(_ai_chunking(100_000), docx_shards, resolve_preset=_resolve).steps
    assert step.tokens is not None and step.tokens.requests == step.tokens.tokens == 0  # all fit: token splitting


@pytest.mark.parametrize("readable", [True, False])
def test_ai_chunking_reuses_only_a_persisted_document_the_stage_would(
    docx_shards: Path, monkeypatch: pytest.MonkeyPatch, readable: bool,
) -> None:
    """The stage's reuse test: a rehydrated Document byte-identical to the narrative is reused; one the
    stage cannot rehydrate is enriched again, so it is counted."""
    from womblex.process import chunk_stage
    from womblex.store.enrichment_doc import write_enrichment_doc_shard

    narratives = _narratives(docx_shards)
    chosen = next(iter(narratives))
    if readable:
        monkeypatch.setattr(chunk_stage, "_load_narrative_overrides",
                            lambda _b, _s: {chosen: type("Doc", (), {"text": narratives[chosen]})()})
    else:
        write_enrichment_doc_shard([(chosen, "elements", "{not a document")], _batch_bases(docx_shards)[0])
    t = plan_workflow(_ai_chunking(1), docx_shards, resolve_preset=_resolve).steps[0].tokens
    assert t is not None and t.detail["reused_documents"] == int(readable)
    assert t.requests == len(narratives) - int(readable)


def test_an_embedding_provider_with_no_token_counter_is_counted_by_texts(
    shards: Path, monkeypatch: pytest.MonkeyPatch,  # noqa: F811
) -> None:
    from womblex.analyse import embed

    real = embed.make_embedding_provider

    def uncounted(config, client=None):
        provider = real(config, client)
        return replace(provider, limits=replace(provider.limits, token_counter=None))

    monkeypatch.setattr(embed, "make_embedding_provider", uncounted)
    monkeypatch.setattr("womblex.analyse.embed_stage.make_embedding_provider", uncounted)
    t = plan_workflow(_stored(["chunk", "embed"]), shards, resolve_preset=_resolve, prices={"embed": 2.0}).steps[1].tokens
    assert t is not None and t.tokens is None and t.texts > 0 and "no token counter" in t.note
    assert t.assumed_cost is None
