"""Tests for the remote per-batch shard-stage runner.

Everything runs against fsspec's local backend, like ``test_cloud.py`` — no S3,
no Postgres, no network. The mechanical runner behaviours (discovery, skip,
all-or-nothing publish, not-ready, in-place, whole-run) are exercised through
synthetic contracts so they test the *runner* rather than any one stage's
dependencies; ``normalise`` (offline, no API) carries the real end-to-end and
local/remote parity checks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

pytest.importorskip("fsspec")

from womblex.cli.cloud import RUN_STAGE_CHOICES, cmd_run_stage
from womblex.cloud.stage_contracts import (
    STAGE_CONTRACTS,
    STAGE_NAMES,
    ConditionalInput,
    MutationMode,
    RunContext,
    StageContract,
    StageScope,
)
from womblex.cloud.stage_runner import remote_bases, run_stage_remote
from womblex.config import WomblexConfig
from womblex.store.remote import RemoteStore

MANIFEST = "._manifest.parquet"
ELEMENTS = ".elements.parquet"
CELLS = ".table_cells.parquet"
FORM_FIELDS = ".form_fields.parquet"


def _config(**kw) -> WomblexConfig:
    base = {
        "dataset": {"name": "t"},
        "paths": {"input_root": ".", "output_root": ".", "checkpoint_dir": "."},
    }
    base.update(kw)
    return WomblexConfig(**base)


def _seed(store: RemoteStore, tmp_path: Path, prefix: str, names: list[str]) -> None:
    """Upload placeholder parquet keys so presence/discovery can be exercised."""
    for name in names:
        local = tmp_path / "seed" / name
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(name.encode())
        store.upload_file(local, f"{prefix}/{name}")


# ---------------------------------------------------------------------------
# Contract table
# ---------------------------------------------------------------------------


def test_cli_choices_match_the_contract_table():
    """The CLI spells the stage names so `--help` skips the pyarrow import."""
    assert RUN_STAGE_CHOICES == STAGE_NAMES


def test_manifest_is_not_a_runner_stage():
    """`womblex finalize` already downloads, consolidates and uploads it."""
    assert "manifest" not in STAGE_CONTRACTS


def test_inputs_and_outputs_disjoint_except_in_place_mutators():
    cfg = _config()
    for name, contract in STAGE_CONTRACTS.items():
        ins = set(contract.input_suffixes(cfg))
        outs = set(contract.outputs(cfg))
        if contract.mutation is MutationMode.SIDECAR:
            assert not (ins & outs), f"{name} overlaps but claims to be a sidecar producer"
        else:
            assert outs <= ins, f"{name} is IN_PLACE but its outputs are not a subset of inputs"


def test_graph_refresh_is_the_only_in_place_mutator():
    in_place = {n for n, c in STAGE_CONTRACTS.items() if c.mutation is MutationMode.IN_PLACE}
    assert in_place == {"graph-refresh"}


def test_quality_is_the_only_whole_run_stage():
    whole = {n for n, c in STAGE_CONTRACTS.items() if c.scope is StageScope.WHOLE_RUN}
    assert whole == {"quality"}
    # It keeps no checkpoint: a partial corpus would change the dup clusters.
    assert STAGE_CONTRACTS["quality"].checkpoint_dirname is None


def test_form_fields_is_discovery_only():
    """No stage reads it, so the runner must never download it."""
    cfg = _config()
    for name, contract in STAGE_CONTRACTS.items():
        assert FORM_FIELDS not in contract.input_suffixes(cfg), name
        assert FORM_FIELDS not in contract.outputs(cfg), name


# ---------------------------------------------------------------------------
# Conditional inputs are a function of config, not of stage name
# ---------------------------------------------------------------------------


def test_chunk_reads_enrichment_doc_only_with_a_chunking_model():
    chunk = STAGE_CONTRACTS["chunk"]
    assert ".enrichment_doc.parquet" not in chunk.input_suffixes(_config())
    ai = _config(chunking={"chunking_model": "kanon-2-chunker"})
    assert ".enrichment_doc.parquet" in chunk.input_suffixes(ai)


@pytest.mark.parametrize(
    ("text_source", "expected"),
    [
        ("elements", None),
        ("normalised", ".normalised_text.parquet"),
        ("spellfix", ".spellfix_text.parquet"),
    ],
)
def test_text_source_selects_the_overlay_input(text_source, expected):
    cfg = _config(processing={"text_source": text_source})
    for stage in ("chunk", "enrich", "money"):
        suffixes = STAGE_CONTRACTS[stage].input_suffixes(cfg)
        for candidate in (".normalised_text.parquet", ".spellfix_text.parquet"):
            assert (candidate in suffixes) is (candidate == expected), (stage, candidate)


def test_money_text_source_outranks_the_pipeline_level_one():
    cfg = _config(
        processing={"text_source": "normalised"}, money={"text_source": "spellfix"},
    )
    suffixes = STAGE_CONTRACTS["money"].input_suffixes(cfg)
    assert ".spellfix_text.parquet" in suffixes
    assert ".normalised_text.parquet" not in suffixes


def test_pii_declares_clean_text_only_when_configured():
    pii = STAGE_CONTRACTS["pii"]
    assert ".clean_text.parquet" in pii.outputs(_config())
    assert ".clean_text.parquet" not in pii.outputs(_config(pii={"write_clean_text": False}))


def test_enrich_declares_the_doc_sidecar_when_persisting_or_ai_chunking():
    enrich = STAGE_CONTRACTS["enrich"]
    assert ".enrichment_doc.parquet" not in enrich.outputs(_config())
    assert ".enrichment_doc.parquet" in enrich.outputs(
        _config(enrichment={"persist_document": True}))
    # cli/link.py persists whenever the same config also enables AI chunking.
    assert ".enrichment_doc.parquet" in enrich.outputs(
        _config(chunking={"chunking_model": "kanon-2-chunker"}))


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_downstream_sidecars_cannot_drive_discovery():
    keys = [
        "runs/r/documents/batch-0001.chunks.parquet",
        "runs/r/documents/batch-0001.embeddings.parquet",
    ]
    assert remote_bases(keys) == []
    assert remote_bases([*keys, "runs/r/documents/batch-0001._manifest.parquet"]) == ["batch-0001"]


def test_a_bare_suffix_is_not_a_base():
    assert remote_bases(["p/.elements.parquet", "p/._manifest.parquet"]) == []


def test_finding_nothing_to_do_exits_non_zero(tmp_path):
    """A typo'd --run-id must not read as success in `run-stage … && next-step`."""
    store_root = tmp_path / "store"
    RemoteStore.from_uri(str(store_root))
    rc = cmd_run_stage(argparse.Namespace(
        stage="normalise", store=str(store_root), shards=None, run_id="no-such-run",
        output_prefix=None, config=None, dsn=None, force=False,
        stage_checkpoints=False, dataset="runner",
    ))
    assert rc == 1


def test_discovery_uses_every_extraction_role_and_skips_corrupt():
    keys = [
        "p/batch-0001.elements.parquet",
        "p/batch-0002.table_cells.parquet",
        "p/batch-0003.form_fields.parquet",
        "p/batch-0004._manifest.parquet",
        "p/batch-0005.corrupt.elements.parquet",
    ]
    assert remote_bases(keys) == ["batch-0001", "batch-0002", "batch-0003", "batch-0004"]


# ---------------------------------------------------------------------------
# Runner mechanics, via synthetic contracts
# ---------------------------------------------------------------------------


def _fake_contract(
    *,
    written: tuple[str, ...],
    declared: tuple[str, ...] | None = None,
    required: tuple[str, ...] = (MANIFEST,),
    conditional: tuple[ConditionalInput, ...] = (),
    scope: StageScope = StageScope.PER_BATCH,
    mutation: MutationMode = MutationMode.SIDECAR,
    seen: list[list[str]] | None = None,
) -> StageContract:
    """A contract that writes *written* for every base it finds staged locally."""
    declared = declared if declared is not None else written

    def run(shard_dir: Path, _config, _ctx) -> None:
        stems = sorted({
            p.name[: -len(MANIFEST)] if p.name.endswith(MANIFEST) else p.name.split(".")[0]
            for p in shard_dir.glob("*.parquet")
        })
        if seen is not None:
            seen.append(stems)
        for stem in stems:
            for suffix in written:
                (shard_dir / f"{stem}{suffix}").write_bytes(b"out")

    return StageContract(
        name="fake",
        scope=scope,
        mutation=mutation,
        required_inputs=required,
        conditional_inputs=lambda _c: conditional,
        outputs=lambda _c: declared,
        run=run,
    )


def test_publishes_declared_outputs_and_is_idempotent(tmp_path):
    store = RemoteStore.from_uri(str(tmp_path / "store"))
    prefix = "runs/r/documents"
    _seed(store, tmp_path, prefix, [f"batch-000{i}{MANIFEST}" for i in (1, 2)])
    contract = _fake_contract(written=(".chunks.parquet",))

    first = run_stage_remote(contract, store, prefix, _config())
    assert (first.processed, first.skipped, first.failed) == (2, 0, 0)
    assert first.published == 2
    assert store.exists(f"{prefix}/batch-0001.chunks.parquet")

    # Re-running is safe and cheap: every base skips on its published output.
    second = run_stage_remote(contract, store, prefix, _config())
    assert (second.processed, second.skipped, second.published) == (0, 2, 0)
    assert second.exit_code == 0

    # A newly landed batch is picked up without touching the finished ones.
    _seed(store, tmp_path, prefix, [f"batch-0003{MANIFEST}"])
    third = run_stage_remote(contract, store, prefix, _config())
    assert (third.processed, third.skipped) == (1, 2)

    forced = run_stage_remote(contract, store, prefix, _config(), force=True)
    assert (forced.processed, forced.skipped) == (3, 0)


def test_partial_output_set_publishes_nothing(tmp_path):
    """A half-written base must never read as complete on the next run."""
    store = RemoteStore.from_uri(str(tmp_path / "store"))
    prefix = "runs/r/documents"
    _seed(store, tmp_path, prefix, [f"batch-0001{MANIFEST}"])
    contract = _fake_contract(
        written=(".spellfix_text.parquet",),
        declared=(".spellfix_text.parquet", ".spellfix_corrections.parquet"),
    )

    summary = run_stage_remote(contract, store, prefix, _config())
    assert (summary.failed, summary.published) == (1, 0)
    assert summary.exit_code == 1
    assert not store.exists(f"{prefix}/batch-0001.spellfix_text.parquet")


def test_all_bases_not_ready_is_a_stage_ordering_error(tmp_path):
    store = RemoteStore.from_uri(str(tmp_path / "store"))
    prefix = "runs/r/documents"
    _seed(store, tmp_path, prefix, [f"batch-000{i}{MANIFEST}" for i in (1, 2)])
    contract = _fake_contract(
        written=(".embeddings.parquet",), required=(".chunks.parquet", MANIFEST),
    )

    summary = run_stage_remote(contract, store, prefix, _config())
    assert (summary.not_ready, summary.processed) == (2, 0)
    assert summary.exit_code == 1


def test_some_bases_not_ready_is_a_draining_fleet(tmp_path):
    store = RemoteStore.from_uri(str(tmp_path / "store"))
    prefix = "runs/r/documents"
    _seed(store, tmp_path, prefix, [
        f"batch-0001{MANIFEST}", "batch-0001.chunks.parquet", f"batch-0002{MANIFEST}",
    ])
    contract = _fake_contract(
        written=(".embeddings.parquet",), required=(".chunks.parquet", MANIFEST),
    )

    summary = run_stage_remote(contract, store, prefix, _config())
    assert (summary.processed, summary.not_ready) == (1, 1)
    assert summary.exit_code == 0


def test_missing_strict_conditional_input_fails_the_base(tmp_path):
    """A selected overlay that is absent would silently fall back to verbatim."""
    store = RemoteStore.from_uri(str(tmp_path / "store"))
    prefix = "runs/r/documents"
    _seed(store, tmp_path, prefix, [f"batch-0001{MANIFEST}"])
    contract = _fake_contract(
        written=(".chunks.parquet",),
        conditional=(ConditionalInput(
            ".normalised_text.parquet", strict=True, reason="text_source='normalised'"),),
    )

    summary = run_stage_remote(contract, store, prefix, _config())
    assert summary.failed == 1
    assert not store.exists(f"{prefix}/batch-0001.chunks.parquet")


def test_missing_soft_conditional_input_is_tolerated(tmp_path):
    store = RemoteStore.from_uri(str(tmp_path / "store"))
    prefix = "runs/r/documents"
    _seed(store, tmp_path, prefix, [f"batch-0001{MANIFEST}"])
    contract = _fake_contract(
        written=(".chunks.parquet",),
        conditional=(ConditionalInput(
            ".enrichment_doc.parquet", strict=False, reason="chunking_model set"),),
    )

    summary = run_stage_remote(contract, store, prefix, _config())
    assert (summary.processed, summary.failed) == (1, 0)


def test_in_place_mutator_is_never_skipped(tmp_path):
    """graph-refresh's outputs are a subset of its inputs — output-exists can't fire."""
    store = RemoteStore.from_uri(str(tmp_path / "store"))
    prefix = "runs/r/documents"
    _seed(store, tmp_path, prefix, [
        f"batch-0001{MANIFEST}", "batch-0001.graph_edges.parquet",
    ])
    contract = _fake_contract(
        written=(".graph_edges.parquet",),
        required=(".graph_edges.parquet", MANIFEST),
        mutation=MutationMode.IN_PLACE,
    )

    for _ in range(2):
        summary = run_stage_remote(contract, store, prefix, _config())
        assert (summary.processed, summary.skipped) == (1, 0)

    dl = store.download_file(f"{prefix}/batch-0001.graph_edges.parquet", tmp_path / "ge.parquet")
    assert dl.read_bytes() == b"out"  # idempotent overwrite


def test_per_batch_stages_stage_one_base_at_a_time(tmp_path):
    store = RemoteStore.from_uri(str(tmp_path / "store"))
    prefix = "runs/r/documents"
    _seed(store, tmp_path, prefix, [f"batch-000{i}{MANIFEST}" for i in (1, 2, 3)])
    seen: list[list[str]] = []
    contract = _fake_contract(written=(".chunks.parquet",), seen=seen)

    run_stage_remote(contract, store, prefix, _config())
    assert seen == [["batch-0001"], ["batch-0002"], ["batch-0003"]]


def test_whole_run_stage_stages_every_base_in_one_pass(tmp_path):
    """quality's dedup clusters are corpus-wide; per-batch would collide ids."""
    store = RemoteStore.from_uri(str(tmp_path / "store"))
    prefix = "runs/r/documents"
    _seed(store, tmp_path, prefix, [
        f"batch-000{i}{MANIFEST}" for i in (1, 2, 3)
    ] + [f"batch-000{i}.chunks.parquet" for i in (1, 2, 3)])
    seen: list[list[str]] = []
    contract = _fake_contract(
        written=(".chunk_quality.parquet",),
        required=(".chunks.parquet",),
        scope=StageScope.WHOLE_RUN,
        seen=seen,
    )

    summary = run_stage_remote(contract, store, prefix, _config())
    assert seen == [["batch-0001", "batch-0002", "batch-0003"]]
    assert summary.published == 3
    assert summary.bases == 1  # one unit of work, not three


# ---------------------------------------------------------------------------
# Real stage, end to end: normalise (offline, no API)
# ---------------------------------------------------------------------------


@pytest.fixture
def extraction_run(tmp_path):
    """Two real extraction shards in a store, from CSV sources (no OCR needed)."""
    from womblex.batch import process_batch
    from womblex.config import (
        ChunkingConfig,
        DatasetConfig,
        ExtractionConfig,
        PathsConfig,
        RedactionConfig,
    )

    cfg = WomblexConfig(
        dataset=DatasetConfig(name="rs"),
        paths=PathsConfig(
            input_root=tmp_path, output_root=tmp_path / "out",
            checkpoint_dir=tmp_path / ".ckpt",
        ),
        extraction=ExtractionConfig(),
        chunking=ChunkingConfig(enabled=False),
        redaction=RedactionConfig(enabled=False),
    )

    local = tmp_path / "local_shards"
    local.mkdir()
    for batch_num, (name, body) in enumerate(
        [
            ("people.csv", "name,role\nAlice Smith,Director\nBob Jones,Analyst\n"),
            ("places.csv", "town,state\nWagga Wagga,NSW\nBallarat,VIC\n"),
        ],
        start=1,
    ):
        src = tmp_path / name
        src.write_text(body)
        process_batch([src], cfg, batch_num=batch_num, shard_dir=local)

    store_root = tmp_path / "store"
    store = RemoteStore.from_uri(str(store_root))
    prefix = "runs/rs/documents"
    for p in sorted(local.glob("batch-*.parquet")):
        store.upload_file(p, f"{prefix}/{p.name}")
    return store, store_root, prefix, local


def test_run_stage_normalise_end_to_end(extraction_run, tmp_path):
    store, store_root, prefix, _local = extraction_run

    rc = cmd_run_stage(argparse.Namespace(
        stage="normalise", store=str(store_root), shards=None, run_id="rs",
        output_prefix=None, config=None, dsn=None, force=False,
        stage_checkpoints=False, dataset="runner",
    ))
    assert rc == 0
    for batch in ("batch-0001", "batch-0002"):
        assert store.exists(f"{prefix}/{batch}.normalised_text.parquet")


def test_local_and_remote_paths_produce_identical_bytes(extraction_run, tmp_path):
    """The parity oracle: one call over the directory == one call per base."""
    store, store_root, prefix, local = extraction_run

    rc_local = cmd_run_stage(argparse.Namespace(
        stage="normalise", store=None, shards=local, run_id=None,
        output_prefix=None, config=None, dsn=None, force=False,
        stage_checkpoints=False, dataset="runner",
    ))
    rc_remote = cmd_run_stage(argparse.Namespace(
        stage="normalise", store=str(store_root), shards=None, run_id="rs",
        output_prefix=None, config=None, dsn=None, force=False,
        stage_checkpoints=False, dataset="runner",
    ))
    assert (rc_local, rc_remote) == (0, 0)

    for batch in ("batch-0001", "batch-0002"):
        name = f"{batch}.normalised_text.parquet"
        remote_copy = store.download_file(f"{prefix}/{name}", tmp_path / "dl" / name)
        assert remote_copy.read_bytes() == (local / name).read_bytes()


def test_checkpoint_directory_is_staged_in_and_out(extraction_run, tmp_path):
    store, store_root, _prefix, _local = extraction_run

    rc = cmd_run_stage(argparse.Namespace(
        stage="normalise", store=str(store_root), shards=None, run_id="rs",
        output_prefix=None, config=None, dsn=None, force=False,
        stage_checkpoints=True, dataset="runner",
    ))
    assert rc == 0
    assert store.list_files("runs/rs/.normalise-checkpoint", "*.json") == [
        "runs/rs/.normalise-checkpoint/runner_normalise_checkpoint.json"
    ]


# ---------------------------------------------------------------------------
# Runtime dependencies fail explicitly
# ---------------------------------------------------------------------------


def test_isaacus_stage_refuses_to_run_without_the_api(monkeypatch, extraction_run):
    """`chunk_shards` would warn, write nothing and return cleanly — a silent no-op."""
    store, store_root, prefix, _local = extraction_run
    monkeypatch.setattr("womblex.utils.availability.isaacus_available", lambda: False)

    rc = cmd_run_stage(argparse.Namespace(
        stage="chunk", store=str(store_root), shards=None, run_id="rs",
        output_prefix=None, config=None, dsn=None, force=False,
        stage_checkpoints=False, dataset="runner",
    ))
    assert rc == 1
    assert not store.exists(f"{prefix}/batch-0001.chunks.parquet")


def test_link_preflight_rejects_a_missing_reference_register(extraction_run, tmp_path):
    """The register is a worker-local file, not a store object."""
    _store, store_root, _prefix, _local = extraction_run
    cfg_path = tmp_path / "link.yaml"
    cfg_path.write_text(
        "dataset:\n  name: t\n"
        "paths:\n  input_root: .\n  output_root: .\n  checkpoint_dir: .\n"
        "linking:\n"
        "  enabled: true\n"
        "  reference:\n"
        f"    path: {tmp_path / 'nope.csv'}\n"
        "    id_col: id\n"
        "    name_col: name\n"
    )

    rc = cmd_run_stage(argparse.Namespace(
        stage="link", store=str(store_root), shards=None, run_id="rs",
        output_prefix=None, config=cfg_path, dsn=None, force=False,
        stage_checkpoints=False, dataset="runner",
    ))
    assert rc == 1


def test_run_id_is_required_with_store(extraction_run):
    _store, store_root, _prefix, _local = extraction_run
    rc = cmd_run_stage(argparse.Namespace(
        stage="normalise", store=str(store_root), shards=None, run_id=None,
        output_prefix=None, config=None, dsn=None, force=False,
        stage_checkpoints=False, dataset="runner",
    ))
    assert rc == 1


def test_run_context_carries_the_client_to_the_stage():
    """needs_client stages receive the constructed client, not a bare None."""
    ctx = RunContext(client=object())
    assert ctx.client is not None
    assert {n for n, c in STAGE_CONTRACTS.items() if c.needs_client} == {"enrich", "embed"}
    assert {n for n, c in STAGE_CONTRACTS.items() if c.needs_isaacus_api} == {
        "chunk", "enrich", "embed"}
