"""Console Pipeline Composer: stage graph, config schema, validation and presets (womblex.ui)."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from tests._ui_helpers import REPO_ROOT
from womblex.ui.app import create_app

_MINIMAL_CONFIG = {"dataset": {"name": "t"}}
_NO_DATASET: dict = {}


class TestComposerApi:
    """The Pipeline Composer reads no run — `config.py` and
    `cloud/stage_contracts.py` are static (docs/functional_requirements.md, Requirement 19), so
    `api_client`'s local/remote parametrisation is irrelevant here; a bare
    `TestClient` is enough.
    """

    @pytest.fixture
    def client(self, tmp_path: Path) -> TestClient:
        return TestClient(create_app(output_root=tmp_path))

    def test_graph_nodes_cover_extract_plus_every_stage(self, client: TestClient) -> None:
        from womblex.cloud.stage_contracts import STAGE_NAMES

        body = client.get("/api/composer/graph").json()
        assert [n["id"] for n in body["nodes"]] == ["extract", *STAGE_NAMES]

    def test_graph_edges_are_one_per_pair_and_resolve_their_producers(
        self, client: TestClient
    ) -> None:
        """chunk reading three extraction sidecars is one dependency, not three —
        otherwise every renderer dedupes to avoid parallel arrows."""
        body = client.get("/api/composer/graph").json()
        pairs = [(e["from"], e["to"]) for e in body["edges"]]
        assert len(pairs) == len(set(pairs))
        edges = {(e["from"], e["to"]): e["suffixes"] for e in body["edges"]}
        assert edges[("extract", "chunk")] == [
            ".elements.parquet", ".table_cells.parquet", "._manifest.parquet",
        ]
        # embed reads chunk's own output — a stage-to-stage edge, not extract.
        assert edges[("chunk", "embed")] == [".chunks.parquet"]

    def test_graph_is_acyclic_and_every_stage_descends_from_extract(
        self, client: TestClient
    ) -> None:
        from womblex.cloud.stage_contracts import STAGE_NAMES

        body = client.get("/api/composer/graph").json()
        adjacency: dict[str, set[str]] = {}
        for edge in body["edges"]:
            adjacency.setdefault(edge["from"], set()).add(edge["to"])
        reached: set[str] = set()

        def walk(node: str, seen: frozenset[str]) -> None:
            for nxt in adjacency.get(node, ()):
                assert nxt not in seen, f"cycle through {nxt}"
                reached.add(nxt)
                walk(nxt, seen | {nxt})

        walk("extract", frozenset({"extract"}))
        # A stage unreachable from extract would be one the composer cannot
        # order — the plan's guardrail is precisely this reachability.
        assert reached == set(STAGE_NAMES)

    def test_graph_node_config_sections_are_real_config_fields(
        self, client: TestClient
    ) -> None:
        """The composer's node → config-section map is hand-declared (nothing in
        `StageContract` names a config field), so a renamed section must fail
        here rather than silently leave a stage's toggle wired to nothing."""
        from womblex.config import WomblexConfig

        body = client.get("/api/composer/graph").json()
        sections = {n["config_section"] for n in body["nodes"]} - {None}
        assert sections and sections <= set(WomblexConfig.model_fields)
        assert {n["id"] for n in body["nodes"] if n["config_section"] is None} == {
            "extract",
            "graph-refresh",
            "layout",
        }

    def test_schema_is_the_womblex_config_schema_minus_paths(self, client: TestClient) -> None:
        """`paths` is the deployment's locations, not operator-retyped per run."""
        body = client.get("/api/composer/schema").json()
        assert set(body["properties"]) >= {"dataset", "chunking", "pii"}
        assert "paths" not in body["properties"]
        assert "paths" not in body.get("required", [])
        assert "dataset" in body.get("required", [])

    def test_presets_list_includes_default_isaacus(self, client: TestClient) -> None:
        body = client.get("/api/composer/presets").json()
        names = [p["name"] for p in body["presets"]]
        assert "DEFAULT-Isaacus" in names
        preset = next(p for p in body["presets"] if p["name"] == "DEFAULT-Isaacus")
        assert preset["formats"] == [".pdf", ".docx", ".md", ".markdown"]

    def test_default_isaacus_enables_the_full_reference_shape(
        self, client: TestClient
    ) -> None:
        """The preset is the reference extract → normalise → enrich → chunk →
        build_graph → embed → money → link pipeline: text cleaning (via
        processing.text_source=normalised, which is also what gates normalise),
        AI chunking, enrichment, embedding, money and linking all on; graph
        produced by enrich + the offline graph-refresh edge rebuild. Embed is
        part of the shape (the demo run carries it); a preset that omitted it
        disagreed with the sample corpus, which is the bug this pins.

        spellfix is NOT in the shape: dictionary-gated repair is a per-corpus
        call, so the reference default must not carry it."""
        preset = client.get("/api/composer/presets/DEFAULT-Isaacus").json()
        cfg = preset["config"]
        assert cfg["chunking"]["enabled"] is True
        assert cfg["chunking"]["chunking_model"] == "kanon-2-enricher"
        assert cfg["enrichment"]["enabled"] is True
        assert cfg["embedding"]["enabled"] is True
        assert cfg["money"]["enabled"] is True
        assert cfg["linking"]["enabled"] is True
        assert "spellfix" not in cfg
        # Text cleaning is selected once; this is also what gates normalise into
        # the downstream-stage dispatch and puts it before enrich.
        assert cfg["processing"]["text_source"] == "normalised"
        # No dataset/paths: the operator supplies the run's identity and paths.
        assert "dataset" not in cfg
        assert "paths" not in cfg

    def test_a_preset_overlaid_on_a_minimal_config_validates(
        self, client: TestClient
    ) -> None:
        """A preset is a partial config; merged onto dataset+paths it must be a
        config the CLI would load — that is the whole point of it being data
        the same `WomblexConfig(**raw)` construction checks."""
        preset = client.get("/api/composer/presets/DEFAULT-Isaacus").json()
        merged = {**_MINIMAL_CONFIG, **preset["config"]}
        resp = client.post("/api/composer/validate", json=merged)
        body = resp.json()
        assert body["valid"] is True

    def test_unknown_preset_404s(self, client: TestClient) -> None:
        assert client.get("/api/composer/presets/nope").status_code == 404

    def test_default_isaacus_config_file_ships_and_loads(self, client: TestClient) -> None:
        """The preset is CLI-runnable: `configs/default-isaacus.yaml` exists and
        `load_config` accepts it (a dev runs the per-stage sequence with it)."""
        from womblex.config import load_config

        cfg_path = REPO_ROOT / "configs" / "default-isaacus.yaml"
        assert cfg_path.is_file(), "configs/default-isaacus.yaml is the CLI source of truth"
        cfg = load_config(cfg_path)
        # The stages the shape names are on.
        assert cfg.chunking.enabled and cfg.chunking.chunking_model == "kanon-2-enricher"
        assert cfg.enrichment.enabled
        assert cfg.embedding.enabled
        assert cfg.money.enabled
        assert cfg.linking.enabled
        assert not cfg.spellfix.enabled, "spellfix is a per-corpus opt-in"
        # Text cleaning runs before enrich/chunk: the overlay both reassemble from.
        assert cfg.processing.text_source == "normalised"
        # AI chunking + enrich both on => reuse auto-wired, so no double enrich.
        assert cfg.enrichment.persist_document is True

    def test_config_file_and_ui_preset_agree(self, client: TestClient) -> None:
        """The console preset mirrors the shipped config file; they must not drift
        on the stage toggles/settings the preset carries."""
        from womblex.config import load_config

        cfg = load_config(REPO_ROOT / "configs" / "default-isaacus.yaml")
        overlay = client.get("/api/composer/presets/DEFAULT-Isaacus").json()["config"]
        assert overlay["chunking"]["chunk_size"] == cfg.chunking.chunk_size
        assert overlay["chunking"]["chunking_model"] == cfg.chunking.chunking_model
        assert overlay["chunking"]["overlap"] == cfg.chunking.overlap
        # The stage toggles the preset carries agree with the config file.
        assert overlay["enrichment"]["enabled"] == cfg.enrichment.enabled
        assert overlay["embedding"]["enabled"] == cfg.embedding.enabled
        assert overlay["money"]["enabled"] == cfg.money.enabled
        assert overlay["money"]["default_currency"] == cfg.money.default_currency
        assert overlay["linking"]["enabled"] == cfg.linking.enabled
        # Neither carries spellfix: absent from the overlay, off in the config.
        assert "spellfix" not in overlay and not cfg.spellfix.enabled
        # The text-cleaning selector (which also gates normalise into the
        # downstream dispatch, before enrich) agrees.
        assert overlay["processing"]["text_source"] == cfg.processing.text_source

    def test_validate_accepts_a_minimal_config(self, client: TestClient) -> None:
        resp = client.post("/api/composer/validate", json=_MINIMAL_CONFIG)
        assert resp.json() == {"valid": True, "errors": []}

    def test_validate_refuses_typos(self, client: TestClient) -> None:
        """The config models forbid extras, so a typo is an error naming the key."""
        resp = client.post("/api/composer/validate", json={
            **_MINIMAL_CONFIG,
            "chunkng": {"chunk_size": 99},            # typo'd section
            "chunking": {"chnk_size": 99},            # typo'd field in a real section
        })
        body = resp.json()
        assert body["valid"] is False
        assert sorted(tuple(err["loc"]) for err in body["errors"]) == [
            ("chunking", "chnk_size"), ("chunkng",),
        ]

    def test_validate_does_not_mistake_free_form_dict_values_for_typos(
        self, client: TestClient
    ) -> None:
        """`normalise.substitutions` is an operator's own letterhead map — its keys
        are data, not config fields, and must not be walked into."""
        resp = client.post("/api/composer/validate", json={
            **_MINIMAL_CONFIG,
            "normalise": {"substitutions": {"Depatment": "Department", "AB C": "ABC"}},
        })
        assert resp.json()["valid"] is True

    def test_validate_refuses_a_stray_key_in_an_optional_nested_model(
        self, client: TestClient
    ) -> None:
        resp = client.post("/api/composer/validate", json={
            **_MINIMAL_CONFIG,
            "linking": {"reference": {"path": "r.csv", "nonsense_key": 1}},
        })
        body = resp.json()
        assert body["valid"] is False
        assert any("nonsense_key" in err["loc"] for err in body["errors"])

    def test_validate_reports_pydantic_errors_for_a_missing_field(self, client: TestClient) -> None:
        resp = client.post("/api/composer/validate", json=_NO_DATASET)
        body = resp.json()
        assert body["valid"] is False
        assert any(err["loc"] == ["dataset"] for err in body["errors"])

    def test_deployment_paths_override_any_submitted_paths(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """A posted `paths` is replaced wholesale, never touched otherwise."""
        resp = client.post("/api/composer/validate", json={
            "dataset": {"name": "t"},
            "paths": {"input_root": "/nonexistent/xyz", "output_root": "/root/.ssh",
                      "checkpoint_dir": "/etc/shadow"},
        })
        assert resp.json()["valid"] is True
        yaml_resp = client.post("/api/composer/yaml", json={"dataset": {"name": "t"}})
        parsed = yaml.safe_load(yaml_resp.text)
        assert parsed["paths"]["output_root"] == str(tmp_path)
        assert parsed["paths"]["input_root"] != "/nonexistent/xyz"

    def test_yaml_download_roundtrips_a_valid_config(self, client: TestClient) -> None:
        resp = client.post("/api/composer/yaml", json=_MINIMAL_CONFIG)
        assert resp.status_code == 200
        assert resp.headers["content-disposition"] == "attachment; filename=womblex.yaml"
        parsed = yaml.safe_load(resp.text)
        assert parsed["dataset"]["name"] == "t"
        # Pydantic-applied defaults are in the download, not just what was posted.
        assert "chunking" in parsed
        # The download's whole purpose: `load_config` must accept it back.
        from womblex.config import WomblexConfig

        assert WomblexConfig(**parsed).dataset.name == "t"
        assert not resp.text.startswith("#")

    def test_yaml_download_422s_on_a_stray_key(self, client: TestClient) -> None:
        resp = client.post("/api/composer/yaml", json={
            **_MINIMAL_CONFIG, "chunkng": {"chunk_size": 99},
        })
        assert resp.status_code == 422
        assert any(err["loc"] == ["chunkng"] for err in resp.json()["detail"])

    def test_yaml_download_422s_on_an_invalid_config(self, client: TestClient) -> None:
        resp = client.post("/api/composer/yaml", json=_NO_DATASET)
        assert resp.status_code == 422
        assert any(err["loc"] == ["dataset"] for err in resp.json()["detail"])

    def test_built_in_presets_are_marked_as_such(self, client: TestClient) -> None:
        """`source` lets the screen offer delete only on saved presets; a built-in
        is code and cannot be removed."""
        body = client.get("/api/composer/presets").json()
        builtin = next(p for p in body["presets"] if p["name"] == "DEFAULT-Isaacus")
        assert builtin["source"] == "builtin"


class TestComposerSavePresets:
    """Saving a composed config as a named preset (docs/functional_requirements.md, Requirement 19).

    Parametrised over local (a writable `presets_dir`) and remote (the store's
    own `presets/` prefix, a sibling of `runs/`). A store-backed console therefore
    saves without any writable mount; a local console without a presets dir
    refuses with 409, the same shape the Execution Controls use.
    """

    def _saving_client(self, tmp_path: Path, mode: str) -> TestClient:
        """A client that *can* save, in the given mode."""
        if mode == "remote":
            pytest.importorskip("fsspec")
            return TestClient(create_app(store_uri=str(tmp_path / "store")))
        return TestClient(create_app(output_root=tmp_path, presets_dir=tmp_path / "presets"))

    def _client(self, tmp_path: Path, *, with_presets_dir: bool = True) -> TestClient:
        presets_dir = (tmp_path / "presets") if with_presets_dir else None
        return TestClient(create_app(output_root=tmp_path, presets_dir=presets_dir))

    @pytest.mark.parametrize("mode", ["local", "remote"])
    def test_save_then_list_and_fetch_round_trips(self, tmp_path: Path, mode: str) -> None:
        client = self._saving_client(tmp_path, mode)
        resp = client.post("/api/composer/presets", json={
            "name": "My-Run", "description": "chunk only", "formats": [".pdf"],
            "config": {"chunking": {"enabled": True, "chunk_size": 320}},
        })
        assert resp.status_code == 201
        assert resp.json()["source"] == "saved"

        names = {p["name"]: p for p in client.get("/api/composer/presets").json()["presets"]}
        assert "My-Run" in names and names["My-Run"]["source"] == "saved"
        # Built-ins still serve alongside the saved one.
        assert "DEFAULT-Isaacus" in names

        one = client.get("/api/composer/presets/My-Run").json()
        assert one["config"]["chunking"]["chunk_size"] == 320

    def test_remote_save_lands_under_the_store_presets_prefix(self, tmp_path: Path) -> None:
        """A store-backed console writes presets to `presets/`, a sibling of
        `runs/` — so the compose ui service needs no writable mount."""
        pytest.importorskip("fsspec")
        store_root = tmp_path / "store"
        client = TestClient(create_app(store_uri=str(store_root)))
        resp = client.post("/api/composer/presets", json={
            "name": "Remote-One", "config": {"chunking": {"chunk_size": 128}},
        })
        assert resp.status_code == 201
        # One object, under the store's presets/ prefix, and nowhere near runs/.
        saved = list((store_root / "presets").glob("*.preset.json"))
        assert [p.name for p in saved] == ["Remote-One.preset.json"]
        assert not (store_root / "runs").exists()

    def test_save_strips_dataset_and_paths(self, tmp_path: Path) -> None:
        """A preset is an overlay — the run's identity is never baked into it."""
        client = self._client(tmp_path)
        resp = client.post("/api/composer/presets", json={
            "name": "Overlay",
            "config": {**_MINIMAL_CONFIG, "money": {"enabled": True}},
        })
        cfg = resp.json()["config"]
        assert "dataset" not in cfg and "paths" not in cfg
        assert cfg["money"]["enabled"] is True

    def test_saved_preset_overlaid_on_a_minimal_config_still_validates(
        self, tmp_path: Path
    ) -> None:
        """The whole point of validating at save time: what is saved must load."""
        client = self._client(tmp_path)
        client.post("/api/composer/presets", json={
            "name": "Round", "config": {"chunking": {"chunk_size": 200}},
        })
        overlay = client.get("/api/composer/presets/Round").json()["config"]
        resp = client.post("/api/composer/validate", json={**_MINIMAL_CONFIG, **overlay})
        assert resp.json()["valid"] is True

    def test_save_rejects_a_config_that_would_not_load(self, tmp_path: Path) -> None:
        """A preset that fails to build a `WomblexConfig` is refused at save (400),
        not left to 500 whoever later picks it."""
        client = self._client(tmp_path)
        resp = client.post("/api/composer/presets", json={
            "name": "Bad", "config": {"chunking": {"chunk_size": "not-an-int"}},
        })
        assert resp.status_code == 400

    def test_save_rejects_an_unsafe_name(self, tmp_path: Path) -> None:
        client = self._client(tmp_path)
        for name in ["../escape", "a/b", "..", ".hidden"]:
            resp = client.post("/api/composer/presets", json={"name": name, "config": {}})
            assert resp.status_code == 400, name
        # Nothing was written anywhere — no file escaped the presets dir.
        assert list(tmp_path.rglob("*.preset.json")) == []

    def test_save_conflicts_without_a_presets_dir(self, tmp_path: Path) -> None:
        client = self._client(tmp_path, with_presets_dir=False)
        resp = client.post("/api/composer/presets", json={"name": "X", "config": {}})
        assert resp.status_code == 409

    @pytest.mark.parametrize("mode", ["local", "remote"])
    def test_delete_removes_a_saved_preset(self, tmp_path: Path, mode: str) -> None:
        client = self._saving_client(tmp_path, mode)
        client.post("/api/composer/presets", json={"name": "Temp", "config": {}})
        assert client.delete("/api/composer/presets/Temp").status_code == 200
        assert client.get("/api/composer/presets/Temp").status_code == 404

    def test_delete_refuses_to_remove_a_built_in(self, tmp_path: Path) -> None:
        """A built-in is code; deleting it is a 404 here, not a silent success."""
        client = self._client(tmp_path)
        assert client.delete("/api/composer/presets/DEFAULT-Isaacus").status_code == 404
        # And it is still there.
        assert client.get("/api/composer/presets/DEFAULT-Isaacus").status_code == 200

    def test_delete_conflicts_without_a_presets_dir(self, tmp_path: Path) -> None:
        client = self._client(tmp_path, with_presets_dir=False)
        assert client.delete("/api/composer/presets/anything").status_code == 409

    def test_a_saved_preset_shadows_a_built_in_of_the_same_name(self, tmp_path: Path) -> None:
        """The operator asked for that name; their save wins, still marked saved."""
        client = self._client(tmp_path)
        client.post("/api/composer/presets", json={
            "name": "DEFAULT-Isaacus", "config": {"chunking": {"chunk_size": 999}},
        })
        one = client.get("/api/composer/presets/DEFAULT-Isaacus").json()
        assert one["source"] == "saved"
        assert one["config"]["chunking"]["chunk_size"] == 999
