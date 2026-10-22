"""The pipeline graph and a composed config's schema, validation and YAML (library and `/v1`)."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from womblex import config_tools
from womblex.cloud.deployment import DeploymentSettings
from womblex.cloud.stage_contracts import STAGE_NAMES
from womblex.cloud.stage_graph import stage_graph
from womblex.config import WomblexConfig

_MINIMAL = {"dataset": {"name": "t"}}


class TestStageGraph:
    def test_nodes_are_extract_and_every_stage(self) -> None:
        assert [n["id"] for n in stage_graph()["nodes"]] == ["extract", *STAGE_NAMES]

    def test_one_edge_per_pair_carrying_its_suffixes(self) -> None:
        edges = stage_graph()["edges"]
        pairs = [(e["from"], e["to"]) for e in edges]
        assert len(pairs) == len(set(pairs))
        by_pair = {(e["from"], e["to"]): e["suffixes"] for e in edges}
        assert by_pair[("extract", "chunk")] == [".elements.parquet", ".table_cells.parquet", "._manifest.parquet"]
        assert by_pair[("chunk", "embed")] == [".chunks.parquet"]

    def test_acyclic_and_every_stage_descends_from_extract(self) -> None:
        adjacency: dict[str, set[str]] = {}
        for e in stage_graph()["edges"]:
            adjacency.setdefault(e["from"], set()).add(e["to"])
        reached: set[str] = set()

        def walk(node: str, seen: frozenset[str]) -> None:
            for nxt in adjacency.get(node, ()):
                assert nxt not in seen, f"cycle through {nxt}"
                reached.add(nxt)
                walk(nxt, seen | {nxt})

        walk("extract", frozenset({"extract"}))
        assert reached == set(STAGE_NAMES)

    def test_config_sections_are_real_config_fields(self) -> None:
        nodes = stage_graph()["nodes"]
        assert {n["config_section"] for n in nodes} - {None} <= set(WomblexConfig.model_fields)
        assert {n["id"] for n in nodes if n["config_section"] is None} == {"extract", "graph-refresh", "layout"}


@pytest.fixture
def local(tmp_path: Path) -> DeploymentSettings:
    return DeploymentSettings(output_root=tmp_path, store_uri=None)


class TestConfigTools:
    def test_schema_leaves_paths_to_the_deployment(self) -> None:
        schema = config_tools.config_schema()
        assert {"dataset", "chunking"} <= set(schema["properties"]) and "paths" not in schema["properties"]
        assert "paths" not in schema["required"] and "dataset" in schema["required"]

    def test_a_minimal_config_is_valid(self, local: DeploymentSettings) -> None:
        assert config_tools.validate_config(_MINIMAL, local) == {"valid": True, "errors": []}

    def test_typos_are_errors_naming_the_key(self, local: DeploymentSettings) -> None:
        result = config_tools.validate_config(
            {**_MINIMAL, "chunkng": {}, "chunking": {"chnk_size": 9}}, local)
        assert sorted(tuple(e["loc"]) for e in result["errors"]) == [("chunking", "chnk_size"), ("chunkng",)]

    def test_free_form_dict_values_are_not_walked_into(self, local: DeploymentSettings) -> None:
        raw = {**_MINIMAL, "normalise": {"substitutions": {"Depatment": "Department"}}}
        assert config_tools.validate_config(raw, local)["valid"] is True

    def test_posted_paths_are_replaced_by_the_deployments(self, local: DeploymentSettings, tmp_path: Path) -> None:
        raw = {**_MINIMAL, "paths": {"input_root": "/x", "output_root": "/root/.ssh", "checkpoint_dir": "/etc"}}
        parsed = yaml.safe_load(config_tools.render_yaml(raw, local))
        assert parsed["paths"]["output_root"] == str(tmp_path)

    def test_yaml_loads_back_with_defaults_applied(self, local: DeploymentSettings) -> None:
        text = config_tools.render_yaml(_MINIMAL, local)
        assert not text.startswith("#")
        parsed = yaml.safe_load(text)
        assert "chunking" in parsed and WomblexConfig(**parsed).dataset.name == "t"

    def test_an_object_store_is_a_placeholder_named_in_a_header(self) -> None:
        settings = DeploymentSettings(output_root=None, store_uri="s3://bucket", ingest_uri="s3://inbox")
        text = config_tools.render_yaml(_MINIMAL, settings)
        assert text.startswith("#") and "$WOMBLEX_STORE_URI" in text and "$WOMBLEX_INGEST_URI" in text
        assert yaml.safe_load(text)["paths"]["output_root"] == "."


class TestRoutes:
    @pytest.fixture
    def api(self, tmp_path: Path):  # type: ignore[no-untyped-def]
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from womblex.api.app import create_api_app
        from womblex.api.auth import hash_token, parse_registry

        registry = parse_registry({"clients": [
            {"client_id": "a", "token_sha256": hash_token("t"), "scopes": ["read"]},
            {"client_id": "s", "token_sha256": hash_token("s"), "scopes": ["submit"]},
        ]})
        return TestClient(create_api_app(store_uri=str(tmp_path / "store"), db_dsn="postgresql://x/y",
                                         registry=registry))

    def test_each_route_needs_the_read_scope(self, api) -> None:  # type: ignore[no-untyped-def]
        for method, path in (("get", "/v1/stages"), ("get", "/v1/config/schema"),
                             ("post", "/v1/config/validate"), ("post", "/v1/config/yaml")):
            assert getattr(api, method)(path).status_code == 401
            assert getattr(api, method)(path, headers={"Authorization": "Bearer s"}).status_code == 403

    def test_the_routes_serve_the_library(self, api) -> None:  # type: ignore[no-untyped-def]
        h = {"Authorization": "Bearer t"}
        assert api.get("/v1/stages", headers=h).json() == stage_graph()
        assert api.get("/v1/config/schema", headers=h).json() == config_tools.config_schema()
        assert api.post("/v1/config/validate", headers=h, json=_MINIMAL).json() == {"valid": True, "errors": []}
        ok = api.post("/v1/config/yaml", headers=h, json=_MINIMAL)
        assert ok.headers["content-disposition"] == "attachment; filename=womblex.yaml"
        assert yaml.safe_load(ok.text)["dataset"]["name"] == "t"
        bad = api.post("/v1/config/yaml", headers=h, json={"chunkng": {}})
        assert bad.status_code == 422
        assert {tuple(e["loc"]) for e in bad.json()["detail"]} == {("dataset",), ("chunkng",)}
