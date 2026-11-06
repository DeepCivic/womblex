"""The committed interface documents (`docs/interfaces/`) describe the API the code serves.

A route, model or scope change that is not regenerated fails here:
``uv run python -m womblex.api.interface_docs`` rewrites the files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from womblex.api import interface_docs

DOCS = Path(__file__).resolve().parent.parent / "docs" / "interfaces"


@pytest.fixture(scope="module")
def built() -> dict[str, str]:
    return {name: interface_docs.render(build()) for name, build in interface_docs.DOCUMENTS.items()}


@pytest.mark.parametrize("name", sorted(interface_docs.DOCUMENTS))
def test_committed_document_is_current(built: dict[str, str], name: str) -> None:
    committed = (DOCS / name).read_text(encoding="utf-8")
    assert committed == built[name], (
        f"docs/interfaces/{name} is stale; run `uv run python -m womblex.api.interface_docs`"
    )


def _operations(paths: dict) -> dict[str, dict]:
    return {f"{m.upper()} {p}": op for p, ms in paths.items() for m, op in ms.items()}


def test_openapi_is_3_2_and_every_operation_names_its_access(built: dict[str, str]) -> None:
    spec = json.loads(built["openapi.json"])
    assert spec["openapi"] == "3.2.0"
    ops = _operations(spec["paths"])
    assert ops["GET /v1/health"]["security"] == []
    assert ops["POST /v1/hooks/{workflow}/{hook}"]["security"] == [{"hookSignature": []}]
    assert ops["GET /v1/runs/{run_id}/documents/{source_hash}/detail"]["security"] == [{"bearerAuth": ["read_raw"]}]
    assert ops["POST /v1/presets"]["security"] == [{"bearerAuth": ["admin"]}]
    for name, op in ops.items():
        for requirement in op["security"]:
            if "bearerAuth" in requirement:
                assert requirement["bearerAuth"], f"{name}: no scope found"


def test_swagger_is_2_0_with_no_openapi_3_constructs(built: dict[str, str]) -> None:
    text = built["swagger.json"]
    spec = json.loads(text)
    assert spec["swagger"] == "2.0"
    for construct in ('"#/components/', '"anyOf"', '"requestBody"', '"const"', '{"type": "null"}'):
        assert construct not in text
    upload = spec["paths"]["/v1/uploads"]["post"]
    assert upload["consumes"] == ["multipart/form-data"]
    assert upload["parameters"][0]["type"] == "file"
    assert _operations(spec["paths"]).keys() == _operations(json.loads(built["openapi.json"])["paths"]).keys()


def test_the_running_service_is_unchanged() -> None:
    """The documents are derived; `/openapi.json` stays FastAPI's own 3.1 output."""
    spec, _ = interface_docs._app_spec()
    assert spec["openapi"].startswith("3.1")
    assert "securitySchemes" not in spec.get("components", {})


def test_mcp_tools_are_provisional_and_cover_every_mapped_operation(built: dict[str, str]) -> None:
    from womblex.api.interface_mcp import MCP_PROTOCOL_VERSION, NOT_MAPPED

    doc = json.loads(built["mcp-tools.json"])
    assert doc["_meta"] == {"womblex/status": "provisional", "womblex/mcpProtocolVersion": MCP_PROTOCOL_VERSION}
    ops = _operations(json.loads(built["openapi.json"])["paths"])
    called = {f"{t['_meta']['womblex/http']['method']} {t['_meta']['womblex/http']['path']}" for t in doc["tools"]}
    skipped = {f"{m.upper()} {p}" for m, p in NOT_MAPPED}
    assert called | skipped == set(ops) and not called & skipped
    for tool in doc["tools"]:
        assert tool["inputSchema"]["type"] == "object"
        assert "#/components/" not in json.dumps(tool)
        assert tool["annotations"]["readOnlyHint"] == (tool["_meta"]["womblex/http"]["method"] == "GET")


def test_agent_card_is_provisional_and_ingests_every_supported_type(built: dict[str, str]) -> None:
    from womblex.api.interface_a2a import A2A_PROTOCOL_VERSION, MEDIA_TYPES, PLACEHOLDER_URL
    from womblex.cli._shared import SUPPORTED_EXTENSIONS

    card = json.loads(built["agent-card.json"])
    assert card["supportedInterfaces"] == [
        {"url": PLACEHOLDER_URL, "protocolBinding": "HTTP+JSON", "protocolVersion": A2A_PROTOCOL_VERSION},
    ]
    assert "Provisional" in card["description"]
    assert set(MEDIA_TYPES) == set(SUPPORTED_EXTENSIONS)
    for required in ("name", "description", "version", "capabilities", "defaultInputModes",
                     "defaultOutputModes", "skills"):
        assert card[required]
    for skill in card["skills"]:
        assert {"id", "name", "description", "tags"} <= set(skill)
