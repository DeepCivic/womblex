"""Provisional: the `/v1` API as the tools an MCP server would list (`docs/interfaces/mcp-tools.json`).

Womblex serves no MCP endpoint. This maps each operation of the OpenAPI document
(`api/interface_docs.openapi_document`) onto MCP 2026-07-28's `Tool` object, shaped as a
`tools/list` result, so the mapping can be reviewed before anything serves it. Every value is
derived mechanically from the operation; nothing is chosen per tool.
"""

from __future__ import annotations

import json
import re
from typing import Any

MCP_PROTOCOL_VERSION = "2026-07-28"
_DIALECT = "https://json-schema.org/draft/2020-12/schema"

#: Operations not mapped, and why (repeated in docs/interfaces.md).
NOT_MAPPED = {
    ("get", "/v1/health"): "a transport probe, not a capability",
    ("get", "/v1/ready"): "a transport probe, not a capability",
    ("post", "/v1/uploads"): "a multipart file body; MCP tool arguments are JSON",
    ("post", "/v1/hooks/{workflow}/{hook}"): "signed by another system's HMAC, not called by a client",
}


def mcp_tools_document() -> dict[str, Any]:
    from womblex.api.interface_docs import openapi_document

    spec = openapi_document()
    schemas = spec["components"].get("schemas", {})
    tools = []
    for path, methods in spec["paths"].items():
        for method, op in methods.items():
            if (method, path) not in NOT_MAPPED:
                tools.append(_tool(method, path, op, schemas))
    names = [t["name"] for t in tools]
    if len(set(names)) != len(names):
        raise ValueError(f"two operations map to one tool name: {sorted(n for n in names if names.count(n) > 1)}")
    return {
        "_meta": {"womblex/status": "provisional", "womblex/mcpProtocolVersion": MCP_PROTOCOL_VERSION},
        "resultType": "complete",
        "ttlMs": 0,
        "cacheScope": "private",  # what a caller sees is scoped to its token (owner-scoped runs)
        "tools": tools,
    }


def _tool(method: str, path: str, op: dict[str, Any], schemas: dict[str, Any]) -> dict[str, Any]:
    scope = next((roles[0] for req in op.get("security", []) for roles in req.values() if roles), None)
    tool: dict[str, Any] = {
        "name": re.sub(r"[^a-z0-9]+", "_", op["summary"].lower()).strip("_"),
        "title": op["summary"],
        "description": (op.get("description") or op["summary"]).strip()
        + f"\n\nCalls `{method.upper()} {path}`" + (f" (scope `{scope}`)." if scope else "."),
        "inputSchema": _input_schema(op, schemas),
    }
    output = _output_schema(op, schemas)
    if output is not None:
        tool["outputSchema"] = output
    read_only = method == "get"
    tool["annotations"] = {"readOnlyHint": read_only}
    if not read_only:
        tool["annotations"] |= {"destructiveHint": method == "delete", "idempotentHint": method in ("put", "delete")}
    tool["_meta"] = {"womblex/http": {"method": method.upper(), "path": path, "scope": scope}}
    return tool


def _input_schema(op: dict[str, Any], schemas: dict[str, Any]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for param in op.get("parameters", []):
        prop = dict(param.get("schema", {}))
        if param.get("description"):
            prop["description"] = param["description"]
        properties[param["name"]] = prop
        if param.get("required"):
            required.append(param["name"])
    body = op.get("requestBody", {}).get("content", {}).get("application/json")
    if body is not None:
        properties["body"] = body.get("schema", {})
        if op["requestBody"].get("required"):
            required.append("body")
    schema: dict[str, Any] = {"$schema": _DIALECT, "type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return _with_defs(schema, schemas)


def _output_schema(op: dict[str, Any], schemas: dict[str, Any]) -> dict[str, Any] | None:
    """The success response's schema, when it is an object (MCP's `structuredContent`)."""
    code = next(c for c in op["responses"] if c.startswith("2"))
    schema = op["responses"][code].get("content", {}).get("application/json", {}).get("schema")
    if not schema or "$ref" not in schema:
        return None
    return _with_defs({"$schema": _DIALECT, **schema}, schemas)


def _with_defs(schema: dict[str, Any], schemas: dict[str, Any]) -> dict[str, Any]:
    """*schema* self-contained: each referenced component copied under `$defs`."""
    text = json.dumps(schema)
    wanted: list[str] = []
    pending = re.findall(r'"#/components/schemas/([^"]+)"', text)
    while pending:
        name = pending.pop()
        if name not in wanted:
            wanted.append(name)
            pending += re.findall(r'"#/components/schemas/([^"]+)"', json.dumps(schemas[name]))
    if wanted:
        schema = {**schema, "$defs": {n: schemas[n] for n in sorted(wanted)}}
    rewritten: dict[str, Any] = json.loads(json.dumps(schema).replace('"#/components/schemas/', '"#/$defs/'))
    return rewritten
