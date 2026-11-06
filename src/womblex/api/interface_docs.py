"""The `/v1` API described as it is, in OpenAPI 3.2.0 and Swagger 2.0 (`docs/interfaces/`).

Both documents are derived from the FastAPI app's own OpenAPI 3.1 output, so they describe
exactly the routes and models the service serves. Two facts the generated output leaves out
are added from the code: the bearer-token scope each route checks (`api/auth.require_scope`)
and the HMAC signature header the inbound hook route checks. Nothing here changes what the
running service answers at `/openapi.json`.

Regenerate with ``python -m womblex.api.interface_docs`` (writes ``docs/interfaces/``);
``tests/test_interface_docs.py`` fails when the committed files differ from what this builds.
"""

from __future__ import annotations

import copy
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

#: Operations that take no bearer token: the probes, and the hook (its own HMAC).
_OPEN = {"/v1/health", "/v1/ready"}
_HOOK = "/v1/hooks/{workflow}/{hook}"
_SIGNATURE_HEADER = "X-Womblex-Signature"

_DESCRIPTION = (
    "The Womblex service API (`womblex serve`), described as it is. Bearer tokens are "
    "service tokens from the client registry (`WOMBLEX_API_CLIENTS`); each operation names "
    "the scope it requires (`read`, `submit`, `read_raw` or `admin`). See docs/service-api.md."
)


def _app_spec() -> tuple[dict[str, Any], dict[tuple[str, str], str]]:
    """The app's generated OpenAPI 3.1 document and each operation's required scope.

    The scope is read from the service's own answer: a token holding no scope is sent to
    each operation, and its 403 names the scope that operation checks (`require_scope`).
    """
    from fastapi.testclient import TestClient

    from womblex.api.app import create_api_app
    from womblex.api.auth import Client, ClientRegistry, hash_token

    token = "interface-docs-probe"  # pragma: allowlist secret
    registry = ClientRegistry((Client("probe", hash_token(token), frozenset()),))
    with tempfile.TemporaryDirectory() as store:
        app = create_api_app(store_uri=store, db_dsn="postgresql://unused/unused", registry=registry)
        spec = app.openapi()
        client = TestClient(app, raise_server_exceptions=False)
        scopes: dict[tuple[str, str], str] = {}
        for path, methods in spec["paths"].items():
            if path in _OPEN or path == _HOOK:
                continue
            url = re.sub(r"\{[^}]+\}", "probe", path)
            for method in methods:
                answer = client.request(method.upper(), url, headers={"Authorization": f"Bearer {token}"})
                found = _SCOPE_RE.search(answer.text) if answer.status_code == 403 else None
                if found:
                    scopes[(path, method)] = found.group(1)
    return copy.deepcopy(spec), scopes


#: The detail `api/auth.require_scope` answers with.
_SCOPE_RE = re.compile(r"requires the '(\w+)' scope")


def openapi_document() -> dict[str, Any]:
    """The API as an OpenAPI 3.2.0 document, with its security declared."""
    spec, scopes = _app_spec()
    spec["openapi"] = "3.2.0"
    spec["info"]["description"] = _DESCRIPTION
    spec.setdefault("components", {})["securitySchemes"] = {
        "bearerAuth": {"type": "http", "scheme": "bearer", "description": "A service token from the client registry."},
        "hookSignature": {
            "type": "apiKey", "in": "header", "name": _SIGNATURE_HEADER,
            "description": "`sha256=<hex>`: HMAC-SHA256 of the raw body under the hook's secret.",
        },
    }
    for path, methods in spec["paths"].items():
        for method, op in methods.items():
            if path in _OPEN:
                op["security"] = []
            elif path == _HOOK:
                op["security"] = [{"hookSignature": []}]
            else:  # OpenAPI 3.1+ lets a non-OAuth requirement list the roles it needs
                scope = scopes.get((path, method))
                op["security"] = [{"bearerAuth": [scope] if scope else []}]
    return spec


def swagger_document() -> dict[str, Any]:
    """The same API as a Swagger 2.0 document."""
    spec = openapi_document()
    out: dict[str, Any] = {"swagger": "2.0", "info": spec["info"], "paths": {}}
    out["securityDefinitions"] = {
        "bearerAuth": {"type": "apiKey", "in": "header", "name": "Authorization",
                       "description": "`Bearer <token>`: a service token from the client registry."},
        "hookSignature": {"type": "apiKey", "in": "header", "name": _SIGNATURE_HEADER,
                          "description": spec["components"]["securitySchemes"]["hookSignature"]["description"]},
    }
    schemas = spec["components"].get("schemas", {})
    out["definitions"] = {name: _schema(s) for name, s in schemas.items()}
    for path, methods in spec["paths"].items():
        out["paths"][path] = {m: _operation(op, schemas) for m, op in methods.items()}
    return out


def _operation(op: dict[str, Any], schemas: dict[str, Any]) -> dict[str, Any]:
    new: dict[str, Any] = {k: op[k] for k in ("tags", "summary", "description", "operationId") if k in op}
    params = [_parameter(p) for p in op.get("parameters", [])]
    body = op.get("requestBody")
    if body:
        content = body["content"]
        if "multipart/form-data" in content:
            new["consumes"] = ["multipart/form-data"]
            params += _form_parameters(content["multipart/form-data"]["schema"], schemas)
        else:
            media = next(iter(content))
            new["consumes"] = [media]
            params.append({"in": "body", "name": "body", "required": body.get("required", False),
                           "schema": _schema(content[media].get("schema", {}))})
    if params:
        new["parameters"] = params
    produces: list[str] = []
    new["responses"] = {}
    for code, resp in op["responses"].items():
        r: dict[str, Any] = {"description": resp.get("description", "")}
        for media, item in resp.get("content", {}).items():
            if media not in produces:
                produces.append(media)
            if "schema" in item and "schema" not in r:
                r["schema"] = _schema(item["schema"])
        new["responses"][code] = r
    if produces:
        new["produces"] = produces
    new["security"] = []
    for requirement in op.get("security", []):
        for name, roles in requirement.items():
            new["security"].append({name: []})  # Swagger 2.0 allows scopes on OAuth2 only
            if roles:
                new["x-womblex-scope"] = roles[0]
    return new


def _parameter(p: dict[str, Any]) -> dict[str, Any]:
    new: dict[str, Any] = {k: p[k] for k in ("name", "in", "required", "description") if k in p}
    schema = _schema(p.get("schema", {}))
    new.update({k: v for k, v in schema.items() if k not in ("title", "x-nullable")})
    new.setdefault("type", "string")
    return new


def _form_parameters(ref: dict[str, Any], schemas: dict[str, Any]) -> list[dict[str, Any]]:
    schema = schemas[ref["$ref"].rsplit("/", 1)[-1]] if "$ref" in ref else ref
    required = set(schema.get("required", []))
    params = []
    for name, prop in schema.get("properties", {}).items():
        is_file = prop.get("contentMediaType") or prop.get("items", {}).get("contentMediaType")
        param: dict[str, Any] = {"name": name, "in": "formData", "required": name in required}
        if is_file:  # Swagger 2.0 has no array of files: one `file` field, repeated per file
            param["type"] = "file"
            if prop.get("type") == "array":
                param["description"] = "Repeat this field once per file."
        else:
            param.update({k: v for k, v in _schema(prop).items() if k != "title"})
        params.append(param)
    return params


def _schema(s: Any) -> Any:
    """A JSON Schema 2020-12 fragment as Swagger 2.0's schema subset."""
    if isinstance(s, list):
        return [_schema(x) for x in s]
    if not isinstance(s, dict):
        return s
    new: dict[str, Any] = {}
    for key, value in s.items():
        if key == "$ref":
            new[key] = value.replace("#/components/schemas/", "#/definitions/")
        elif key == "anyOf":
            branches = [b for b in value if b != {"type": "null"}]
            if len(branches) < len(value):
                new["x-nullable"] = True
            if len(branches) == 1:
                new.update(_schema(branches[0]))
            else:  # no union type in Swagger 2.0: left open, the branches kept as an extension
                new["x-anyOf"] = _schema(branches)
        elif key in ("exclusiveMinimum", "exclusiveMaximum") and not isinstance(value, bool):
            new["minimum" if key == "exclusiveMinimum" else "maximum"] = value  # draft-4 form
            new[key] = True
        elif key == "const":
            new["enum"] = [value]
        elif key == "contentMediaType":
            new["format"] = "binary"
        elif key == "examples":
            if isinstance(value, list) and value:
                new["example"] = value[0]
        elif key in ("properties", "patternProperties"):
            new[key] = {k: _schema(v) for k, v in value.items()}
        else:
            new[key] = _schema(value)
    if "$ref" in new and len(new) > 1:  # Swagger 2.0 ignores siblings of $ref
        return {"allOf": [{"$ref": new.pop("$ref")}], **new}
    return new


def render(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def _mcp_tools() -> dict[str, Any]:
    from womblex.api.interface_mcp import mcp_tools_document

    return mcp_tools_document()


def _agent_card() -> dict[str, Any]:
    from womblex.api.interface_a2a import agent_card_document

    return agent_card_document()


#: Committed file name -> builder.
DOCUMENTS = {
    "openapi.json": openapi_document, "swagger.json": swagger_document,
    "mcp-tools.json": _mcp_tools, "agent-card.json": _agent_card,
}


def write_documents(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, build in DOCUMENTS.items():
        target = out_dir / name
        target.write_text(render(build()), encoding="utf-8")
        written.append(target)
    return written


if __name__ == "__main__":
    for path in write_documents(Path(sys.argv[1] if len(sys.argv) > 1 else "docs/interfaces")):
        print(path)
