"""Provisional: the `/v1` API described as an A2A 1.0 Agent Card (`docs/interfaces/agent-card.json`).

Womblex serves no A2A endpoint and no `/.well-known/agent-card.json`. This card describes the
service as an agent would advertise it, for review: one skill per OpenAPI tag, each naming
the operations behind it and the scopes they need. Field names follow the v1.0.1 protocol
definitions (`a2a.proto`) in protobuf JSON form.
"""

from __future__ import annotations

from typing import Any

A2A_PROTOCOL_VERSION = "1.0"
#: No A2A endpoint exists; `.invalid` (RFC 2606) says so where the card needs a URL.
PLACEHOLDER_URL = "https://womblex.invalid/a2a/v1"

#: The media type of each file extension the service ingests (`cli/_shared.SUPPORTED_EXTENSIONS`).
MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls": "application/vnd.ms-excel",
    ".csv": "text/csv",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".jp2": "image/jp2",
}

#: Tags whose operations are not a capability an agent offers.
_NOT_SKILLS = {"ops"}


def agent_card_document() -> dict[str, Any]:
    from womblex import __version__
    from womblex.api.interface_docs import openapi_document

    spec = openapi_document()
    skills: dict[str, dict[str, Any]] = {}
    for path, methods in spec["paths"].items():
        for method, op in methods.items():
            for tag in op.get("tags", []):
                if tag not in _NOT_SKILLS and path != "/v1/hooks/{workflow}/{hook}":  # another system's HMAC
                    _add(skills, tag, method, path, op)
    inputs = sorted(set(MEDIA_TYPES.values()))
    return {
        "name": "Womblex",
        "description": (
            "Extracts text, tables and forms from government document releases (scanned, redacted "
            "and mixed PDFs, DOCX, spreadsheets, images), chunks it and prepares it for semantic "
            "analysis. Provisional: describes the /v1 service API; no A2A endpoint is served."
        ),
        "supportedInterfaces": [
            {"url": PLACEHOLDER_URL, "protocolBinding": "HTTP+JSON", "protocolVersion": A2A_PROTOCOL_VERSION},
        ],
        "provider": {"organization": "DeepCivic", "url": "https://github.com/DeepCivic/womblex"},
        "version": __version__,
        "documentationUrl": "https://github.com/DeepCivic/womblex/blob/main/docs/service-api.md",
        "capabilities": {"streaming": False, "pushNotifications": False, "extendedAgentCard": False},
        "securitySchemes": {
            "bearerAuth": {"httpAuthSecurityScheme": {
                "scheme": "Bearer", "description": "A service token from the client registry (WOMBLEX_API_CLIENTS).",
            }},
        },
        "securityRequirements": [{"schemes": {"bearerAuth": {"list": []}}}],
        "defaultInputModes": ["application/json", *inputs],
        "defaultOutputModes": ["application/json"],
        "skills": [_finish(s) for _, s in sorted(skills.items())],
    }


def _add(skills: dict[str, dict[str, Any]], tag: str, method: str, path: str, op: dict[str, Any]) -> None:
    skill = skills.setdefault(tag, {"id": tag, "name": tag.capitalize(), "operations": [], "scopes": set()})
    skill["operations"].append(f"{op['summary']} (`{method.upper()} {path}`)")
    for requirement in op.get("security", []):
        for roles in requirement.values():
            skill["scopes"].update(roles)


def _finish(skill: dict[str, Any]) -> dict[str, Any]:
    scopes = sorted(skill["scopes"])
    return {
        "id": skill["id"],
        "name": skill["name"],
        "description": "The `" + skill["id"] + "` operations of the /v1 API: " + "; ".join(skill["operations"]) + ".",
        "tags": [skill["id"]],
        "securityRequirements": [{"schemes": {"bearerAuth": {"list": scopes}}}] if scopes else [],
    }
