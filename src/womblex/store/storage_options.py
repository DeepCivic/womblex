"""Storage options for one location, passed to fsspec unchanged.

A location (the run store, the ingest location, a workflow destination) may carry its own
fsspec ``storage_options``. Womblex adds no per-backend option: the dictionary is the
backend's own (``account_name`` for ``adlfs``, ``endpoint_url`` under ``client_kwargs`` for
an S3-compatible store). A secret is never written in one: it is ``{"env": "NAME"}``, the
name of an environment variable, resolved only when the location is opened. What is shown or
recorded is the variable's name, never its value.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any, cast

#: A key containing one of these names a secret; its value may only be an ``{"env": NAME}`` reference.
_SECRET_WORDS = ("key", "secret", "token", "password", "passwd", "sas", "credential", "connection_string")

MASK = "•" * 8


def looks_secret(key: str) -> bool:
    """Whether an option called *key* is taken to hold a secret."""
    lowered = key.lower()
    return any(word in lowered for word in _SECRET_WORDS)


def _is_ref(value: Any) -> bool:
    return isinstance(value, Mapping) and set(value) == {"env"} and isinstance(value["env"], str)


def parse_options(raw: str, source: str) -> dict[str, Any]:
    """The options object in *raw*; ``ValueError`` naming *source* unless it is a JSON object."""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"{source} is not valid JSON: {e}") from e
    if not isinstance(value, dict):
        raise ValueError(  # noqa: TRY004 -- a configuration error, reported as every other is
            f"{source} must be a JSON object of storage options, not {type(value).__name__}")
    return value


def resolve_options(options: Mapping[str, Any] | None, source: str = "storage options") -> dict[str, Any]:
    """*options* with each ``{"env": NAME}`` replaced by that variable's value, nested ones too.

    Raises ``ValueError`` naming the option and the variable when one is unset or empty.
    """
    def walk(node: Any, path: str) -> Any:
        if _is_ref(node):
            name = node["env"]
            value = os.environ.get(name)
            if not value:
                raise ValueError(f"{source}: {path} reads environment variable {name}, which is not set")
            return value
        if isinstance(node, Mapping):
            return {k: walk(v, f"{path}.{k}" if path else k) for k, v in node.items()}
        return node

    return cast(dict[str, Any], walk(dict(options or {}), ""))


def secrets_outside_env(options: Mapping[str, Any] | None) -> list[str]:
    """Paths of secret-looking options written as literal strings instead of ``{"env": NAME}``."""
    found: list[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, Mapping) and not _is_ref(node):
            for k, v in node.items():
                here = f"{path}.{k}" if path else k
                if looks_secret(k) and isinstance(v, str):
                    found.append(here)
                else:
                    walk(v, here)

    walk(options or {}, "")
    return found


def describe_options(options: Mapping[str, Any] | None) -> dict[str, Any]:
    """*options* as it may be shown: an env reference as itself, a secret-looking literal masked."""
    def walk(node: Any, key: str = "") -> Any:
        if _is_ref(node):
            return {"env": node["env"]}
        if isinstance(node, Mapping):
            return {k: walk(v, k) for k, v in node.items()}
        return MASK if looks_secret(key) and node not in (None, "") else node

    return cast(dict[str, Any], walk(dict(options or {})))


def merge_options(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    """*over* on top of *base*, key by key; a dictionary in both (``client_kwargs``) is merged the same way."""
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping) and not _is_ref(value):
            out[key] = merge_options(out[key], value)
        else:
            out[key] = value
    return out
