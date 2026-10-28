"""The `openai-compatible` deployment: any server that answers `/v1/embeddings`.

`OPENAI_COMPAT_BASE_URL` names it (hosted: `https://api.openai.com/v1`; self-hosted:
`http://localhost:8000/v1`), `OPENAI_COMPAT_API_KEY` is sent as a bearer token when set. A hosted
vendor bills per request; a self-hosted server bills nothing and is off the network, which
`OPENAI_COMPAT_BILLED=false` says. The setting is read when the deployment is registered.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from urllib.parse import urlsplit

from womblex.utils.deployment_registry import Deployment, client_for, register

NAME = "openai-compatible"
BASE_URL_ENV = "OPENAI_COMPAT_BASE_URL"
API_KEY_ENV = "OPENAI_COMPAT_API_KEY"  # pragma: allowlist secret -- env var name, not a key
BILLED_ENV = "OPENAI_COMPAT_BILLED"
TIMEOUT_SECONDS = 60.0
_PROBE = "The applicant, Jane Citizen, signed the declaration on 5 March 2024."


def billed_by_setting() -> bool:
    """Whether calls are billed: yes unless `OPENAI_COMPAT_BILLED` says otherwise (`false`, `0`, `no`, `off`)."""
    return os.environ.get(BILLED_ENV, "true").strip().lower() not in {"false", "0", "no", "off"}


def make_client(models: Iterable[str] = ()) -> object:
    """An `httpx.Client` rooted at the base URL, with the bearer token when a key is set."""
    import httpx

    base = (os.environ.get(BASE_URL_ENV) or "").strip().rstrip("/")
    if not base:
        raise ValueError(f"{BASE_URL_ENV} is not set")
    key = (os.environ.get(API_KEY_ENV) or "").strip()
    return httpx.Client(base_url=base, timeout=TIMEOUT_SECONDS,
                        headers={"Authorization": f"Bearer {key}"} if key else {})


def check(models: Sequence[str]) -> None:
    """Free: the server answers `/models`. A 404 still means it answers; a refused key or a 5xx does not."""
    client = make_client()
    try:
        response = client.get("/models")  # type: ignore[attr-defined]
        if response.status_code != 404:
            response.raise_for_status()
    finally:
        client.close()  # type: ignore[attr-defined]


def smoke(slot: str, model: str) -> None:
    """One billed request: embed a sentence and expect a vector back."""
    client = client_for(NAME)
    response = client.post("/embeddings", json={"model": model, "input": [_PROBE]})  # type: ignore[attr-defined]
    response.raise_for_status()
    if not response.json()["data"][0]["embedding"]:
        raise RuntimeError("returned no embedding for the probe sentence")


def detail(model: str) -> dict[str, str]:
    """Where it ran: the host only. The path, query and any credentials in the URL are left out."""
    host = urlsplit(os.environ.get(BASE_URL_ENV) or "").hostname
    return {"endpoint": host} if host else {}


_billed = billed_by_setting()
register(Deployment(
    name=NAME,
    check=check,
    needs_network=_billed,
    billed=_billed,
    env=((BASE_URL_ENV,),),
    env_optional=(API_KEY_ENV, BILLED_ENV),
    smoke=smoke if _billed else None,
    detail=detail,
    make_client=make_client,
))
