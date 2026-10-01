"""Response and request models for the `/v1` API — the OpenAPI surface."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class Health(BaseModel):
    status: Literal["ok"] = "ok"


class Ready(BaseModel):
    ready: bool
    store: bool
    queue: bool


class RunStatus(BaseModel):
    """One run's queue rollup. ``state`` is derived from ``counts``."""

    run_id: str
    state: Literal["pending", "running", "done", "failed"]
    counts: dict[str, int]
    total: int
    created_at: str | None = None
    updated_at: str | None = None


class RunList(BaseModel):
    runs: list[RunStatus]


class RunManifest(BaseModel):
    run_id: str
    documents: list[dict[str, Any]]


class RunMetrics(BaseModel):
    run_id: str
    stats: dict[str, int]
    workers: list[dict[str, Any]]
    throughput: dict[str, Any]
