"""`/v1/store/runs` and the per-run reads over a run's files: stage presence, the shard audit, a document's chunk detail and the batch logs.

The store is read directly, so these answer for a run the queue never saw
(one `womblex run` wrote, or one seeded). Listing the store is ``admin``: it
names every run, whoever submitted it. A per-run read needs ``read`` and the
run's ownership, through *authorise*; a document's detail carries raw chunk
text and PII, so it needs ``read_raw``. The bodies are :mod:`womblex.store.run_reads`,
:mod:`womblex.store.document_detail` and :mod:`womblex.store.run_logs`.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse

from womblex.api.auth import Caller
from womblex.api.models import DocumentDetail, RunLog, RunLogs, StagePresence, StoredRun, StoredRunList
from womblex.cloud.deployment import DeploymentSettings
from womblex.store import run_logs, run_reads
from womblex.store.document_detail import document_detail
from womblex.store.remote_manifest import StoreUnreachable
from womblex.store.retention import STAGE_SUFFIXES


def unreachable(e: StoreUnreachable) -> HTTPException:
    return HTTPException(status_code=503, detail=f"run store unreachable: {e}")


def not_found(run_id: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"run not found: {run_id}")


def runs_router(
    settings: DeploymentSettings,
    can_read: Callable[..., Caller],
    can_admin: Callable[..., Caller],
    can_read_raw: Callable[..., Caller],
    authorise: Callable[[Caller, str], None],
) -> APIRouter:
    router = APIRouter(tags=["runs"])

    @router.get("/v1/store/runs", response_model=StoredRunList, dependencies=[Depends(can_admin)])
    def list_stored_runs() -> StoredRunList:
        try:
            runs = run_reads.list_run_summaries(settings)
        except StoreUnreachable as e:
            raise unreachable(e) from e
        return StoredRunList(runs=[StoredRun(**asdict(r)) for r in runs])

    @router.get("/v1/runs/{run_id}/stages/{stage}", response_model=StagePresence)
    def get_stage_presence(run_id: str, stage: str, caller: Caller = Depends(can_read)) -> StagePresence:  # noqa: B008
        """The documents *stage* wrote a sidecar row for; 400 for a stage with no sidecar of its own."""
        if stage not in STAGE_SUFFIXES:
            raise HTTPException(status_code=400, detail=f"unknown stage: {stage}; one of {sorted(STAGE_SUFFIXES)}")
        authorise(caller, run_id)
        try:
            hashes = run_reads.stage_presence(settings, run_id, stage)
        except StoreUnreachable as e:
            raise unreachable(e) from e
        if hashes is None:
            raise not_found(run_id)
        return StagePresence(run_id=run_id, stage=stage, source_hashes=hashes)

    @router.get("/v1/runs/{run_id}/audit")
    def get_audit(run_id: str, caller: Caller = Depends(can_read)) -> dict[str, Any]:  # noqa: B008
        """`verify-shards`' cross-batch structural audit of the run's extraction shards."""
        authorise(caller, run_id)
        try:
            audit = run_reads.shard_audit(settings, run_id)
        except StoreUnreachable as e:
            raise unreachable(e) from e
        if audit is None:
            raise not_found(run_id)
        return audit

    @router.get("/v1/runs/{run_id}/documents/{source_hash}/detail", response_model=DocumentDetail)
    def get_document_detail(run_id: str, source_hash: str, caller: Caller = Depends(can_read_raw)) -> DocumentDetail:  # noqa: B008
        """The document's chunks, entities, PII spans, narrative money and quality rows."""
        authorise(caller, run_id)
        try:
            detail = document_detail(settings, run_id, source_hash)
        except StoreUnreachable as e:
            raise unreachable(e) from e
        if detail is None:
            raise not_found(run_id)
        return DocumentDetail(run_id=run_id, source_hash=source_hash, **detail)

    @router.get("/v1/runs/{run_id}/logs", response_model=RunLogs)
    def list_logs(run_id: str, caller: Caller = Depends(can_read)) -> RunLogs:  # noqa: B008
        authorise(caller, run_id)
        try:
            logs = run_logs.list_run_logs(settings, run_id)
        except StoreUnreachable as e:
            raise unreachable(e) from e
        if logs is None:
            raise not_found(run_id)
        return RunLogs(run_id=run_id, logs=[RunLog(**entry) for entry in logs])

    @router.get("/v1/runs/{run_id}/logs/{name}", response_class=PlainTextResponse)
    def read_log(run_id: str, name: str, download: bool = False, caller: Caller = Depends(can_read)) -> PlainTextResponse:  # noqa: B008
        """One log as text (``?download=true`` names it as a file).

        A name that is not a log's and an absent log give the same 404 with the
        logs there are, so the route cannot probe outside the run's ``logs/``.
        """
        authorise(caller, run_id)
        try:
            text = run_logs.read_run_log(settings, run_id, name)
            available = None if text is not None else run_logs.list_run_logs(settings, run_id)
        except StoreUnreachable as e:
            raise unreachable(e) from e
        if text is None:
            if available is None:
                raise not_found(run_id)
            raise HTTPException(status_code=404, detail={"message": f"log not found: {name}", "available": available})
        headers = {"Content-Disposition": f'attachment; filename="{name}"'} if download else None
        return PlainTextResponse(text, headers=headers)

    return router
