"""`/v1/collections`: owner-scoped reads, adding to and rerunning a collection through a stored workflow."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("dbos")
from fastapi.testclient import TestClient

from tests.test_workflow_runner import _PRESET
from tests.test_workflow_schedules import _inbox, saved_preset  # noqa: F401
from tests.test_workflows import fleet, work  # noqa: F401
from womblex.api.app import create_api_app
from womblex.api.auth import hash_token, parse_registry
from womblex.cloud.jobs import RunBoard
from womblex.cloud.workflow_store import WorkflowStore, ensure_workflow_schema, validate_workflow_spec


def _spec(ingest: str, steps: list[str], collection: str = "c1") -> object:
    return validate_workflow_spec({
        "preset": _PRESET.name, "steps": [{"stage": s} for s in steps], "collection": collection,
        "input_binding": {"store": ingest, "ingest_prefix": "inbox"},
    })


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def api(fleet, saved_preset) -> TestClient:  # noqa: F811
    import os
    import shutil

    store_dir, ingest, _ = fleet
    shutil.copytree(os.environ["WOMBLEX_UI_PRESETS_DIR"], Path(str(store_dir)) / "presets")  # the API reads the store's
    _inbox(fleet)
    ensure_workflow_schema(None)
    with WorkflowStore(None) as store:
        store.save("feed", _spec(ingest, ["normalise"]), owner="alice")
        store.save("feed-b", _spec(ingest, ["normalise"], "c2"), owner="bob")
    registry = parse_registry({"clients": [
        {"client_id": "alice", "token_sha256": hash_token("ta"), "scopes": ["submit", "read"]},
        {"client_id": "bob", "token_sha256": hash_token("tb"), "scopes": ["read"]},
        {"client_id": "ops", "token_sha256": hash_token("to"), "scopes": ["admin"]},
    ]})
    return TestClient(create_api_app(store_uri=str(store_dir), db_dsn="", registry=registry, ingest_uri=ingest))


def test_a_caller_lists_its_own_collections_and_admin_lists_all(api) -> None:
    mine = api.get("/v1/collections", headers=auth("ta")).json()["collections"]
    assert mine == [{"name": "c1", "workflows": ["feed"]}]
    assert {c["name"] for c in api.get("/v1/collections", headers=auth("to")).json()["collections"]} == {"c1", "c2"}


def test_another_owners_collection_is_404_on_every_route(api) -> None:
    for method, path in (("get", ""), ("post", "/add"), ("post", "/rerun")):
        assert getattr(api, method)(f"/v1/collections/c2{path}", headers=auth("ta")).status_code == 404
    assert api.get("/v1/collections/c2", headers=auth("to")).status_code == 200


def test_adding_needs_the_submit_scope(api) -> None:
    assert api.post("/v1/collections/c2/add", headers=auth("tb")).status_code == 403


def test_add_fires_the_stored_workflow_into_the_collection(api, fleet) -> None:  # noqa: F811
    sent = api.post("/v1/collections/c1/add", headers=auth("ta"))
    assert sent.status_code == 202, sent.json()
    assert sent.json()["workflow"] == "feed"
    assert sent.json()["firing"].startswith("c1:trigger:manual-")
    work(fleet, idle_timeout=4)
    detail = api.get("/v1/collections/c1", headers=auth("ta")).json()
    assert detail["workflows"] == ["feed"]
    assert (detail["documents"], detail["current_documents"]) == (1, 1)
    assert [w["name"] for w in detail["written_by"]] == ["feed"]
    with RunBoard(None) as board:
        assert [r.run_id for r in board.runs(owner="alice")] == ["c1"]  # owned by the workflow's owner


def test_a_collection_with_no_documents_to_add_is_refused_before_any_row(api, fleet) -> None:  # noqa: F811
    for f in (Path(fleet[1]) / "inbox").iterdir():
        f.unlink()
    refused = api.post("/v1/collections/c1/add", headers=auth("ta"))
    assert refused.status_code == 409 and refused.json()["detail"]["code"] == "no_documents"
    with RunBoard(None) as board:
        assert board.runs() == []


def test_rerun_dispatches_the_stored_steps_over_the_collection(api, fleet) -> None:  # noqa: F811
    assert api.post("/v1/collections/c1/add", headers=auth("ta")).status_code == 202
    work(fleet, idle_timeout=4)
    with WorkflowStore(None) as store:
        store.save("feed", _spec(fleet[1], ["normalise", "spellfix"]), owner="alice")
    sent = api.post("/v1/collections/c1/rerun", headers=auth("ta"))
    assert sent.status_code == 202 and sent.json()["stages"] == ["normalise", "spellfix"]
    work(fleet, idle_timeout=4)
    assert (Path(str(fleet[0])) / "runs/c1/documents/batch-0001.spellfix_text.parquet").exists()


def test_a_collection_fed_by_several_workflows_needs_one_named(api, fleet) -> None:  # noqa: F811
    with WorkflowStore(None) as store:
        store.save("feed-2", _spec(fleet[1], ["normalise"]), owner="alice")
    both = api.post("/v1/collections/c1/add", headers=auth("ta"))
    assert both.status_code == 409 and both.json()["detail"]["workflows"] == ["feed", "feed-2"]
    assert api.post("/v1/collections/c1/add", headers=auth("ta"), json={"workflow": "nope"}).status_code == 404
    assert api.post("/v1/collections/c1/add", headers=auth("ta"), json={"workflow": "feed-2"}).status_code == 202
