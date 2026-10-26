"""Tests for the stored workflow object: validation, digest, and the table."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from womblex.cloud.workflow_store import (
    StoredWorkflow,
    WorkflowStore,
    ensure_workflow_schema,
    is_safe_workflow_id,
    validate_workflow_spec,
    workflow_digest,
)

_MINIMAL = {
    "preset": "DEFAULT-Isaacus",
    "steps": [{"stage": "enrich"}, {"stage": "chunk"}],
}


@pytest.fixture
def store(tmp_path):
    dsn = f"sqlite:///{tmp_path}/workflows.db"
    ensure_workflow_schema(dsn)
    ensure_workflow_schema(dsn)  # idempotent: a second call must not raise
    with WorkflowStore(dsn) as s:
        yield s


# --- validation --------------------------------------------------------------


def test_validate_minimal_spec() -> None:
    spec = validate_workflow_spec(_MINIMAL)
    assert [s.stage for s in spec.steps] == ["enrich", "chunk"]
    assert spec.preset == "DEFAULT-Isaacus"
    assert spec.triggers == ()


@pytest.mark.parametrize(("steps", "match"), [
    ([{"stage": "not-a-stage"}], "unknown stage"),
    ([], "at least one step"),
    ([{"stage": "chunk"}, {"stage": "chunk"}], "more than once"),
])
def test_validate_rejects_bad_steps(steps: list, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_workflow_spec({**_MINIMAL, "steps": steps})


@pytest.mark.parametrize("cron", ["not a cron", "a b c d e", "61 * * * *"])
def test_validate_rejects_bad_cron(cron: str) -> None:
    with pytest.raises(ValueError, match="valid cron"):
        validate_workflow_spec({**_MINIMAL, "triggers": [{"cron": cron}]})


def test_validate_checks_timezone_and_accepts_dbos_seconds_cron() -> None:
    triggers = [{"cron": "*/30 * * * * *", "timezone": "Australia/Sydney"}]
    validate_workflow_spec({**_MINIMAL, "triggers": triggers})
    with pytest.raises(ValueError, match="unknown timezone"):
        validate_workflow_spec({**_MINIMAL, "triggers": [{**triggers[0], "timezone": "Not/AZone"}]})


def test_validate_rejects_bad_config_overlay() -> None:
    bad = {**_MINIMAL, "config": {"not_a_real_section": {"foo": "bar"}}}
    with pytest.raises(ValidationError):
        validate_workflow_spec(bad)


def test_validate_overlay_ignores_run_identity_like_a_preset() -> None:
    raw = {**_MINIMAL, "config": {"dataset": {"name": "x"}, "paths": {"input_root": "."}}}
    validate_workflow_spec(raw)  # dropped by `_validate_overlay`, not a TypeError


def test_validate_round_trips_triggers() -> None:
    raw = {
        **_MINIMAL,
        "triggers": [{"cron": "0 0 * * *"}],
    }
    spec = validate_workflow_spec(raw)
    assert spec.triggers[0].cron == "0 0 * * *"


@pytest.mark.parametrize(("workflow_id", "safe"), [
    ("ok-1", True), ("a", True), ("A.b_c-9", True),
    ("", False), (".", False), ("..", False), ("../escape", False), ("a/b", False),
])
def test_workflow_id_safety(workflow_id: str, safe: bool) -> None:
    assert is_safe_workflow_id(workflow_id) is safe


# --- digest -------------------------------------------------------------------


def test_digest_stable_then_changes_with_spec() -> None:
    spec_a = validate_workflow_spec(_MINIMAL)
    spec_b = validate_workflow_spec(_MINIMAL)
    spec_c = validate_workflow_spec({**_MINIMAL, "steps": [{"stage": "chunk"}]})
    assert workflow_digest(spec_a) == workflow_digest(spec_b)
    assert workflow_digest(spec_a).startswith("sha256:")
    assert workflow_digest(spec_a) != workflow_digest(spec_c)


def test_collection_names_the_run_and_stays_out_of_the_digest() -> None:
    plain = validate_workflow_spec(_MINIMAL)
    kept = validate_workflow_spec({**_MINIMAL, "collection": "c1"})
    assert workflow_digest(kept) == workflow_digest(plain)  # where documents are kept is not what runs on them
    assert StoredWorkflow("wf", kept, "").run_id == "c1"
    assert StoredWorkflow("wf", plain, "").run_id == "wf"


@pytest.mark.parametrize("collection", ["", "a/b", "..", "has space"])
def test_an_unsafe_collection_is_refused(collection: str) -> None:
    with pytest.raises(ValueError, match="collection"):
        validate_workflow_spec({**_MINIMAL, "collection": collection})


# --- storage ------------------------------------------------------------------


def test_round_trip_save_get_list_delete(store) -> None:
    spec = validate_workflow_spec(_MINIMAL)
    stored = store.save("wf-1", spec)
    assert stored.id == "wf-1"
    assert stored.digest == workflow_digest(spec)

    fetched = store.get("wf-1")
    assert fetched is not None and fetched.spec.preset == "DEFAULT-Isaacus"
    assert [w.id for w in store.list()] == ["wf-1"]

    store.delete("wf-1")
    assert store.get("wf-1") is None
    assert store.list() == []


def test_a_row_saved_with_a_retired_field_still_loads(store, caplog) -> None:
    import json

    import sqlalchemy as sa

    store.save("wf-old", validate_workflow_spec(_MINIMAL))
    raw = {**_MINIMAL, "conditions": [{"column": "ext", "op": "eq", "value": ".pdf"}],
           "destinations": [{"kind": "webhook", "target": "https://example.invalid/hook"}]}
    with store._engine.begin() as conn:
        conn.execute(sa.update(store._table).values(spec_json=json.dumps(raw), digest="sha256:with-conditions"))
    fetched = store.get("wf-old")
    assert fetched is not None and [s.stage for s in fetched.spec.steps] == ["enrich", "chunk"]
    assert "stored conditions are no longer supported" in caplog.text
    assert "stored destinations predate the current kinds" in caplog.text
    assert fetched.digest == workflow_digest(fetched.spec)  # not the digest stored with the field
    assert "recomputed" in caplog.text


def test_save_rejects_unsafe_id(store) -> None:
    spec = validate_workflow_spec(_MINIMAL)
    with pytest.raises(ValueError, match="unsafe workflow id"):
        store.save("../escape", spec)


def test_save_update_replaces(store) -> None:
    spec_a = validate_workflow_spec(_MINIMAL)
    spec_b = validate_workflow_spec({**_MINIMAL, "steps": [{"stage": "chunk"}]})
    store.save("wf-1", spec_a)
    first = store.get("wf-1")
    updated = store.save("wf-1", spec_b)
    assert updated.created_at == first.created_at
    assert updated.digest == workflow_digest(spec_b)


def test_save_losing_an_insert_race_replaces_the_winner(store, monkeypatch) -> None:
    store.save("wf-1", validate_workflow_spec(_MINIMAL))
    real_get_row = store._get_row
    calls = iter([None])  # the first read misses the row, as a racing save would
    monkeypatch.setattr(store, "_get_row", lambda conn, wid: next(calls, real_get_row(conn, wid)))
    spec = validate_workflow_spec({**_MINIMAL, "steps": [{"stage": "chunk"}]})
    assert store.save("wf-1", spec).digest == workflow_digest(spec)


def test_a_workflow_is_stored_with_its_owner_and_a_resave_replaces_it(store) -> None:
    spec = validate_workflow_spec(_MINIMAL)
    assert store.save("wf", spec).owner is None
    assert store.save("wf", spec, owner="alice").owner == "alice"
    assert store.get("wf").owner == "alice" and [w.owner for w in store.list()] == ["alice"]
    assert store.save("wf", spec).owner is None  # stored again by the CLI


def test_a_table_made_before_owners_gains_the_column_and_reads_unowned(tmp_path) -> None:
    import sqlalchemy as sa

    dsn = f"sqlite:///{tmp_path}/old.db"
    engine = sa.create_engine(dsn)
    with engine.begin() as conn:
        conn.execute(sa.text(
            "CREATE TABLE womblex_workflows (id TEXT PRIMARY KEY, spec_json TEXT NOT NULL, digest TEXT NOT NULL, "
            "created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"))
        conn.execute(sa.text(
            "INSERT INTO womblex_workflows VALUES ('old', :spec, 'sha256:x', 't', 't')"),
            {"spec": json.dumps(validate_workflow_spec(_MINIMAL).model_dump(mode="json"))})
    engine.dispose()
    ensure_workflow_schema(dsn)
    with WorkflowStore(dsn) as s:
        assert s.get("old").owner is None
        assert s.save("old", validate_workflow_spec(_MINIMAL), owner="bob").owner == "bob"


# --- inbound hooks ------------------------------------------------------------

_HOOK = {"name": "crm", "secret_env": "CRM_KEY", "events": ["documents_added", "run_requested"]}


def test_inbound_hooks_parse_and_stay_out_of_the_digest_only_when_empty() -> None:
    plain = validate_workflow_spec(_MINIMAL)
    hooked = validate_workflow_spec({**_MINIMAL, "inbound": [_HOOK]})
    assert hooked.inbound[0].events == ("documents_added", "run_requested")
    assert workflow_digest(validate_workflow_spec({**_MINIMAL, "inbound": []})) == workflow_digest(plain)
    assert workflow_digest(hooked) != workflow_digest(plain)  # which senders may act is part of the recipe


@pytest.mark.parametrize(("hooks", "match"), [
    ([{**_HOOK, "name": "a/b"}], "safe token"),
    ([_HOOK, {**_HOOK, "secret_env": "OTHER"}], "more than once"),
    ([{**_HOOK, "events": ["nope"]}], "subset of"),
    ([{**_HOOK, "events": []}], "subset of"),
    ([{**_HOOK, "secret_env": " "}], "secret_env"),
    ([_HOOK, {**_HOOK, "name": "erp"}], "share a secret_env"),
])
def test_a_bad_inbound_hook_is_refused(hooks: list[dict], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_workflow_spec({**_MINIMAL, "inbound": hooks})


def test_a_secret_shared_with_another_stored_workflow_is_refused_unless_insecure(store, monkeypatch, caplog) -> None:
    store.save("one", validate_workflow_spec({**_MINIMAL, "inbound": [_HOOK]}))
    rival = validate_workflow_spec({**_MINIMAL, "inbound": [{**_HOOK, "name": "other"}]})
    with pytest.raises(ValueError, match="also named by stored workflow 'one'"):
        store.save("two", rival)
    assert store.get("two") is None
    store.save("one", validate_workflow_spec({**_MINIMAL, "inbound": [_HOOK]}))  # resaving itself is fine
    monkeypatch.setenv("WOMBLEX_INSECURE_INBOUND", "1")
    assert store.save("two", rival).id == "two"
    assert "allowed by WOMBLEX_INSECURE_INBOUND" in caplog.text


def test_a_spec_cannot_turn_the_insecure_switch_on() -> None:
    with pytest.raises(ValidationError):
        validate_workflow_spec({**_MINIMAL, "insecure_inbound": True})
    with pytest.raises(ValueError, match="share a secret_env"):
        validate_workflow_spec({**_MINIMAL, "inbound": [_HOOK, {**_HOOK, "name": "erp"}]})
