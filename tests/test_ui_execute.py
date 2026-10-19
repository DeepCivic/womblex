"""Console execution: enqueue dispatch and ingest preflight (womblex.ui)."""
from __future__ import annotations

from pathlib import Path
from typing import Self

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from womblex.ui.app import create_app


class _FakeQueue:
    """A `RunBoard` stand-in — the execute happy path must not need Postgres.

    Records what `enqueue_extraction` hands it (schema call, run_id, specs) so
    a test can assert the batching without a live database, and returns a
    fixed `newly` count for `enqueue`.
    """

    last: _FakeQueue | None = None
    schema_ensured = False

    def __init__(self, dsn: str, **_kw: object) -> None:
        self.dsn = dsn
        self.enqueued: tuple[str, list] | None = None
        _FakeQueue.last = self

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def enqueue(self, run_id: str, specs: list, *, owner: str | None = None, config: dict | None = None) -> int:
        self.enqueued = (run_id, specs)
        return len(specs)


@pytest.fixture(autouse=True)
def _fake_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dispatch provisions the DBOS schema through a module function; record, don't run it."""
    _FakeQueue.schema_ensured = False
    monkeypatch.setattr(
        "womblex.cloud.dispatch.ensure_schema",
        lambda _dsn: setattr(_FakeQueue, "schema_ensured", True),
    )


def _seed_store_inputs(store_root: Path, prefix: str, names: list[str]) -> None:
    """Write source documents under `<store_root>/<prefix>` for enqueue to list."""
    in_dir = store_root / prefix
    in_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        (in_dir / name).write_bytes(b"%PDF-1.4 stub")


class TestExecuteApi:
    """Dispatch requires a store, an ingest location, and a DSN
    (docs/functional_requirements.md, Requirement 19)."""

    def test_status_reflects_the_flags(self, tmp_path: Path) -> None:
        client = TestClient(create_app(output_root=tmp_path))
        body = client.get("/api/execute/status").json()
        assert body["can_execute"] is False
        assert body == {
            "can_execute": False,
            "has_store": False, "has_ingest": False, "has_queue": False,
            "ingest_uri": None, "output_uri": str(tmp_path),
        }

    def test_status_advertises_no_stage_list(self, tmp_path: Path) -> None:
        """The pipeline's shape is the composer's graph, not a dispatch flag.

        `stages` here used to be `STAGE_NAMES` — read as "what this console
        dispatches" beside `can_execute`, while no dispatch path for a stage
        existed and no caller read the field.
        """
        client = TestClient(create_app(output_root=tmp_path))
        assert "stages" not in client.get("/api/execute/status").json()

    def test_status_can_execute_with_a_store_ingest_and_queue(self, tmp_path: Path) -> None:
        client = TestClient(create_app(
            store_uri=str(tmp_path / "store"), ingest_uri=str(tmp_path / "inbox"),
            db_dsn="postgresql://x/y",
        ))
        body = client.get("/api/execute/status").json()
        assert body["can_execute"] is True
        assert body["ingest_uri"] == str(tmp_path / "inbox")
        assert body["output_uri"] == str(tmp_path / "store")

    def test_enqueue_conflict_without_a_store(self, tmp_path: Path) -> None:
        """A local output_root can configure and audit but not dispatch."""
        client = TestClient(create_app(
            output_root=tmp_path, ingest_uri=str(tmp_path / "inbox"), db_dsn="postgresql://x/y",
        ))
        resp = client.post("/api/execute/enqueue", json={})
        assert resp.status_code == 409

    def test_enqueue_conflict_without_an_ingest_location(self, tmp_path: Path) -> None:
        pytest.importorskip("fsspec")
        client = TestClient(create_app(
            store_uri=str(tmp_path / "store"), db_dsn="postgresql://x/y",
        ))
        resp = client.post("/api/execute/enqueue", json={})
        assert resp.status_code == 409

    def test_enqueue_conflict_without_a_queue(self, tmp_path: Path) -> None:
        pytest.importorskip("fsspec")
        client = TestClient(create_app(
            store_uri=str(tmp_path / "store"), ingest_uri=str(tmp_path / "inbox"),
        ))
        resp = client.post("/api/execute/enqueue", json={})
        assert resp.status_code == 409

    def test_enqueue_400_when_no_documents_match(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty ingest root is bad input (400), not a disabled console (403/409)."""
        pytest.importorskip("fsspec")
        monkeypatch.setattr("womblex.cloud.jobs.RunBoard", _FakeQueue)
        ingest_root = tmp_path / "inbox"
        _seed_store_inputs(ingest_root, "", ["notes.txt"])  # unsupported ext
        client = TestClient(create_app(
            store_uri=str(tmp_path / "store"), ingest_uri=str(ingest_root),
            db_dsn="postgresql://x/y",
        ))
        resp = client.post("/api/execute/enqueue", json={})
        assert resp.status_code == 400

    def test_enqueue_plans_batches_into_the_queue(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """List the whole ingest root (no prefix posted), batch, one idempotent
        row each, stamped with the ingest root."""
        pytest.importorskip("fsspec")
        monkeypatch.setattr("womblex.cloud.jobs.RunBoard", _FakeQueue)
        ingest_root = tmp_path / "inbox"
        _seed_store_inputs(ingest_root, "", [f"doc-{i}.pdf" for i in range(5)])
        client = TestClient(create_app(
            store_uri=str(tmp_path / "store"), ingest_uri=str(ingest_root),
            db_dsn="postgresql://x/y",
        ))
        resp = client.post(
            "/api/execute/enqueue", json={"run_id": "run-exec", "batch_size": 2},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["run_id"] == "run-exec"
        assert body["document_count"] == 5
        assert body["batch_count"] == 3  # 2 + 2 + 1
        assert body["newly_enqueued"] == 3
        assert body["shard_prefix"] == "runs/run-exec/documents"

        queue = _FakeQueue.last
        assert queue is not None and queue.schema_ensured
        run_id, specs = queue.enqueued  # type: ignore[misc]
        assert run_id == "run-exec"
        assert [s.batch_num for s in specs] == [1, 2, 3]
        assert [len(s.input_keys) for s in specs] == [2, 2, 1]
        assert all(s.shard_prefix == "runs/run-exec/documents" for s in specs)
        assert all(s.ingest_root == str(ingest_root) for s in specs)

    def test_enqueue_refuses_a_nested_ingest_layout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """P11: a run ingests what is directly under the location it is given.
        The console refuses the same layout the local CLI does, as bad input,
        and nothing reaches the queue."""
        pytest.importorskip("fsspec")
        monkeypatch.setattr("womblex.cloud.jobs.RunBoard", _FakeQueue)
        _FakeQueue.last = None
        ingest_root = tmp_path / "inbox"
        _seed_store_inputs(ingest_root, "", ["top.pdf"])
        _seed_store_inputs(ingest_root, "2026-08", ["nested.pdf"])
        client = TestClient(create_app(
            store_uri=str(tmp_path / "store"), ingest_uri=str(ingest_root),
            db_dsn="postgresql://x/y",
        ))
        resp = client.post(
            "/api/execute/enqueue", json={"run_id": "run-exec", "batch_size": 2},
        )
        assert resp.status_code == 400
        assert "2026-08/" in resp.json()["detail"]
        assert _FakeQueue.last is None

    def test_the_prefix_previewed_is_the_prefix_enqueued(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """P12: preview and dispatch cannot disagree about which documents a
        press will run — the count the preflight reports for a prefix is the
        count that prefix enqueues, and the keys are scoped to it."""
        pytest.importorskip("fsspec")
        monkeypatch.setattr("womblex.cloud.jobs.RunBoard", _FakeQueue)
        ingest_root = tmp_path / "inbox"
        _seed_store_inputs(ingest_root, "2026-08", ["a.pdf", "b.pdf"])
        _seed_store_inputs(ingest_root, "2026-09", ["c.pdf"])
        client = TestClient(create_app(
            store_uri=str(tmp_path / "store"), ingest_uri=str(ingest_root),
            db_dsn="postgresql://x/y",
        ))
        preview = client.get(
            "/api/execute/ingest", params={"input_prefix": "2026-08"},
        ).json()
        resp = client.post("/api/execute/enqueue", json={
            "run_id": "run-p", "input_prefix": preview["input_prefix"],
        })
        assert resp.status_code == 200
        assert resp.json()["document_count"] == preview["document_count"] == 2
        _, specs = _FakeQueue.last.enqueued  # type: ignore[union-attr, misc]
        assert sorted(specs[0].input_keys) == ["2026-08/a.pdf", "2026-08/b.pdf"]

    def test_the_keys_enqueued_carry_no_dot_segment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A queued key is downloaded by the worker and recorded as the
        document's `source_relpath`, so a prefix spelled `./2026-08` must not
        publish provenance naming a key that only a local path would resolve."""
        pytest.importorskip("fsspec")
        monkeypatch.setattr("womblex.cloud.jobs.RunBoard", _FakeQueue)
        ingest_root = tmp_path / "inbox"
        _seed_store_inputs(ingest_root, "2026-08", ["a.pdf"])
        client = TestClient(create_app(
            store_uri=str(tmp_path / "store"), ingest_uri=str(ingest_root),
            db_dsn="postgresql://x/y",
        ))
        resp = client.post("/api/execute/enqueue", json={
            "run_id": "run-dot", "input_prefix": "./2026-08/",
        })
        assert resp.status_code == 200
        _, specs = _FakeQueue.last.enqueued  # type: ignore[union-attr, misc]
        assert specs[0].input_keys == ["2026-08/a.pdf"]

    def test_enqueue_refuses_a_prefix_escaping_the_ingest_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Bad input (400), before any listing, and nothing reaches the queue —
        the treatment an unsafe run id already gets."""
        pytest.importorskip("fsspec")
        monkeypatch.setattr("womblex.cloud.jobs.RunBoard", _FakeQueue)
        _FakeQueue.last = None
        ingest_root = tmp_path / "inbox"
        _seed_store_inputs(ingest_root, "", ["top.pdf"])
        client = TestClient(create_app(
            store_uri=str(tmp_path / "store"), ingest_uri=str(ingest_root),
            db_dsn="postgresql://x/y",
        ))
        resp = client.post(
            "/api/execute/enqueue", json={"input_prefix": "../../etc"},
        )
        assert resp.status_code == 400
        assert _FakeQueue.last is None

    def test_enqueue_mints_a_run_id_when_omitted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("fsspec")
        monkeypatch.setattr("womblex.cloud.jobs.RunBoard", _FakeQueue)
        ingest_root = tmp_path / "inbox"
        _seed_store_inputs(ingest_root, "", ["doc-0.pdf"])
        client = TestClient(create_app(
            store_uri=str(tmp_path / "store"), ingest_uri=str(ingest_root),
            db_dsn="postgresql://x/y",
        ))
        resp = client.post("/api/execute/enqueue", json={})
        assert resp.json()["run_id"].startswith("run-")

    def test_enqueue_rejects_a_batch_size_below_one(self, tmp_path: Path) -> None:
        """Pydantic's `ge=1` refuses it at the boundary (422), before the guard runs."""
        client = TestClient(create_app(
            store_uri=str(tmp_path / "store"), ingest_uri=str(tmp_path / "inbox"),
            db_dsn="postgresql://x/y",
        ))
        resp = client.post("/api/execute/enqueue", json={"batch_size": 0})
        assert resp.status_code == 422


class TestIngestPreflight:
    """`GET /api/execute/ingest` — reachability + document count."""

    def test_no_ingest_configured(self, tmp_path: Path) -> None:
        client = TestClient(create_app(output_root=tmp_path))
        body = client.get("/api/execute/ingest").json()
        assert body == {
            "uri": None, "input_prefix": "", "kind": None, "reachable": False,
            "document_count": 0, "sample": [], "subdirectories": [],
            "error": "no ingest location configured",
        }

    def test_nested_prefixes_report_the_refusal_not_a_count(self, tmp_path: Path) -> None:
        """P11: nested documents are refused, not ingested, so the preflight
        reports the refusal rather than a count the enqueue will not honour.

        Object stores have no folders — `inbox/2026-08/foo.pdf` is one key — so
        this layout is a normal upload and the operator needs to be told why it
        is rejected, not shown 0 documents with no reason.

        P12: the refusal also names the immediate subdirectories that hold
        documents, with a count each and the prefix that selects one — the
        recovery the message describes, made selectable from this screen
        instead of from the CLI."""
        pytest.importorskip("fsspec")
        ingest_root = tmp_path / "inbox"
        _seed_store_inputs(ingest_root, "", ["top.pdf"])
        _seed_store_inputs(ingest_root, "2026-08/health", ["nested.pdf"])
        _seed_store_inputs(ingest_root, "scratch", ["a.pdf", "b.pdf"])
        client = TestClient(create_app(
            store_uri=str(tmp_path / "store"), ingest_uri=str(ingest_root),
        ))
        body = client.get("/api/execute/ingest").json()
        assert body["reachable"] is True
        assert body["document_count"] == 0
        assert "2026-08/" in body["error"]
        assert body["subdirectories"] == [
            {"name": "2026-08", "input_prefix": "2026-08", "document_count": 1},
            {"name": "scratch", "input_prefix": "scratch", "document_count": 2},
        ]

    def test_a_refusal_under_a_prefix_names_prefixes_that_select_deeper(
        self, tmp_path: Path
    ) -> None:
        """The subdirectory prefixes are ingest-relative, so one press deeper
        selects the next level rather than restarting from the root."""
        pytest.importorskip("fsspec")
        ingest_root = tmp_path / "inbox"
        _seed_store_inputs(ingest_root, "2026-08/health", ["nested.pdf"])
        client = TestClient(create_app(
            store_uri=str(tmp_path / "store"), ingest_uri=str(ingest_root),
        ))
        body = client.get("/api/execute/ingest", params={"input_prefix": "2026-08"}).json()
        assert body["input_prefix"] == "2026-08"
        assert body["subdirectories"] == [
            {"name": "health", "input_prefix": "2026-08/health", "document_count": 1},
        ]

    def test_a_prefix_resolving_flat_reports_its_count(self, tmp_path: Path) -> None:
        """P12: the recovery the refusal names works, from the refusal alone —
        the prefix it hands back previews as a count, with no second tool and
        nothing for the operator to compose by hand."""
        pytest.importorskip("fsspec")
        ingest_root = tmp_path / "inbox"
        _seed_store_inputs(ingest_root, "", ["top.pdf"])
        _seed_store_inputs(ingest_root, "2026-08", ["a.pdf", "b.pdf"])
        client = TestClient(create_app(
            store_uri=str(tmp_path / "store"), ingest_uri=str(ingest_root),
        ))
        refused = client.get("/api/execute/ingest").json()
        chosen = refused["subdirectories"][0]["input_prefix"]
        body = client.get("/api/execute/ingest", params={"input_prefix": chosen}).json()
        assert body["error"] is None
        assert body["input_prefix"] == "2026-08"
        assert body["document_count"] == 2
        # The sample is the keys, prefix and all, as the enqueue would stage them.
        assert sorted(body["sample"]) == ["2026-08/a.pdf", "2026-08/b.pdf"]

    def test_a_prefix_is_echoed_normalised_so_the_composer_posts_one_scope(
        self, tmp_path: Path
    ) -> None:
        """Spelling-by-spelling normalisation is pinned on the shared rule in
        `test_discovery_parity`; what this holds is that the preflight echoes
        the normalised prefix, since that string is what the composer posts."""
        pytest.importorskip("fsspec")
        ingest_root = tmp_path / "inbox"
        _seed_store_inputs(ingest_root, "2026-08", ["a.pdf"])
        client = TestClient(create_app(
            store_uri=str(tmp_path / "store"), ingest_uri=str(ingest_root),
        ))
        body = client.get(
            "/api/execute/ingest", params={"input_prefix": "./2026-08/"},
        ).json()
        assert body["input_prefix"] == "2026-08"
        assert body["sample"] == ["2026-08/a.pdf"]

    def test_a_prefix_escaping_the_ingest_root_is_refused(self, tmp_path: Path) -> None:
        """Same footing as the unsafe-run-id check: 400, and no listing."""
        pytest.importorskip("fsspec")
        ingest_root = tmp_path / "inbox"
        _seed_store_inputs(ingest_root, "", ["top.pdf"])
        _seed_store_inputs(tmp_path / "elsewhere", "", ["secret.pdf"])
        client = TestClient(create_app(
            store_uri=str(tmp_path / "store"), ingest_uri=str(ingest_root),
            db_dsn="postgresql://x/y",
        ))
        resp = client.get("/api/execute/ingest", params={"input_prefix": "../elsewhere"})
        assert resp.status_code == 400
        assert "unsafe input_prefix" in resp.json()["detail"]

    def test_reachable_ingest_reports_count_and_sample(self, tmp_path: Path) -> None:
        pytest.importorskip("fsspec")
        ingest_root = tmp_path / "inbox"
        _seed_store_inputs(ingest_root, "", [f"doc-{i}.pdf" for i in range(3)])
        client = TestClient(create_app(
            store_uri=str(tmp_path / "store"), ingest_uri=str(ingest_root),
        ))
        body = client.get("/api/execute/ingest").json()
        assert body["kind"] == "local"
        assert body["reachable"] is True
        assert body["document_count"] == 3
        assert sorted(body["sample"]) == ["doc-0.pdf", "doc-1.pdf", "doc-2.pdf"]
