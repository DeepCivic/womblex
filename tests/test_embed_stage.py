"""Tests for the per-stage embed wiring over a shard directory.

Builds a real extraction shard from the budget-statement DOCX fixture,
writes a couple of canned chunks, mocks the Kanon-2 embedding call (no
API/key needed), runs ``embed_shards``, and asserts the embeddings sidecar
+ checkpoint behaviour (skip-on-resume, no-checkpoint-on-failure).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from womblex.analyse.embed_stage import embed_shards
from womblex.config import EmbeddingConfig
from womblex.ingest.strategies_file import DocxExtractor
from womblex.store.checkpoint import CheckpointManager
from womblex.store.output import (
    embeddings_path_for,
    read_embeddings,
    read_manifest,
    write_chunks,
    write_results,
)

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "fixtures"
_BUDGET_DOCX = (
    _FIXTURES / "womblex-collection" / "_documents"
    / "foreign-affairs-and-trade-2025-26-portfolio-budget-statements.docx"
)


def _fake_embed(texts, *a, **k):
    return [[0.1, 0.2, 0.3] for _ in texts]


@pytest.fixture
def shard_with_chunks(tmp_path) -> Path:
    if not _BUDGET_DOCX.exists():
        pytest.skip(f"fixture not present: {_BUDGET_DOCX}")
    d = tmp_path / "documents"
    d.mkdir()
    base = d / "batch-0001.parquet"
    write_results([("budget", str(_BUDGET_DOCX), DocxExtractor().extract_path(_BUDGET_DOCX))],
                  base, collection_id="test")
    src = read_manifest(base).column("source_hash").to_pylist()[0]
    write_chunks([
        {"source_hash": src, "chunk_index": 0, "text": "first chunk text",
         "start_char": 0, "end_char": 16, "content_type": "narrative",
         "has_redaction": False, "page_start": 1, "page_end": 1},
        {"source_hash": src, "chunk_index": 1, "text": "second chunk text",
         "start_char": 16, "end_char": 33, "content_type": "narrative",
         "has_redaction": False, "page_start": 1, "page_end": 1},
    ], base)
    return d


class TestEmbedShards:
    def test_writes_embeddings_sidecar(self, shard_with_chunks, monkeypatch):
        monkeypatch.setattr("womblex.analyse.embed_stage.embed_texts", _fake_embed)
        result = embed_shards(shard_with_chunks, EmbeddingConfig(), client=object())
        assert result.chunks_embedded == 2
        assert embeddings_path_for(shard_with_chunks / "batch-0001.parquet").exists()
        t = read_embeddings(shard_with_chunks)
        assert t.num_rows == 2
        rows = t.to_pylist()
        assert all(r["dim"] == 3 and len(r["vector"]) == 3 for r in rows)
        assert all(r["model"] == "kanon-2-embedder" for r in rows)
        assert {r["chunk_index"] for r in rows} == {0, 1}

    def test_checkpoint_skips_on_resume(self, shard_with_chunks, monkeypatch, tmp_path):
        calls = {"n": 0}

        def _counting(texts, *a, **k):
            calls["n"] += 1
            return _fake_embed(texts)

        monkeypatch.setattr("womblex.analyse.embed_stage.embed_texts", _counting)
        ckpt = CheckpointManager(tmp_path / ".embed-ckpt", "t_embed")
        ckpt.load()
        embed_shards(shard_with_chunks, EmbeddingConfig(), client=object(), checkpoint_mgr=ckpt)
        assert calls["n"] == 1
        embed_shards(shard_with_chunks, EmbeddingConfig(), client=object(), checkpoint_mgr=ckpt)
        assert calls["n"] == 1  # batch skipped on resume

    def test_failure_not_checkpointed(self, shard_with_chunks, monkeypatch, tmp_path):
        def _boom(*a, **k):
            raise RuntimeError("Embedding failed: Connection error.")

        monkeypatch.setattr("womblex.analyse.embed_stage.embed_texts", _boom)
        ckpt = CheckpointManager(tmp_path / ".embed-ckpt", "t_embed")
        ckpt.load()
        embed_shards(shard_with_chunks, EmbeddingConfig(), client=object(), checkpoint_mgr=ckpt)
        assert "budget" not in ckpt.state.processed_ids
