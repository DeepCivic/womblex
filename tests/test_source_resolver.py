"""A published row resolves back to its source document (P2).

Covers the four ways resolution can end — found, found-but-changed, missing,
and a hash basis that is not over file bytes — and the property that makes the
resolver trustworthy: it enumerates a corpus by the same rule a run ingests by.
"""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from womblex.cli._shared import NestedCorpusError, discover_files
from womblex.store.output import MANIFEST_SCHEMA, PARSER_VERSION, _source_hash
from womblex.store.source_provenance import qualify_root
from womblex.store.source_resolver import (
    HASH_BASIS_FILE_BYTES,
    HASH_BASIS_RECORD_ID_TEXT,
    HASH_MISMATCH,
    NOT_FOUND,
    RESOLVED,
    UNSUPPORTED_BASIS,
    SourceResolver,
)


def _write_docs(root: Path, contents: dict[str, bytes]) -> dict[str, str]:
    """Write documents under *root*; return relpath → sha256 of its bytes."""
    root.mkdir(parents=True, exist_ok=True)
    digests = {}
    for rel, data in contents.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        digests[rel] = _source_hash(str(p))
    return digests


def _manifest(root: Path, digests: dict[str, str], *, method: str = "native") -> pa.Table:
    rows = [
        {
            "source_hash": h, "collection_id": "test", "doc_id": Path(rel).stem,
            "ingest_root": qualify_root(root), "source_relpath": rel,
            "filename": Path(rel).name, "ext": Path(rel).suffix,
            "extraction_method": method, "elements_count": 1, "table_cells_count": 0,
            "form_fields_count": 0, "status": "completed", "error": "",
            "extracted_at_iso": "2026-01-01T00:00:00Z", "parser_version": PARSER_VERSION,
        }
        for rel, h in digests.items()
    ]
    return pa.Table.from_pylist(rows, schema=MANIFEST_SCHEMA)


@pytest.fixture
def corpus(tmp_path: Path):
    root = tmp_path / "corpus"
    digests = _write_docs(root, {"a.pdf": b"%PDF-1.4 alpha", "b.pdf": b"%PDF-1.4 beta"})
    return root, digests, _manifest(root, digests)


class TestResolution:
    def test_resolves_by_path_and_verifies_the_hash(self, corpus):
        root, digests, manifest = corpus
        resolver = SourceResolver(manifest, root)
        r = resolver.resolve(digests["a.pdf"])
        assert r.status == RESOLVED and r.ok
        assert r.path == root / "a.pdf"
        assert r.hash_basis == HASH_BASIS_FILE_BYTES
        assert "verified" in r.detail
        # The corpus is where the run left it, so the full scan is never paid for.
        assert resolver._index is None

    def test_unknown_hash_is_an_explicit_result_not_an_empty_return(self, corpus):
        root, _, manifest = corpus
        r = SourceResolver(manifest, root).resolve("0" * 64)
        assert r.status == NOT_FOUND
        assert r.path is None
        assert "no manifest row" in r.detail

    def test_missing_file_is_not_found_naming_the_root(self, corpus):
        root, digests, manifest = corpus
        (root / "a.pdf").unlink()
        r = SourceResolver(manifest, root).resolve(digests["a.pdf"])
        assert r.status == NOT_FOUND
        assert str(root) in r.detail
        assert r.doc_id == "a"

    def test_resaved_corpus_resolves_by_path_and_reports_the_mismatch(self, corpus):
        root, digests, manifest = corpus
        (root / "a.pdf").write_bytes(b"%PDF-1.4 alpha ")  # same content, different bytes
        r = SourceResolver(manifest, root).resolve(digests["a.pdf"])
        assert r.status == HASH_MISMATCH
        assert r.path == root / "a.pdf"
        assert "contents differ" in r.detail

    def test_document_moved_within_the_corpus_resolves_by_hash(self, corpus):
        root, digests, manifest = corpus
        (root / "a.pdf").rename(root / "renamed.pdf")
        r = SourceResolver(manifest, root).resolve(digests["a.pdf"])
        assert r.status == RESOLVED
        assert r.path == root / "renamed.pdf"
        assert "not where the manifest names it" in r.detail

    def test_one_hash_on_two_rows_resolves_once(self, tmp_path):
        # The same bytes ingested under two names: both rows name one file, so
        # resolve_all reports the document once rather than the rows twice.
        root = tmp_path / "corpus"
        digests = _write_docs(root, {"a.pdf": b"%PDF-1.4 same"})
        rows = _manifest(root, digests).to_pylist()
        rows.append({**rows[0], "doc_id": "copy", "source_relpath": "copy.pdf",
                     "filename": "copy.pdf"})
        manifest = pa.Table.from_pylist(rows, schema=MANIFEST_SCHEMA)
        results = SourceResolver(manifest, root).resolve_all()
        assert len(results) == 1
        assert results[0].ok and results[0].path == root / "a.pdf"

    def test_resolve_all_covers_every_row(self, corpus):
        root, digests, manifest = corpus
        results = SourceResolver(manifest, root).resolve_all()
        assert len(results) == len(digests)
        assert all(r.ok for r in results)


class TestMovedCorpus:
    def test_new_root_returns_the_same_documents(self, tmp_path):
        old = tmp_path / "old"
        digests = _write_docs(old, {"a.pdf": b"%PDF-1.4 alpha", "b.pdf": b"%PDF-1.4 beta"})
        manifest = _manifest(old, digests)
        new = tmp_path / "new"
        new.mkdir()
        for rel in digests:
            (new / rel).write_bytes((old / rel).read_bytes())

        before = {r.source_hash: r.source_relpath for r in SourceResolver(manifest, old).resolve_all()}
        after = SourceResolver(manifest, new).resolve_all()

        assert all(r.ok for r in after)
        assert {r.source_hash: r.source_relpath for r in after} == before
        assert {r.path for r in after} == {new / "a.pdf", new / "b.pdf"}


class TestHashBasis:
    def test_records_ingest_is_declined_naming_the_back_link(self, tmp_path):
        root = tmp_path / "corpus"
        root.mkdir()
        manifest = _manifest(root, {"rec-1": "deadbeef" * 8}, method="records")
        r = SourceResolver(manifest, root).resolve("deadbeef" * 8)
        assert r.status == UNSUPPORTED_BASIS
        assert r.hash_basis == HASH_BASIS_RECORD_ID_TEXT
        assert "provenance.parquet" in r.detail
        assert r.doc_id == "rec-1"


class TestIndexAgreesWithIngest:
    def test_index_is_built_from_the_run_s_own_enumeration(self, corpus):
        root, digests, manifest = corpus
        resolver = SourceResolver(manifest, root)
        assert set(resolver.index.values()) == set(discover_files(root))
        assert set(resolver.index) == set(digests.values())

    def test_unsupported_files_are_indexed_by_neither(self, corpus):
        root, _, manifest = corpus
        (root / "notes.txt").write_text("not a document")
        assert set(SourceResolver(manifest, root).index.values()) == set(discover_files(root))

    def test_nested_corpus_is_refused_by_index_and_by_ingest_alike(self, tmp_path):
        root = tmp_path / "corpus"
        digests = _write_docs(root, {"2026-08/a.pdf": b"%PDF-1.4 alpha"})
        manifest = _manifest(root, digests)
        with pytest.raises(NestedCorpusError):
            discover_files(root)
        resolver = SourceResolver(manifest, root)
        assert resolver.index == {}
        r = resolver.resolve(digests["2026-08/a.pdf"])
        assert r.status == NOT_FOUND
        assert "subdirectories" in r.detail
