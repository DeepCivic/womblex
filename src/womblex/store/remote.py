"""Object-storage staging for distributed runs (S3 / MinIO / GCS / local).

The cloud worker model is **stage-in / stage-out**: a worker pulls a job's
input documents from object storage to a local scratch dir, runs the ordinary
``Path``-based pipeline, then pushes the resulting ``batch-NNNN.*.parquet``
shards back. This keeps every extraction/stage module untouched — they still
see plain local paths — and confines all remote-storage knowledge to this one
adapter, exactly the way ``store/`` sidecar modules are self-contained.

fsspec gives one API across ``s3://`` (MinIO is S3-compatible), ``gs://``,
``az://`` and the local filesystem, so the air-gapped / CPU-first default is
free: a ``file://`` or bare local path uses the same code path with no S3
dependency exercised.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import cast

logger = logging.getLogger(__name__)

_REMOTE_PROTOCOLS = ("s3://", "gs://", "gcs://", "az://", "abfs://")


def is_remote_uri(uri: str) -> bool:
    """True when *uri* names an object store (not a local/``file://`` path)."""
    return uri.startswith(_REMOTE_PROTOCOLS)


def storage_options_from_env(uri: str) -> dict:
    """Build fsspec storage options for *uri* from standard env vars.

    Only ``s3://`` gets explicit options: the AWS conventions s3fs already
    reads (``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``, ``AWS_REGION``)
    plus an endpoint override (``WOMBLEX_S3_ENDPOINT`` or
    ``AWS_ENDPOINT_URL``) so MinIO and other S3-compatible stores work
    without code changes. Every other backend — ``gs://``, ``az://``, local —
    returns an empty dict and authenticates via its own native mechanism
    (gcsfs/adlfs ambient credentials); these kwargs are s3fs-shaped and
    would misconfigure them.
    """
    if not uri.startswith("s3://"):
        return {}
    endpoint = os.environ.get("WOMBLEX_S3_ENDPOINT") or os.environ.get("AWS_ENDPOINT_URL")
    opts: dict = {}
    key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if key and secret:
        opts["key"] = key
        opts["secret"] = secret
    if endpoint:
        opts["client_kwargs"] = {"endpoint_url": endpoint}
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if region:
        opts.setdefault("client_kwargs", {})["region_name"] = region
    return opts


def _require_fsspec():  # type: ignore[no-untyped-def]
    try:
        import fsspec
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "Object-storage staging requires the 'cloud' extra. "
            "Install with: pip install womblex[cloud]"
        ) from e
    import fsspec

    return fsspec


@dataclass
class RemoteStore:
    """Thin fsspec wrapper rooted at a base URI.

    All paths passed to methods are *relative* to the store root; the store
    joins them. Build one with :meth:`from_uri`.
    """

    fs: object  # fsspec.AbstractFileSystem
    root: str

    @classmethod
    def from_uri(cls, uri: str, *, storage_options: dict | None = None) -> RemoteStore:
        """Open a store at *uri* (e.g. ``s3://bucket/runs`` or ``/data/runs``)."""
        fsspec = _require_fsspec()
        opts = storage_options if storage_options is not None else storage_options_from_env(uri)
        fs, root = fsspec.core.url_to_fs(uri, **opts)
        return cls(fs=fs, root=root.rstrip("/"))

    def _full(self, rel: str) -> str:
        rel = rel.strip("/")
        return f"{self.root}/{rel}" if rel else self.root

    def exists(self, rel: str) -> bool:
        return bool(self.fs.exists(self._full(rel)))  # type: ignore[attr-defined]

    def read_text(self, rel: str, *, encoding: str = "utf-8") -> str:
        """Read a small text object in place (no staging) — e.g. a saved preset."""
        with self.fs.open(self._full(rel), "r", encoding=encoding) as handle:  # type: ignore[attr-defined]
            return cast(str, handle.read())

    def delete(self, rel: str) -> None:
        """Remove one object. Assumes it exists; callers check first if that matters."""
        self.fs.rm_file(self._full(rel))  # type: ignore[attr-defined]

    def list_files(self, rel: str, pattern: str = "*") -> list[str]:
        """List paths under *rel* matching *pattern*, returned store-relative."""
        full = self._full(rel)
        matches: list[str] = self.fs.glob(f"{full}/{pattern}")  # type: ignore[attr-defined]
        prefix = self.root + "/"
        return [m.removeprefix(prefix) for m in matches]

    def list_dirs(self, rel: str) -> list[str]:
        """List immediate child directory names under *rel* (name only, not full path).

        Object stores have no real directories, but fsspec surfaces the
        common-prefix convention (keys with more path segments beneath *rel*)
        as pseudo-directory entries. That is what lets a caller enumerate
        ``runs/<run_id>/`` without knowing the run ids ahead of time.
        """
        full = self._full(rel)
        if not self.fs.exists(full):  # type: ignore[attr-defined]
            return []
        entries: list[dict] = self.fs.ls(full, detail=True)  # type: ignore[attr-defined]
        return sorted(
            e["name"].rstrip("/").rsplit("/", 1)[-1]
            for e in entries
            if e.get("type") == "directory"
        )

    def download_file(self, rel: str, local_path: Path) -> Path:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.fs.get_file(self._full(rel), str(local_path))  # type: ignore[attr-defined]
        return local_path

    def upload_file(self, local_path: Path, rel: str) -> str:
        full = self._full(rel)
        parent = full.rsplit("/", 1)[0]
        self.fs.makedirs(parent, exist_ok=True)  # type: ignore[attr-defined]
        self.fs.put_file(str(local_path), full)  # type: ignore[attr-defined]
        return rel

    def download_to_dir(self, rels: list[str], local_dir: Path) -> list[Path]:
        """Fetch each store-relative key in *rels* into *local_dir* (flat)."""
        local_dir.mkdir(parents=True, exist_ok=True)
        out: list[Path] = []
        for rel in rels:
            name = rel.rsplit("/", 1)[-1]
            out.append(self.download_file(rel, local_dir / name))
        logger.info("Staged %d input file(s) -> %s", len(out), local_dir)
        return out

    def upload_glob(self, local_dir: Path, glob: str, remote_rel_dir: str) -> list[str]:
        """Push every file in *local_dir* matching *glob* under *remote_rel_dir*."""
        uploaded: list[str] = []
        for p in sorted(local_dir.glob(glob)):
            if p.is_file():
                uploaded.append(self.upload_file(p, f"{remote_rel_dir.strip('/')}/{p.name}"))
        logger.info("Published %d shard file(s) -> %s/%s", len(uploaded), self.root, remote_rel_dir)
        return uploaded


__all__ = [
    "RemoteStore",
    "is_remote_uri",
    "storage_options_from_env",
]
