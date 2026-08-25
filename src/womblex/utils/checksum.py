"""File checksum helpers shared by the standalone register ingests."""

from __future__ import annotations

import hashlib
from pathlib import Path


def md5_file(path: Path) -> str:
    """MD5 hex digest of a file, streamed in 64 KB chunks."""
    # Content addressing / provenance, not a security signature.
    # nosemgrep: python.lang.security.insecure-hash-algorithms-md5.insecure-hash-algorithm-md5
    h = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
