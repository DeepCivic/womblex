"""Shared CLI helpers and the Command record used by topic modules."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable, NamedTuple

logger = logging.getLogger("womblex")

SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".xlsx", ".xls", ".docx"}


class Command(NamedTuple):
    """One CLI subcommand: name, help, parser registration, handler."""

    name: str
    help: str
    register: Callable[[argparse.ArgumentParser], None]
    handler: Callable[[argparse.Namespace], int]


def make_isaacus_client():  # type: ignore[no-untyped-def]
    """Construct an Isaacus client, stripping whitespace from the API key.

    The Isaacus SDK reads ``ISAACUS_API_KEY`` from the environment as-is,
    so a stray trailing newline (common when a key is pasted into a
    ``.env`` file on Windows) reaches httpx as an illegal header value and
    fails with a cryptic ``LocalProtocolError``. Stripping the key here
    makes the bootstrap robust to that. Falls back to the SDK default
    (env-var lookup) when the variable is unset.
    """
    import os

    import isaacus

    key = os.environ.get("ISAACUS_API_KEY")
    if key is not None and key.strip() != key:
        return isaacus.Isaacus(api_key=key.strip())
    return isaacus.Isaacus()


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def discover_files(input_root: Path, limit: int | None = None, skip: int = 0) -> list[Path]:
    """Discover supported documents in *input_root*."""
    files = sorted(
        (f for f in input_root.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS),
        key=lambda p: p.name,
    )
    if skip:
        files = files[skip:]
    if limit:
        files = files[:limit]
    return files


def format_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    return f"{hours}h {mins}m"
