"""Shared CLI helpers and the Command record used by topic modules."""
from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger("womblex")

SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".xlsx", ".xls", ".docx"}


class Command(NamedTuple):
    """One CLI subcommand: name, help, parser registration, handler."""

    name: str
    help: str
    register: Callable[[argparse.ArgumentParser], None]
    handler: Callable[[argparse.Namespace], int]


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
