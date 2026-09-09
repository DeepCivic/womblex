"""Shared CLI helpers and the Command record used by topic modules."""
from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path, PurePosixPath
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


class NestedCorpusError(ValueError):
    """Supported documents sit in subdirectories of the ingest location.

    A run ingests what is directly under the location it is given. Documents a
    level down would be ingested by some entry points and passed over by others,
    so every entry point refuses them instead — see ``select_supported``.

    Subclasses ``ValueError`` so the console's enqueue route reports it as bad
    input rather than a 500.
    """

    def __init__(self, location: str, nested: Mapping[str, int]) -> None:
        self.location = location
        self.nested = dict(nested)
        named = ", ".join(
            f"{d}/ ({n} document{'s' if n != 1 else ''})"
            for d, n in sorted(nested.items())[:3]
        )
        more = len(nested) - 3
        if more > 0:
            named += f", and {more} more subdirector{'ies' if more != 1 else 'y'}"
        super().__init__(
            f"{location} holds documents in subdirectories: {named}. "
            f"A run ingests only what is directly under the location it is given. "
            f"Point at the subdirectory that holds the documents, or flatten the corpus."
        )


def select_supported(relpaths: Iterable[str], *, location: str) -> list[str]:
    """The supported documents directly under *location*, from its full listing.

    *relpaths* is every file under the location, relative to it, from a
    recursive listing — the walk is what makes a nested corpus visible rather
    than silently skipped.

    Raises :class:`NestedCorpusError` if any supported document sits in a
    subdirectory, whether or not the top level holds documents too: a mixed
    layout ingests in part, which is the same defect as ingesting nothing.
    Subdirectories holding no supported document are ignored, so a corpus
    alongside ``.git`` or a notes directory still runs.

    The one enumeration rule, shared by the local CLI, the cloud enqueue and the
    console, so the three cannot drift apart again.
    """
    top: list[str] = []
    nested: dict[str, int] = {}
    for rel in relpaths:
        parts = PurePosixPath(rel).parts
        if not parts or PurePosixPath(rel).suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if len(parts) == 1:
            top.append(rel)
        else:
            nested[parts[0]] = nested.get(parts[0], 0) + 1
    if nested:
        raise NestedCorpusError(location, nested)
    return sorted(top)


def discover_files(input_root: Path, limit: int | None = None, skip: int = 0) -> list[Path]:
    """Discover supported documents directly under *input_root*.

    Raises :class:`NestedCorpusError` for a corpus with documents a level down.
    """
    listing = (
        p.relative_to(input_root).as_posix()
        for p in input_root.rglob("*")
        if p.is_file()
    )
    files = [input_root / name for name in select_supported(listing, location=str(input_root))]
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
