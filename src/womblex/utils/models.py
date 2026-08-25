"""Local model path resolution for offline / air-gapped deployment.

Roots are searched **per artefact**, in this order:

1. ``WOMBLEX_MODELS_DIR`` environment variable (explicit override).
2. ``_models/`` bundled inside the installed package — this is the path used
   after ``pip install womblex`` and is what makes air-gapped use viable.
3. ``models/`` sibling of ``src/`` — backward compatibility for editable
   installs and the historical repo layout.

Per artefact, not "first root wins", because the roots hold *different*
artefacts. A container image mounts the large ones (layout, embedding, OCR)
at ``WOMBLEX_MODELS_DIR=/app/models`` while the small ones (``en_AU``,
``kanon-2-tokenizer``) ship inside the wheel — so a single-root resolver makes
the override **shadow** the bundled artefacts rather than supplement them, and
``resolve_local_model_path("en_AU")`` starts returning the bare string.

Supports both HuggingFace hub-style cache layouts and flat model directories.
"""

from __future__ import annotations

import os
from pathlib import Path


def _repo_models_dir() -> Path | None:
    """``models/`` beside ``src/`` — the editable-install / repo layout."""
    current = Path(__file__).resolve().parent
    for _ in range(8):
        candidate = current / "models"
        if candidate.is_dir() and (current / "src").is_dir():
            return candidate
        current = current.parent
    return None


def model_roots() -> tuple[Path, ...]:
    """Every existing models root, in resolution order.

    Duplicates are dropped, so pointing ``WOMBLEX_MODELS_DIR`` at the bundled
    directory does not search it twice.
    """
    roots: list[Path] = []

    def add(path: Path | None) -> None:
        if path is not None and path.is_dir() and path not in roots:
            roots.append(path)

    env_override = os.environ.get("WOMBLEX_MODELS_DIR")
    add(Path(env_override) if env_override else None)
    add(Path(__file__).resolve().parent.parent / "_models")
    add(_repo_models_dir())
    return tuple(roots)


def find_models_dir() -> Path | None:
    """The highest-priority existing models root, or None if there is none.

    Kept for callers that want *a* root rather than a resolved artefact. Do
    not use it to build an artefact path — the artefact may live under a
    lower-priority root; call :func:`resolve_local_model_path` instead.
    """
    roots = model_roots()
    return roots[0] if roots else None


def _resolve_under(root: Path, model_name: str) -> Path | None:
    """*model_name* under *root*, or None if this root does not hold it."""
    local = root / model_name

    if local.is_file():
        return local

    if local.is_dir():
        refs_main = local / "refs" / "main"
        if refs_main.is_file():
            snapshot_dir = local / "snapshots" / refs_main.read_text().strip()
            if snapshot_dir.is_dir():
                return snapshot_dir
        return local

    return None


def resolve_local_model_path(model_name: str) -> str | Path:
    """Return a local path to *model_name* if pre-downloaded, else the name itself.

    Every root from :func:`model_roots` is searched, in order, and the first
    that actually holds *model_name* wins.

    Understands the HuggingFace hub cache layout::

        <root>/<model_name>/refs/main           → contains snapshot hash
        <root>/<model_name>/snapshots/<hash>/   → actual model files

    If a flat directory ``<root>/<model_name>/`` exists without the hub
    layout, that directory is returned directly.

    For non-directory artefacts (e.g. ``yolov8n.pt``), pass the filename as
    *model_name* and the full file path is returned if it exists.

    Args:
        model_name: HuggingFace model identifier or bare filename.

    Returns:
        Local ``Path`` if found, otherwise the original *model_name* string
        (so callers can pass the result directly to library constructors).
    """
    for root in model_roots():
        resolved = _resolve_under(root, model_name)
        if resolved is not None:
            return resolved
    return model_name


__all__ = ["find_models_dir", "model_roots", "resolve_local_model_path"]
