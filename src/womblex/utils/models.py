"""Local model path resolution for offline / air-gapped deployment.

Resolution order:

1. ``WOMBLEX_MODELS_DIR`` environment variable (explicit override).
2. ``_models/`` bundled inside the installed package — this is the path used
   after ``pip install womblex`` and is what makes air-gapped use viable.
3. ``models/`` sibling of ``src/`` — backward compatibility for editable
   installs and the historical repo layout.

Supports both HuggingFace hub-style cache layouts and flat model directories.
"""

from __future__ import annotations

import os
from pathlib import Path


def find_models_dir() -> Path | None:
    """Locate the directory containing pre-downloaded model artefacts.

    Returns:
        Path to the models directory, or None if not found.
    """
    env_override = os.environ.get("WOMBLEX_MODELS_DIR")
    if env_override:
        p = Path(env_override)
        if p.is_dir():
            return p

    bundled = Path(__file__).resolve().parent.parent / "_models"
    if bundled.is_dir():
        return bundled

    current = Path(__file__).resolve().parent
    for _ in range(8):
        candidate = current / "models"
        if candidate.is_dir() and (current / "src").is_dir():
            return candidate
        current = current.parent

    return None


def resolve_local_model_path(model_name: str) -> str | Path:
    """Return a local path to *model_name* if pre-downloaded, else the name itself.

    Understands the HuggingFace hub cache layout::

        <models_dir>/<model_name>/refs/main           → contains snapshot hash
        <models_dir>/<model_name>/snapshots/<hash>/   → actual model files

    If a flat directory ``<models_dir>/<model_name>/`` exists without the hub
    layout, that directory is returned directly.

    For non-directory artefacts (e.g. ``yolov8n.pt``), pass the filename as
    *model_name* and the full file path is returned if it exists.

    Args:
        model_name: HuggingFace model identifier or bare filename.

    Returns:
        Local ``Path`` if found, otherwise the original *model_name* string
        (so callers can pass the result directly to library constructors).
    """
    models_dir = find_models_dir()
    if models_dir is None:
        return model_name

    local = models_dir / model_name

    if local.is_file():
        return local

    if local.is_dir():
        refs_main = local / "refs" / "main"
        if refs_main.is_file():
            snapshot_hash = refs_main.read_text().strip()
            snapshot_dir = local / "snapshots" / snapshot_hash
            if snapshot_dir.is_dir():
                return snapshot_dir
        return local

    return model_name
