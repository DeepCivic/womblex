"""Local-model resolution: per-artefact search across every models root.

The regression this pins is the one the container image hit and the suite
missed: ``WOMBLEX_MODELS_DIR`` points at a root holding only the large
artefacts, so a single-root resolver stopped finding the wheel-bundled
``en_AU`` dictionary and spellfix died inside spylls on the relative path
``en_AU/index.aff``. The override must *supplement* the bundled root, not
shadow it — and an unresolved dictionary must fail with an actionable message
rather than a FileNotFoundError several frames away.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import womblex
from womblex.utils.models import find_models_dir, model_roots, resolve_local_model_path

#: The wheel-bundled root, taken from the installed package (not the repo
#: layout) so the test pins what a `pip install womblex` actually resolves.
BUNDLED = Path(womblex.__file__).resolve().parent / "_models"


@pytest.fixture
def only_env_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A WOMBLEX_MODELS_DIR holding the large artefacts and nothing else."""
    root = tmp_path / "app-models"
    (root / "all-MiniLM-L6-v2").mkdir(parents=True)
    (root / "yolov8n.pt").write_bytes(b"weights")
    monkeypatch.setenv("WOMBLEX_MODELS_DIR", str(root))
    return root


def test_env_root_does_not_shadow_bundled_dictionary(only_env_root: Path):
    """en_AU still resolves when the override root does not carry it."""
    resolved = resolve_local_model_path("en_AU")
    assert isinstance(resolved, Path), "en_AU fell back to the bare string"
    assert resolved == BUNDLED / "en_AU"
    assert (resolved / "index.aff").is_file()


def test_env_root_still_wins_for_what_it_holds(only_env_root: Path):
    """The override keeps priority for the artefacts it does carry."""
    assert resolve_local_model_path("all-MiniLM-L6-v2") == only_env_root / "all-MiniLM-L6-v2"
    assert resolve_local_model_path("yolov8n.pt") == only_env_root / "yolov8n.pt"


def test_roots_are_ordered_and_deduplicated(only_env_root: Path):
    roots = model_roots()
    assert roots[0] == only_env_root
    assert BUNDLED in roots
    assert len(roots) == len(set(roots))
    assert find_models_dir() == only_env_root


def test_bundled_dictionary_loads_under_env_override(only_env_root: Path):
    """The end-to-end symptom: spellfix's dictionary loader must not raise."""
    pytest.importorskip("spylls")
    from womblex.process import spellfix

    spellfix._dictionary.cache_clear()
    try:
        assert spellfix._dictionary("en_AU").lookup("child")
    finally:
        spellfix._dictionary.cache_clear()


def test_missing_dictionary_raises_actionable_error(only_env_root: Path):
    """A bare-string return must not reach spylls as a relative path."""
    pytest.importorskip("spylls")
    from womblex.process import spellfix

    spellfix._dictionary.cache_clear()
    try:
        with pytest.raises(FileNotFoundError, match="no models root holds"):
            spellfix._dictionary("en_ZZ")
    finally:
        spellfix._dictionary.cache_clear()


def test_hub_snapshot_layout_still_resolves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """refs/main → snapshots/<hash>/ is unchanged by the multi-root search."""
    root = tmp_path / "models"
    model = root / "some-model"
    (model / "refs").mkdir(parents=True)
    (model / "refs" / "main").write_text("deadbeef\n")
    snapshot = model / "snapshots" / "deadbeef"
    snapshot.mkdir(parents=True)
    monkeypatch.setenv("WOMBLEX_MODELS_DIR", str(root))

    assert resolve_local_model_path("some-model") == snapshot


def test_dangling_refs_main_falls_back_to_the_flat_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """A refs/main naming a snapshot that is not there returns the model dir."""
    root = tmp_path / "models"
    model = root / "some-model"
    (model / "refs").mkdir(parents=True)
    (model / "refs" / "main").write_text("missing")
    monkeypatch.setenv("WOMBLEX_MODELS_DIR", str(root))

    assert resolve_local_model_path("some-model") == model


def test_unknown_artefact_echoes_the_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WOMBLEX_MODELS_DIR", str(tmp_path))
    assert resolve_local_model_path("org/not-vendored") == "org/not-vendored"


def test_nonexistent_env_root_is_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A typo'd override must not remove the bundled root from the search."""
    monkeypatch.setenv("WOMBLEX_MODELS_DIR", str(tmp_path / "nope"))
    assert resolve_local_model_path("en_AU") == BUNDLED / "en_AU"
