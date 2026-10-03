"""Plugin-supplied model roots resolve offline like bundled models."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from womblex.utils import models


def _install(monkeypatch: pytest.MonkeyPatch, *values: object) -> None:
    eps = [SimpleNamespace(name=f"r{i}", load=lambda v=v: v) for i, v in enumerate(values)]
    monkeypatch.setattr("importlib.metadata.entry_points", lambda group: eps)
    models._plugin_roots.cache_clear()


@pytest.fixture(autouse=True)
def _reset():
    yield
    models._plugin_roots.cache_clear()


def test_plugin_root_resolves_a_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "my-model").mkdir()
    _install(monkeypatch, str(tmp_path))
    assert models.resolve_local_model_path("my-model", record=False) == tmp_path / "my-model"


def test_callable_and_iterable_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _install(monkeypatch, lambda: a, [b])
    roots = models.model_roots()
    assert a in roots and b in roots


def test_plugin_root_is_searched_last(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / "env"
    plug = tmp_path / "plug"
    for root in (env, plug):
        (root / "m").mkdir(parents=True)
    monkeypatch.setenv("WOMBLEX_MODELS_DIR", str(env))
    _install(monkeypatch, plug)
    assert models.resolve_local_model_path("m", record=False) == env / "m"


def test_broken_entry_point_is_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> None:
        raise RuntimeError("bad")

    _install(monkeypatch, boom, tmp_path)
    assert tmp_path in models.model_roots()
