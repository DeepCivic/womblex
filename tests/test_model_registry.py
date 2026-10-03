"""Model registry: name resolution, plugin entry points, error contract."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from womblex.ingest.paddle_ocr import get_ocr_reader, is_llm_engine
from womblex.utils import model_registry as reg


@pytest.fixture
def clean_registry(monkeypatch: pytest.MonkeyPatch):
    """Isolate registrations from the process-wide registry."""
    for attr in ("_entries", "_aliases"):
        monkeypatch.setattr(
            reg, attr, {slot: dict(v) for slot, v in getattr(reg, attr).items()}
        )
    monkeypatch.setattr(reg, "_loaded", set(reg._loaded))


def test_builtin_names_and_aliases_keep_resolving() -> None:
    for alias, canonical in {
        "paddle": "paddleocr", "rapidocr": "paddleocr", "PaddleOCR": "paddleocr",
        "mistral": "mistral-ocr", "pixtral": "mistral-ocr", "bedrock": "mistral-ocr",
        "ollama-ocr": "ollama",
    }.items():
        assert reg.resolve(reg.SLOT_OCR, alias).name == canonical


def test_markdown_trait_marks_llm_engines() -> None:
    assert is_llm_engine("pixtral") and is_llm_engine("ollama")
    assert not is_llm_engine("rapidocr")


def test_unknown_name_lists_known_names() -> None:
    with pytest.raises(ValueError, match="paddleocr") as exc:
        get_ocr_reader(engine="nope")
    assert "mistral-ocr" in str(exc.value)


@pytest.mark.parametrize("spec", ["pkg.mod:make", "pkg.mod", "a/b"])
def test_import_paths_are_refused(spec: str) -> None:
    with pytest.raises(ValueError, match="import paths are not accepted"):
        reg.resolve(reg.SLOT_OCR, spec)


def test_plugin_entry_point_is_selectable_by_name(
    clean_registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    def factory(lang: str = "eng", **opts):
        return ("reader", lang, opts)

    factory.womblex_traits = {"markdown": True}  # type: ignore[attr-defined]
    ep = SimpleNamespace(
        name="my-ocr", load=lambda: factory, dist=SimpleNamespace(name="my-pkg")
    )
    monkeypatch.setattr(reg, "_loaded", set())
    monkeypatch.setattr(reg, "entry_points", lambda group: [ep] if group.endswith(".ocr") else [])

    assert "my-ocr" in reg.known_names(reg.SLOT_OCR)
    assert get_ocr_reader(engine="my-ocr", lang="fra", beam=3) == ("reader", "fra", {"beam": 3})
    assert is_llm_engine("my-ocr")
    assert reg.resolve(reg.SLOT_OCR, "my-ocr").source == "my-pkg"


def test_plugin_cannot_shadow_a_builtin(
    clean_registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    ep = SimpleNamespace(name="pixtral", load=lambda: object, dist=None)
    monkeypatch.setattr(reg, "_loaded", set())
    monkeypatch.setattr(reg, "entry_points", lambda group: [ep] if group.endswith(".ocr") else [])
    assert reg.resolve(reg.SLOT_OCR, "pixtral").name == "mistral-ocr"
