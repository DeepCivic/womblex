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


@pytest.mark.parametrize("name", ["pixtral", "paddleocr"])
def test_plugin_cannot_shadow_a_builtin(
    clean_registry, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    ep = SimpleNamespace(name=name, load=lambda: object, dist=None)
    monkeypatch.setattr(reg, "_loaded", set())
    monkeypatch.setattr(reg, "entry_points", lambda group: [ep] if group.endswith(".ocr") else [])
    assert reg.resolve(reg.SLOT_OCR, name).source == "builtin"


# --- layout slot -----------------------------------------------------------

from womblex.ingest.interfaces.protocols import (
    LAYOUT_BLOCK_TYPES,
    LayoutRegionResult,
    check_layout_regions,
)
from womblex.ingest.layout_onnx import LABEL_MAP
from womblex.ingest.paddle_ocr import get_layout_analyzer


def _region(block_type: str = "paragraph", y0: float = 0.0, conf: float = 0.9):
    return LayoutRegionResult((0, y0, 10, y0 + 5), "x", block_type, conf)


def test_layout_default_name_and_alias_resolve() -> None:
    assert reg.resolve(reg.SLOT_LAYOUT, "PP-DocLayout").name == "pp-doclayout-m"


def test_layout_unknown_name_lists_known_names() -> None:
    with pytest.raises(ValueError, match="pp-doclayout-m"):
        get_layout_analyzer("nope")


def test_builtin_label_map_conforms_to_vocabulary() -> None:
    assert set(LABEL_MAP.values()) <= LAYOUT_BLOCK_TYPES


def test_conformance_accepts_a_good_result_and_empty() -> None:
    check_layout_regions([])
    check_layout_regions([_region("table", 0), _region("figure", 20)])


@pytest.mark.parametrize(
    "regions",
    [
        [_region("textbox")],
        [LayoutRegionResult((5, 5, 1, 1), "x", "table", 0.5)],
        [_region(conf=1.5)],
        [_region(y0=50), _region(y0=10)],
    ],
)
def test_conformance_rejects_bad_results(regions) -> None:
    with pytest.raises(ValueError):
        check_layout_regions(regions)


def test_plugin_layout_model_is_selectable_and_receives_options(
    clean_registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Fake:
        def __init__(self, **opts):
            self.opts = opts

    ep = SimpleNamespace(name="my-layout", load=lambda: Fake, dist=SimpleNamespace(name="p"))
    monkeypatch.setattr(reg, "_loaded", set())
    monkeypatch.setattr(reg, "entry_points", lambda group: [ep] if group.endswith(".layout") else [])
    a = get_layout_analyzer("my-layout", size=3)
    assert a.opts == {"size": 3}  # type: ignore[attr-defined]
    assert get_layout_analyzer("my-layout", size=3) is a


# --- pii-context slot ------------------------------------------------------

import numpy as np

from womblex.pii.cleaner import PIICleaner


def test_pii_context_default_and_alias_resolve() -> None:
    entry = reg.resolve(reg.SLOT_PII_CONTEXT, "Sentence-Transformers/all-MiniLM-L6-v2")
    assert entry.name == "all-minilm-l6-v2"


def test_pii_context_unknown_name_fails_at_construction() -> None:
    with pytest.raises(ValueError, match="all-minilm-l6-v2"):
        PIICleaner(model="nope")


def test_plugin_context_model_scores_candidates(
    clean_registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict = {}

    class Encoder:
        def __init__(self, **opts):
            seen.update(opts)

        def encode(self, texts):
            # every text maps to the same direction: cosine similarity 1.0
            return np.ones((len(texts), 4))

    ep = SimpleNamespace(name="my-ctx", load=lambda: Encoder, dist=SimpleNamespace(name="p"))
    monkeypatch.setattr(reg, "_loaded", set())
    monkeypatch.setattr(
        reg, "entry_points", lambda group: [ep] if group.endswith(".pii-context") else []
    )
    cleaner = PIICleaner(model="my-ctx", model_options={"dim": 4})
    import re

    text = "Signed by Janine Fairburn today."
    match = re.search("Janine Fairburn", text)
    assert cleaner._score_context_batch(text, [match]) == pytest.approx([1.0])  # type: ignore[list-item]
    assert seen == {"dim": 4}


# --- redaction layout filter -----------------------------------------------

from pathlib import Path

import fitz

from womblex.config import RedactionConfig
from womblex.redact.stage import _layout_exclude_rects, build_detector, detect_redactions


def test_redaction_unknown_layout_name_raises(tmp_path: Path) -> None:
    pdf = tmp_path / "blank.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(pdf)
    with pytest.raises(ValueError, match="pp-doclayout-m"):
        detect_redactions(pdf, 1, build_detector(RedactionConfig()), layout_model="nope")


def test_redaction_filter_drops_non_conforming_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analyzer = SimpleNamespace(analyze=lambda img: [_region("textbox")])
    monkeypatch.setattr(
        "womblex.ingest.paddle_ocr.get_layout_analyzer", lambda *a, **k: analyzer
    )
    assert _layout_exclude_rects(np.zeros((10, 10, 3), dtype=np.uint8)) is None
