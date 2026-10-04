"""Pre-run model check: levels, scopes, failure reporting and what is remembered."""

from __future__ import annotations

import importlib

import pytest

from womblex.config import WomblexConfig
from womblex.utils import model_registry as reg
from womblex.utils.model_check import (
    _BUILTIN_MODULES,
    SCOPE_CHUNK,
    SCOPE_EXTRACT,
    SCOPE_PII,
    SCOPE_SPELLFIX,
    CheckLevel,
    ModelCheckError,
    check_models,
    configured_models,
)

# The built-ins register at import; do it before the fixture copies the registry,
# or they would be registered into the copy and lost when it is restored.
for _module in _BUILTIN_MODULES:
    importlib.import_module(_module)

_PATHS = {"input_root": "/tmp/i", "output_root": "/tmp/o", "checkpoint_dir": "/tmp/c"}


def _config(**sections: dict) -> WomblexConfig:
    return WomblexConfig(dataset={"name": "t"}, paths=_PATHS, **sections)


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch):
    """Isolate registrations and what the process remembers having checked."""
    for attr in ("_entries", "_aliases"):
        monkeypatch.setattr(
            reg, attr, {slot: dict(v) for slot, v in getattr(reg, attr).items()}
        )
    yield


class _Dictionary:
    def __init__(self, knows: bool = True) -> None:
        self._knows = knows

    def lookup(self, word: str) -> bool:
        return self._knows


def _dictionary_config(name: str, **options: object) -> WomblexConfig:
    return _config(spellfix={"enabled": True, "dict_name": name, "dict_options": options})


class TestWhatTheConfigNames:
    def test_default_config_names_the_extraction_and_chunk_models(self) -> None:
        uses = {(u.slot, u.name): u.scopes for u in configured_models(_config())}
        assert uses[("ocr", "paddleocr")] == (SCOPE_EXTRACT,)
        # OCR and redaction share one analyser: listed once, loaded once.
        assert uses[("layout", "pp-doclayout-m")] == (SCOPE_EXTRACT,)
        assert uses[("tokenizer", "isaacus/kanon-2-tokenizer")] == (SCOPE_CHUNK,)

    def test_scopes_narrow_the_list(self) -> None:
        names = {u.slot for u in configured_models(_config(), scopes=(SCOPE_CHUNK,))}
        assert names == {"tokenizer"}

    def test_stages_that_are_off_are_not_checked(self) -> None:
        slots = {u.slot for u in configured_models(_config())}
        assert "spellfix-dictionary" not in slots
        assert "pii-context" not in slots

    def test_the_pii_model_is_named_only_when_the_backstop_uses_it(self) -> None:
        graph_only = _config(pii={"enabled": True})
        backstop = _config(pii={"enabled": True, "use_regex_backstop": True})
        assert not [u for u in configured_models(graph_only) if u.scopes == (SCOPE_PII,)]
        assert [u.name for u in configured_models(backstop, scopes=(SCOPE_PII,))] == [
            "all-minilm-l6-v2",
        ]

    def test_a_markdown_engine_has_no_ocr_layout_model(self) -> None:
        cfg = _config(
            extraction={"ocr": {"engine": "ollama"}},
            redaction={"enabled": False},
        )
        assert "layout" not in {u.slot for u in configured_models(cfg)}


class TestLevels:
    def test_off_checks_nothing_and_says_so(self) -> None:
        result = check_models(_config(), "off")
        assert result.checks == () and result.ok

    def test_load_builds_the_model_through_its_factory(self) -> None:
        reg.register(
            reg.SLOT_SPELLFIX_DICTIONARY, "tiny", lambda **_: _Dictionary(), source="tiny-dist",
        )
        result = check_models(_dictionary_config("tiny"), "load", scopes=(SCOPE_SPELLFIX,))
        (check,) = result.checks
        assert (check.ok, check.name, check.source, check.level) == (
            True, "tiny", "tiny-dist", CheckLevel.LOAD,
        )

    def test_load_reports_a_factory_that_raises(self) -> None:
        def broken(**_: object) -> object:
            raise FileNotFoundError("weights missing")

        reg.register(reg.SLOT_SPELLFIX_DICTIONARY, "broken", broken, source="x")
        result = check_models(_dictionary_config("broken"), "load", scopes=(SCOPE_SPELLFIX,))
        (bad,) = result.failures
        assert "FileNotFoundError: weights missing" in bad.reason
        with pytest.raises(ModelCheckError, match="weights missing"):
            result.raise_for_failures()

    def test_smoke_catches_what_load_does_not(self) -> None:
        reg.register(
            reg.SLOT_SPELLFIX_DICTIONARY, "empty", lambda **_: _Dictionary(knows=False), source="x",
        )
        cfg = _dictionary_config("empty")
        assert check_models(cfg, "load", scopes=(SCOPE_SPELLFIX,)).failures == ()
        (bad,) = check_models(cfg, "smoke", scopes=(SCOPE_SPELLFIX,)).failures
        assert "the" in bad.reason

    def test_an_unknown_name_fails_and_lists_the_known_names(self) -> None:
        result = check_models(_dictionary_config("nope"), "load", scopes=(SCOPE_SPELLFIX,))
        (bad,) = result.failures
        assert "en_au" in bad.reason and "nope" in bad.reason

    def test_the_level_defaults_to_the_config(self) -> None:
        cfg = _config(processing={"models_check": "off"})
        assert check_models(cfg).level is CheckLevel.OFF


class TestVariant:
    def test_a_model_names_the_variant_it_resolved(self) -> None:
        class Reader:
            model_variant = "v5"

            def read_page(self, img: object) -> object:
                raise AssertionError("load must not infer")

        reg.register(reg.SLOT_OCR, "variant-ocr", lambda **_: Reader(), source="d")
        cfg = _config(extraction={"ocr": {"engine": "variant-ocr"}}, redaction={"enabled": False})
        result = check_models(cfg, "load", scopes=(SCOPE_EXTRACT,))
        assert [(c.slot, c.variant) for c in result.checks if c.slot == "ocr"] == [("ocr", "v5")]

    def test_paddleocr_reports_which_model_set_loaded(self) -> None:
        pytest.importorskip("rapidocr_onnxruntime")
        cfg = _config(redaction={"enabled": False})
        result = check_models(cfg, "load", scopes=(SCOPE_EXTRACT,))
        ocr = next(c for c in result.checks if c.slot == "ocr")
        assert ocr.ok, ocr.reason
        assert ocr.variant in {"paddleocr-v5", "rapidocr-bundled-v4"}

    def test_paddleocr_smoke_reads_the_probe_image(self) -> None:
        pytest.importorskip("rapidocr_onnxruntime")
        cfg = _config(redaction={"enabled": False})
        result = check_models(cfg, "smoke", scopes=(SCOPE_EXTRACT,))
        assert [c.reason for c in result.failures] == []
