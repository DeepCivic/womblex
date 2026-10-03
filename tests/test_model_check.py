"""Pre-run model check: levels, scopes, failure reporting and what is remembered."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from womblex.cloud.queue import Job
from womblex.config import WomblexConfig
from womblex.store.run_stamp import (
    MODEL_CHECK_KEY,
    RunStamp,
    read_footer_model_check,
)
from womblex.utils import model_registry as reg
from womblex.utils.model_check import (
    _BUILTIN_MODULES,
    SCOPE_CHUNK,
    SCOPE_EXTRACT,
    SCOPE_PII,
    SCOPE_SPELLFIX,
    CheckLevel,
    ModelCheckError,
    ModelCheckResult,
    SlotCheck,
    check_models,
    configured_models,
    footer_payload,
    reset_model_check,
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
    reset_model_check()
    yield
    reset_model_check()


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
        assert footer_payload() == {"checks": [], "level": "off"}

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


class TestWhatIsRemembered:
    def test_off_is_remembered_and_says_so(self) -> None:
        check_models(_config(), "off")
        assert footer_payload() == {"checks": [], "level": "off"}

    def test_nothing_asked_means_nothing_to_write(self) -> None:
        assert footer_payload() is None

    def test_the_footer_carries_the_checks_and_reads_back(self) -> None:
        reg.register(
            reg.SLOT_SPELLFIX_DICTIONARY, "tiny", lambda **_: _Dictionary(), source="tiny-dist",
        )
        check_models(_dictionary_config("tiny"), "load", scopes=(SCOPE_SPELLFIX,))
        meta = RunStamp.declare("run-1", _config(), stage="extract").footer_metadata()
        payload = read_footer_model_check(meta)
        assert payload is not None
        (entry,) = payload["checks"]
        assert (entry["slot"], entry["name"], entry["status"], entry["level"]) == (
            "spellfix-dictionary", "tiny", "ok", "load",
        )
        assert MODEL_CHECK_KEY.encode() in meta

    def test_an_unstamped_process_writes_no_key(self) -> None:
        meta = RunStamp.declare("run-1", _config(), stage="extract").footer_metadata()
        assert MODEL_CHECK_KEY.encode() not in meta

    def test_a_malformed_footer_reads_as_absent(self) -> None:
        assert read_footer_model_check({MODEL_CHECK_KEY.encode(): b"{not json"}) is None


def _failed(slot: str, scope: str) -> ModelCheckResult:
    check = SlotCheck(slot, "gone", (scope,), CheckLevel.LOAD, ok=False, reason="not found")
    return ModelCheckResult(CheckLevel.LOAD, (check,))


class TestRunStopsBeforeTheFirstDocument:
    def test_a_failed_check_writes_nothing(self, tmp_path: Path) -> None:
        import argparse

        from womblex.cli.pipeline import cmd_run

        inbox = tmp_path / "in"
        inbox.mkdir()
        (inbox / "a.csv").write_text("a,b\n1,2\n")
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            f"dataset:\n  name: t\npaths:\n  input_root: {inbox}\n"
            f"  output_root: {tmp_path / 'out'}\n  checkpoint_dir: {tmp_path / 'ckpt'}\n"
            "extraction:\n  ocr:\n    engine: no-such-engine\n"
            "redaction:\n  enabled: false\n"
        )
        args = argparse.Namespace(
            config=cfg, resume=False, limit=None, skip=0, batch_size=None, run_id="t",
            models_check=None,
        )
        assert cmd_run(args) == 1
        assert not (tmp_path / "out").exists()

    def test_the_cli_override_beats_the_config(self, tmp_path: Path) -> None:
        import argparse

        from womblex.cli._shared import apply_models_check

        cfg = _config()
        apply_models_check(cfg, argparse.Namespace(models_check="smoke"))
        assert cfg.processing.models_check == "smoke"
        apply_models_check(cfg, argparse.Namespace(models_check=None))
        assert cfg.processing.models_check == "smoke"



class TestWorkerRefusal:
    def test_a_job_is_refused_only_for_the_models_it_needs(self) -> None:
        from womblex.cloud.worker import _model_refusal

        batch = Job(1, "r", 1, ["a.pdf"], "p", 1)
        chunk = Job(2, "r", 0, [], "p", 1, kind="stage", stage="chunk")
        money = Job(3, "r", 0, [], "p", 1, kind="stage", stage="money")

        extraction_down = _failed("ocr", SCOPE_EXTRACT)
        assert "not found" in (_model_refusal(batch, extraction_down) or "")
        assert _model_refusal(chunk, extraction_down) is None

        tokenizer_down = _failed("tokenizer", SCOPE_CHUNK)
        assert _model_refusal(batch, tokenizer_down) is None
        assert "tokenizer model" in (_model_refusal(chunk, tokenizer_down) or "")
        assert _model_refusal(money, tokenizer_down) is None

    def test_the_loop_releases_a_refused_job_without_running_it(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        from womblex.cloud import worker

        released: list[tuple[int, str]] = []

        class FakeQueue:
            def __init__(self, dsn: str) -> None:
                self._jobs = [Job(7, "r", 1, ["a.pdf"], "runs/r/documents", 1)]

            def claim(self, worker_id: str, run_id: str | None) -> Job | None:
                return self._jobs.pop() if self._jobs else None

            def release(self, job_id: int, error: str) -> None:
                released.append((job_id, error))

            def close(self) -> None:
                pass

        def must_not_run(*_: object, **__: object) -> None:
            raise AssertionError("a refused job must not run")

        monkeypatch.setattr(worker, "JobQueue", FakeQueue)
        monkeypatch.setattr(worker, "check_models", lambda config: _failed("ocr", SCOPE_EXTRACT))
        monkeypatch.setattr(worker, "_process_job", must_not_run)
        store = tmp_path / "store"
        store.mkdir()

        completed = worker.run_worker(
            "dsn", str(store), _config(), run_id="r", once=True, poll_interval=0,
        )

        assert completed == 0
        assert len(released) == 1 and released[0][0] == 7
        assert "model check failed on this worker" in released[0][1]


class TestStagePreflight:
    def test_a_stage_whose_model_fails_is_refused_before_it_runs(self) -> None:
        from womblex.cloud.stage_contracts import STAGE_CONTRACTS
        from womblex.cloud.stage_runner import StagePreconditionError, prepare_stage_context

        cfg = _config(chunking={"tokenizer": "nope"})
        with pytest.raises(StagePreconditionError, match="chunk model check failed.*nope"):
            prepare_stage_context(STAGE_CONTRACTS["chunk"], cfg)

    def test_a_stage_without_a_model_is_unaffected(self) -> None:
        from womblex.cloud.stage_contracts import STAGE_CONTRACTS
        from womblex.cloud.stage_runner import prepare_stage_context

        cfg = _config(chunking={"tokenizer": "nope"})
        assert prepare_stage_context(STAGE_CONTRACTS["money"], cfg) is not None

    def test_off_skips_the_check(self) -> None:
        from womblex.cloud.stage_contracts import STAGE_CONTRACTS
        from womblex.cloud.stage_runner import prepare_stage_context

        cfg = _config(chunking={"tokenizer": "nope"}, processing={"models_check": "off"})
        assert prepare_stage_context(STAGE_CONTRACTS["chunk"], cfg) is not None
