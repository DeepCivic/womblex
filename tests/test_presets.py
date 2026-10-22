"""Built-in and operator-saved presets (`womblex.presets`)."""
from __future__ import annotations

from pathlib import Path

import pytest

from womblex import presets
from womblex.cloud.deployment import DeploymentSettings
from womblex.config import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
_MINIMAL = {"dataset": {"name": "t"}}


def _save(directory: Path, name: str, config: dict) -> None:
    record, _ = presets.build_preset_record(name=name, description="", formats=(), config=config)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / presets.preset_filename(name)).write_text(presets.serialise_preset_record(record))


class TestDefaultIsaacus:
    def test_is_listed_with_its_formats(self) -> None:
        listed = {p["name"]: p for p in presets.list_presets()}
        assert listed["DEFAULT-Isaacus"]["formats"] == [".pdf", ".docx", ".md", ".markdown"]
        assert listed["DEFAULT-Isaacus"]["source"] == "builtin"

    def test_enables_the_full_reference_shape(self) -> None:
        """Extract, normalise, enrich, chunk, graph, embed, money and link, with AI
        chunking. spellfix is a per-corpus call, so the reference default omits it."""
        cfg = presets.get_preset("DEFAULT-Isaacus").config
        assert cfg["chunking"]["enabled"] is True
        assert cfg["chunking"]["chunking_model"] == "kanon-2-enricher"
        for stage in ("enrichment", "embedding", "money", "linking"):
            assert cfg[stage]["enabled"] is True
        assert "spellfix" not in cfg
        # Text cleaning is selected once; this is also what gates normalise into dispatch.
        assert cfg["processing"]["text_source"] == "normalised"
        # The operator supplies the run's identity and paths.
        assert "dataset" not in cfg and "paths" not in cfg

    def test_overlaid_on_a_minimal_config_it_validates(self) -> None:
        presets._validate_overlay({**_MINIMAL, **presets.get_preset("DEFAULT-Isaacus").config})

    def test_the_config_file_ships_and_loads(self) -> None:
        """The preset is CLI-runnable: `configs/default-isaacus.yaml` loads."""
        cfg = load_config(REPO_ROOT / "configs" / "default-isaacus.yaml")
        assert cfg.chunking.enabled and cfg.chunking.chunking_model == "kanon-2-enricher"
        assert cfg.enrichment.enabled and cfg.embedding.enabled
        assert cfg.money.enabled and cfg.linking.enabled
        assert not cfg.spellfix.enabled, "spellfix is a per-corpus opt-in"
        assert cfg.processing.text_source == "normalised"
        # AI chunking + enrich both on => reuse auto-wired, so no double enrich.
        assert cfg.enrichment.persist_document is True

    def test_the_config_file_and_the_preset_agree(self) -> None:
        """They must not drift on the stage toggles and settings the preset carries."""
        cfg = load_config(REPO_ROOT / "configs" / "default-isaacus.yaml")
        overlay = presets.get_preset("DEFAULT-Isaacus").config
        assert overlay["chunking"]["chunk_size"] == cfg.chunking.chunk_size
        assert overlay["chunking"]["chunking_model"] == cfg.chunking.chunking_model
        assert overlay["chunking"]["overlap"] == cfg.chunking.overlap
        assert overlay["enrichment"]["enabled"] == cfg.enrichment.enabled
        assert overlay["embedding"]["enabled"] == cfg.embedding.enabled
        assert overlay["money"]["enabled"] == cfg.money.enabled
        assert overlay["money"]["default_currency"] == cfg.money.default_currency
        assert overlay["linking"]["enabled"] == cfg.linking.enabled
        assert "spellfix" not in overlay and not cfg.spellfix.enabled
        assert overlay["processing"]["text_source"] == cfg.processing.text_source


class TestSavedPresets:
    def test_a_local_directory_is_read_and_a_corrupt_file_skipped(self, tmp_path: Path) -> None:
        _save(tmp_path / "presets", "My-Run", {"chunking": {"enabled": True, "chunk_size": 320}})
        (tmp_path / "presets" / "Broken.preset.json").write_text("{not json")
        settings = DeploymentSettings(output_root=tmp_path, store_uri=None, presets_dir=tmp_path / "presets")
        saved = presets.read_saved_presets(settings)
        assert set(saved) == {"My-Run"}
        assert saved["My-Run"].config["chunking"]["chunk_size"] == 320
        assert saved["My-Run"].source == "saved"

    def test_no_directory_is_no_saved_presets(self, tmp_path: Path) -> None:
        settings = DeploymentSettings(output_root=tmp_path, store_uri=None)
        assert presets.read_saved_presets(settings) == {}

    def test_a_store_is_read_under_its_presets_prefix(self, tmp_path: Path) -> None:
        pytest.importorskip("fsspec")
        store_root = tmp_path / "store"
        _save(store_root / presets.PRESETS_DIRNAME, "Remote-One", {"chunking": {"chunk_size": 128}})
        settings = DeploymentSettings(output_root=None, store_uri=str(store_root))
        assert set(presets.read_saved_presets(settings)) == {"Remote-One"}

    def test_a_saved_preset_shadows_a_built_in_of_the_same_name(self, tmp_path: Path) -> None:
        _save(tmp_path / "presets", "DEFAULT-Isaacus", {"chunking": {"chunk_size": 999}})
        settings = DeploymentSettings(output_root=tmp_path, store_uri=None, presets_dir=tmp_path / "presets")
        one = presets.get_any_preset(settings, "DEFAULT-Isaacus")
        assert one is not None and one.source == "saved"
        assert one.config["chunking"]["chunk_size"] == 999

    def test_an_unsafe_name_reaches_only_the_built_ins(self, tmp_path: Path) -> None:
        settings = DeploymentSettings(output_root=tmp_path, store_uri=None, presets_dir=tmp_path)
        assert presets.get_any_preset(settings, "../DEFAULT-Isaacus") is None
        assert presets.get_any_preset(settings, "nope") is None
        assert presets.get_any_preset(settings, "DEFAULT-Isaacus") is not None

    def test_a_config_that_would_not_load_is_not_saved(self) -> None:
        with pytest.raises(Exception, match="chnk_size"):
            presets.build_preset_record(
                name="Bad", description="", formats=(), config={"chunking": {"chnk_size": 1}},
            )

    def test_an_unsafe_name_is_refused(self) -> None:
        with pytest.raises(ValueError):
            presets.build_preset_record(name="../x", description="", formats=(), config={})
