"""Tests for womblex.config — YAML loading and Pydantic validation."""


from pathlib import Path


import pytest

from pydantic import ValidationError


from womblex.config import (

    ChunkingConfig,

    DatasetConfig,

    DetectionConfig,

    ExtractionConfig,
    WomblexConfig,

    RedactionConfig,

    load_config,
)



# ---------------------------------------------------------------------------

# Defaults

# ---------------------------------------------------------------------------



class TestDetectionConfigDefaults:

    def test_default_min_text_coverage(self) -> None:

        cfg = DetectionConfig()

        assert cfg.min_text_coverage == 0.3


    def test_default_form_signal_threshold(self) -> None:

        cfg = DetectionConfig()

        assert cfg.form_signal_threshold == 0.5


    def test_default_table_signal_threshold(self) -> None:

        cfg = DetectionConfig()

        assert cfg.table_signal_threshold == 0.4



class TestChunkingConfigDefaults:

    def test_default_tokenizer(self) -> None:

        cfg = ChunkingConfig()

        assert cfg.tokenizer == "isaacus/kanon-2-tokenizer"


    def test_default_chunk_size(self) -> None:

        cfg = ChunkingConfig()

        assert cfg.chunk_size == 480


    def test_default_enabled(self) -> None:

        cfg = ChunkingConfig()

        assert cfg.enabled is True


    def test_default_chunk_tables(self) -> None:

        cfg = ChunkingConfig()

        assert cfg.chunk_tables is True



# ---------------------------------------------------------------------------

# Validation

# ---------------------------------------------------------------------------



class TestDetectionConfigValidation:

    def test_rejects_negative_text_coverage(self) -> None:

        with pytest.raises(ValidationError):

            DetectionConfig(min_text_coverage=-0.1)


    def test_rejects_text_coverage_above_one(self) -> None:

        with pytest.raises(ValidationError):

            DetectionConfig(min_text_coverage=1.1)


    def test_accepts_boundary_values(self) -> None:

        cfg = DetectionConfig(min_text_coverage=0.0, form_signal_threshold=1.0)

        assert cfg.min_text_coverage == 0.0

        assert cfg.form_signal_threshold == 1.0



class TestChunkingConfigValidation:

    def test_rejects_zero_chunk_size(self) -> None:

        with pytest.raises(ValidationError):

            ChunkingConfig(chunk_size=0)


    def test_accepts_small_chunk_size(self) -> None:

        cfg = ChunkingConfig(chunk_size=1)

        assert cfg.chunk_size == 1



class TestExtractionConfigDefaults:

    def test_default_ocr_dpi(self) -> None:

        cfg = ExtractionConfig()

        assert cfg.ocr.dpi == 200


    def test_default_ocr_lang(self) -> None:

        cfg = ExtractionConfig()
        assert cfg.ocr.lang == "eng"




class TestRedactionConfigDefaults:

    def test_default_enabled(self) -> None:

        cfg = RedactionConfig()

        assert cfg.enabled is True


    def test_default_mode(self) -> None:

        cfg = RedactionConfig()
        assert cfg.mode == "flag"


    def test_accepts_all_modes(self) -> None:

        for mode in ("flag", "blackout", "delete"):

            cfg = RedactionConfig(mode=mode)
            assert cfg.mode == mode


    def test_default_dpi(self) -> None:

        cfg = RedactionConfig()

        assert cfg.dpi == 150


# ---------------------------------------------------------------------------

# YAML loading

# ---------------------------------------------------------------------------



class TestLoadConfig:

    def test_loads_valid_yaml(self, sample_config_path: Path) -> None:
        cfg = load_config(sample_config_path)
        assert cfg.dataset.name == "test_dataset"

        assert cfg.paths.input_root == Path("./data/raw/test")

        assert cfg.detection.min_text_coverage == 0.3

        assert cfg.chunking.chunk_size == 480

        assert cfg.chunking.enabled is False

        assert cfg.processing.batch_size == 10


    def test_raises_on_missing_file(self, tmp_path: Path) -> None:

        with pytest.raises(FileNotFoundError):

            load_config(tmp_path / "nonexistent.yaml")


    def test_raises_on_invalid_schema(self, tmp_path: Path) -> None:

        bad = tmp_path / "bad.yaml"

        bad.write_text("dataset:\n  name: 123\npaths:\n  wrong: true\n")

        with pytest.raises(ValidationError):

            load_config(bad)


    def test_loads_example_config(self) -> None:

        """Verify the example project config file parses correctly."""

        cfg_path = Path(__file__).resolve().parent.parent / "configs" / "example.yaml"

        if cfg_path.exists():
            cfg = load_config(cfg_path)

            assert cfg.dataset.name == "my_dataset"

            assert cfg.chunking.tokenizer == "isaacus/kanon-2-tokenizer"



# ---------------------------------------------------------------------------

# PipelineConfig construction

# ---------------------------------------------------------------------------



class TestPipelineConfig:

    def test_minimal_construction(self) -> None:
        cfg = WomblexConfig(

            dataset=DatasetConfig(name="test"),

            paths={

                "input_root": "/tmp/in",

                "output_root": "/tmp/out",

                "checkpoint_dir": "/tmp/ckpt",

            },
        )
        assert cfg.dataset.name == "test"

        assert cfg.detection.min_text_coverage == 0.3  # default


    def test_override_detection(self) -> None:
        cfg = WomblexConfig(

            dataset=DatasetConfig(name="test"),

            paths={

                "input_root": "/tmp/in",

                "output_root": "/tmp/out",

                "checkpoint_dir": "/tmp/ckpt",

            },

            detection=DetectionConfig(min_text_coverage=0.5),
        )

        assert cfg.detection.min_text_coverage == 0.5

