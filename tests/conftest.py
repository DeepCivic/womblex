"""Shared test fixtures.

All test data comes from real documents in ``fixtures/``. No synthetic
data is generated — the curated fixture set represents the hardest
extraction challenges from real government document releases.
"""

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "fixtures"


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test outputs."""
    return tmp_path


# ---------------------------------------------------------------------------
# Real fixture paths
# ---------------------------------------------------------------------------


@pytest.fixture
def funsd_image_dir() -> Path:
    """FUNSD form images directory."""
    return FIXTURES_DIR / "funsd" / "images"


@pytest.fixture
def funsd_annotation_dir() -> Path:
    """FUNSD annotation JSON directory."""
    return FIXTURES_DIR / "funsd" / "annotations"


@pytest.fixture
def iam_line_dir() -> Path:
    """IAM handwriting line images and ground truth."""
    return FIXTURES_DIR / "iam_line"


@pytest.fixture
def doclaynet_dir() -> Path:
    """DocLayNet layout pages and annotations."""
    return FIXTURES_DIR / "doclaynet"


@pytest.fixture
def spreadsheet_dir() -> Path:
    """Real spreadsheet fixtures (CSV, Excel)."""
    return FIXTURES_DIR / "womblex-collection" / "_spreadsheets"


@pytest.fixture
def sample_config_path(tmp_path: Path) -> Path:
    """Write a minimal valid config YAML and return its path."""
    config_text = """\
dataset:
  name: test_dataset

paths:
  input_root: ./data/raw/test
  output_root: ./data/processed/test
  checkpoint_dir: ./data/checkpoints/test

detection:
  min_text_coverage: 0.3
  form_signal_threshold: 0.5
  table_signal_threshold: 0.4

extraction:
  native:
    include_tables: true
  ocr:
    engine: paddleocr
    dpi: 200
    lang: eng

redaction:
  enabled: true
  mode: flag
  threshold: 50
  min_area_ratio: 0.001
  max_area_ratio: 0.9
  dpi: 150

chunking:
  tokenizer: "isaacus/kanon-2-tokenizer"
  chunk_size: 480
  enabled: false
  chunk_tables: true

processing:
  batch_size: 10
  checkpoint_every: 10
"""
    p = tmp_path / "test_config.yaml"
    p.write_text(config_text)
    return p
