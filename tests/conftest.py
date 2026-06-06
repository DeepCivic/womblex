"""Shared test fixtures.

All test data comes from real documents in ``fixtures/``. No synthetic
data is generated — the curated fixture set represents the hardest
extraction challenges from real government document releases.
"""

import logging
import os
import threading
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "fixtures"

# Load the local .env so real-service tests (Isaacus embed/enrich) can run
# against the live API — the repo validates against real services locally, not
# mocks (CLAUDE.md "Isaacus calls ... real for local validation"). No-op in CI
# where .env is absent, so those tests skip cleanly.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


@pytest.fixture(scope="session")
def isaacus_client():
    """Live Isaacus client for embed/enrich tests. Skips (does not mock) when
    the SDK or ISAACUS_API_KEY is unavailable, so CI stays green without a key
    while local runs exercise the real API."""
    isaacus = pytest.importorskip("isaacus")
    if not os.environ.get("ISAACUS_API_KEY"):
        pytest.skip("ISAACUS_API_KEY not set — add it to .env for real embed/enrich validation")
    return isaacus.Isaacus()


@pytest.fixture(scope="session")
def bad_isaacus_client():
    """Live client with an invalid key — induces a *real* API auth failure for
    error-path tests (checkpoint-not-written-on-failure) without mocking."""
    isaacus = pytest.importorskip("isaacus")
    return isaacus.Isaacus(api_key="iuak_invalid_key_for_failure_path_testing")


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


# ---------------------------------------------------------------------------
# Slow-test / hang monitor
# ---------------------------------------------------------------------------
# Emits a WARNING naming any test still running after PYTEST_SLOW_ALERT seconds
# (default 60), re-firing each interval. With the `log_cli`/`log_file` config in
# pyproject.toml this surfaces a wedged test by name — live and in pytest.log —
# instead of having to infer it from a frozen progress bar.

_monitor_log = logging.getLogger("pytest.monitor")
_SLOW_ALERT_S = float(os.environ.get("PYTEST_SLOW_ALERT", "60"))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: pytest.Item):
    stop = threading.Event()

    def _watch() -> None:
        elapsed = 0.0
        while not stop.wait(_SLOW_ALERT_S):
            elapsed += _SLOW_ALERT_S
            _monitor_log.warning(
                "SLOW/HANG: %s still running after %.0fs", item.nodeid, elapsed
            )

    watcher = threading.Thread(target=_watch, name="slow-test-monitor", daemon=True)
    watcher.start()
    try:
        yield
    finally:
        stop.set()
