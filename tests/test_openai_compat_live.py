"""A live call to an OpenAI-compatible endpoint; skipped without one, like the Isaacus ones."""

from __future__ import annotations

import os

import pytest

from womblex.config import EmbeddingConfig
from womblex.utils.deployment_registry import client_for

pytestmark = pytest.mark.skipif(
    not (os.environ.get("OPENAI_COMPAT_BASE_URL") and os.environ.get("OPENAI_COMPAT_TEST_MODEL")),
    reason="OPENAI_COMPAT_BASE_URL and OPENAI_COMPAT_TEST_MODEL not set — add them for a real endpoint check",
)


def test_the_provider_embeds_against_a_live_endpoint() -> None:
    from womblex.analyse.embed import make_embedding_provider

    config = EmbeddingConfig(model="openai-compatible",
                             provider_options={"model": os.environ["OPENAI_COMPAT_TEST_MODEL"]})
    vectors = make_embedding_provider(config, client_for("openai-compatible")).embed(["The applicant signed."])
    assert len(vectors) == 1 and len(vectors[0]) > 0
