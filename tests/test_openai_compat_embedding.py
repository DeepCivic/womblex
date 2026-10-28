"""The OpenAI-compatible embedding provider, over a mocked transport."""

from __future__ import annotations

import json

import httpx
import pytest

from tests.test_embed_stage import shard_with_chunks  # noqa: F401
from tests.test_model_check import _config
from tests.test_provider_conformance import check_declarations
from womblex.analyse.embed import capability_problems, make_embedding_provider
from womblex.config import EmbeddingConfig
from womblex.providers.openai_compat import deployment as dep
from womblex.providers.openai_compat.embedding import OpenAICompatEmbedding
from womblex.utils import deployment_registry as dr
from womblex.utils import model_registry as mr
from womblex.utils.model_check import SCOPE_EMBED, check_models, check_registered


class Server:
    """A mock `/v1/embeddings` server behind `httpx.MockTransport`."""

    def __init__(self, *, width: int = 3, status: int = 200, ignore_dimensions: bool = False) -> None:
        self.requests: list[httpx.Request] = []
        self.width, self.status, self.ignore_dimensions = width, status, ignore_dimensions

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path.endswith("/models"):
            return httpx.Response(self.status, json={"data": []})
        body = json.loads(request.content)
        if self.status != 200:
            return httpx.Response(self.status, json={"error": {"message": "input too long"}})
        width = self.width if self.ignore_dimensions else body.get("dimensions", self.width)
        # returned out of order, as a server may
        data = [{"index": i, "embedding": [float(i)] * width} for i in reversed(range(len(body["input"])))]
        return httpx.Response(200, json={"data": data})

    def client(self) -> httpx.Client:
        return httpx.Client(base_url="http://server.test/v1", transport=httpx.MockTransport(self.handler))


def _provider(server: Server, **config) -> OpenAICompatEmbedding:
    cfg = EmbeddingConfig(model="openai-compatible", provider_options={"model": "text-embed-1"}, **config)
    return OpenAICompatEmbedding(server.client(), cfg)


def test_it_conforms_to_the_embedding_interface_and_names_its_deployment() -> None:
    check_declarations(mr.SLOT_EMBEDDING, "openai-compatible")
    assert mr.resolve(mr.SLOT_EMBEDDING, "openai").name == "openai-compatible"  # the alias
    assert mr.resolve(mr.SLOT_EMBEDDING, "openai-compatible").deployments == ("openai-compatible",)
    server = Server()
    p = make_embedding_provider(EmbeddingConfig(model="openai-compatible", provider_options={"model": "m"}),
                                client=server.client())
    assert p.name == "openai-compatible" and p.limits.max_texts_per_request == 64


def test_it_posts_the_texts_and_returns_the_vectors_in_input_order() -> None:
    server = Server()
    vectors = _provider(server).embed(["a", "b", "c"])
    assert vectors == [[0.0] * 3, [1.0] * 3, [2.0] * 3]
    (request,) = server.requests
    assert request.url.path == "/v1/embeddings"
    assert json.loads(request.content) == {"model": "text-embed-1", "input": ["a", "b", "c"]}


def test_roles_are_ignored_and_dimensions_are_sent_and_checked() -> None:
    assert _provider(Server()).embed(["a"], role="query") == [[0.0] * 3]
    server = Server()
    assert [len(v) for v in _provider(server, dimensions=5).embed(["a"])] == [5]
    assert json.loads(server.requests[0].content)["dimensions"] == 5
    with pytest.raises(RuntimeError, match="width 3, not the 5 asked for"):
        _provider(Server(ignore_dimensions=True), dimensions=5).embed(["a"])


def test_a_chunk_the_server_rejects_is_that_batchs_error_and_the_stage_goes_on(shard_with_chunks, monkeypatch) -> None:  # noqa: F811
    from womblex.analyse import embed_stage

    server = Server(status=400)
    monkeypatch.setattr(embed_stage, "make_embedding_provider", lambda *a, **k: _provider(server))
    with pytest.raises(httpx.HTTPStatusError):
        _provider(server).embed(["too long"])
    result = embed_stage.embed_shards(shard_with_chunks, EmbeddingConfig(), client=None)
    assert result.batches_written == 1 and result.chunks_embedded == 0  # the batch is recorded as failed, not retried here


def test_the_server_returning_the_wrong_number_of_vectors_is_an_error() -> None:
    class Short(Server):
        def handler(self, request):
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

    with pytest.raises(RuntimeError, match="1 vectors for 2 texts"):
        _provider(Short()).embed(["a", "b"])


def test_the_batch_size_is_the_providers_setting() -> None:
    p = make_embedding_provider(EmbeddingConfig(
        model="openai-compatible", provider_options={"model": "m", "max_texts_per_request": 8}), client=None)
    assert p.limits.max_texts_per_request == 8


def test_the_server_model_id_is_required_before_launch() -> None:
    config = _config(embedding={"enabled": True, "model": "openai-compatible"})
    ((scope, _slot, _model, message),) = capability_problems(config)
    assert scope == "embed" and "needs embedding.provider_options: model" in message
    with pytest.raises(ValueError, match="needs embedding.provider_options: model"):
        check_registered(config)
    assert capability_problems(_config(embedding={"model": "openai-compatible", "provider_options": {"model": "m"}})) == []


def test_dimensions_are_honoured_so_a_width_is_not_refused() -> None:
    config = _config(embedding={"model": "openai-compatible", "dimensions": 64, "provider_options": {"model": "m"}})
    assert capability_problems(config) == []


def test_the_sidecar_records_the_servers_model_id(shard_with_chunks, monkeypatch) -> None:  # noqa: F811
    from womblex.analyse import embed_stage
    from womblex.store.embed_output import read_embeddings

    monkeypatch.setattr(embed_stage, "make_embedding_provider", lambda *a, **k: _provider(Server()))
    result = embed_stage.embed_shards(shard_with_chunks, EmbeddingConfig(model="openai-compatible"), client=None)
    assert result.chunks_embedded >= 1
    assert {r["model"] for r in read_embeddings(shard_with_chunks).to_pylist()} == {"text-embed-1"}


@pytest.fixture
def base_url(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "https://user:pw@api.vendor.test/v1?token=abc")
    monkeypatch.delenv("OPENAI_COMPAT_API_KEY", raising=False)


def test_the_deployment_is_a_built_in_found_without_importing_its_module() -> None:
    import subprocess
    import sys

    code = "from womblex.utils.deployment_registry import known_deployments; print(known_deployments())"
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True).stdout
    assert "openai-compatible" in out


def test_the_deployment_is_billed_and_networked_by_default_and_conforms() -> None:
    from tests.test_deployment_registry import assert_conforms

    d = dr.resolve("openai-compatible")
    assert d.env == (("OPENAI_COMPAT_BASE_URL",),) and d.make_client is not None
    assert_conforms(d)
    assert dep.billed_by_setting() is True


@pytest.mark.parametrize(("value", "billed"), [("false", False), ("0", False), ("No", False), ("true", True), ("", True)])
def test_a_self_hosted_server_is_marked_unbilled_by_its_own_setting(monkeypatch, value, billed) -> None:
    monkeypatch.setenv("OPENAI_COMPAT_BILLED", value)
    assert dep.billed_by_setting() is billed


def test_an_unbilled_self_hosted_deployment_is_off_the_network_and_conforms(monkeypatch) -> None:
    from tests.test_deployment_registry import assert_conforms

    monkeypatch.setenv("OPENAI_COMPAT_BILLED", "false")
    monkeypatch.setattr(dr, "_deployments", dict(dr._deployments))
    import importlib

    importlib.reload(dep)  # registers again under this setting
    d = dr.resolve("openai-compatible")
    assert (d.billed, d.needs_network, d.smoke) == (False, False, None)
    assert_conforms(d)


def test_the_client_is_rooted_at_the_base_url_with_the_key_as_a_bearer_token(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://server.test/v1/")
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "k-123")  # pragma: allowlist secret -- a test placeholder
    client = dep.make_client()
    assert str(client.base_url) == "http://server.test/v1/" and client.headers["authorization"] == "Bearer k-123"
    monkeypatch.delenv("OPENAI_COMPAT_API_KEY")
    assert "authorization" not in dep.make_client().headers


def test_no_base_url_is_a_refusal_naming_the_variable(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_COMPAT_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="OPENAI_COMPAT_BASE_URL is not set"):
        dep.make_client()
    assert dr.resolve("openai-compatible").unset() == [("OPENAI_COMPAT_BASE_URL",)]


@pytest.mark.parametrize(("status", "ok"), [(200, True), (404, True), (401, False), (503, False)])
def test_the_free_check_asks_the_server_and_a_refused_key_or_an_outage_fails_it(monkeypatch, status, ok) -> None:
    server = Server(status=status)
    monkeypatch.setattr(dep, "make_client", lambda models=(): server.client())
    if ok:
        dep.check(["m"])
    else:
        with pytest.raises(httpx.HTTPStatusError):
            dep.check(["m"])
    assert server.requests[0].url.path.endswith("/models")


def test_what_the_footer_may_say_is_the_host_and_never_the_credentials_or_query(base_url) -> None:
    assert dep.detail("m") == {"endpoint": "api.vendor.test"}
    text = json.dumps(dep.detail("m"))
    assert "pw" not in text and "abc" not in text


def test_a_run_over_it_is_checked_and_recorded_through_its_deployment(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://server.test/v1")
    server = Server()
    monkeypatch.setattr(dep, "make_client", lambda models=(): server.client())
    dr.reset_clients()
    config = _config(embedding={"enabled": True, "model": "openai-compatible", "provider_options": {"model": "m"}})
    result = check_models(config, "load", scopes=(SCOPE_EMBED,))
    assert result.ok and [c.source for c in result.checks] == ["openai-compatible"]
    assert any(r.url.path.endswith("/models") for r in server.requests)
