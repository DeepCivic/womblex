"""The deployment registry, and the conformance every registered deployment must meet."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from womblex.utils import deployment_registry as dr
from womblex.utils import model_registry as mr
from womblex.utils.deployment_registry import Deployment

_VARIABLE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def assert_conforms(d: Deployment) -> None:
    """What a deployment owes the pre-run checks, whatever vendor supplies it."""
    assert d.name == d.name.strip().lower() and d.name
    for group in d.env:
        assert group, "an empty alternatives group can never be satisfied"
        assert all(_VARIABLE.match(v) for v in group), f"{d.name}: env names variables, never a value"
    assert all(_VARIABLE.match(v) for v in d.env_optional)
    assert callable(d.check)
    assert isinstance(d.serves("a-model-id"), bool)
    assert d.billed or not d.needs_network, f"{d.name}: a deployment that needs the network is billed"
    assert d.smoke is None or d.billed, f"{d.name}: a smoke call is a billed call"


@pytest.fixture
def clean(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(dr, "_deployments", {})
    monkeypatch.setattr(dr, "_loaded", True)  # no built-ins or plugins unless a test registers them
    monkeypatch.setattr(dr, "plugin_objects", lambda group: iter(()))


def _deployment(name: str = "acme", **kw) -> Deployment:
    kw.setdefault("check", lambda models: None)
    kw.setdefault("needs_network", True)
    kw.setdefault("billed", True)
    return Deployment(name=name, **kw)


@pytest.mark.parametrize("name", dr.known_deployments())
def test_every_registered_deployment_conforms(name: str) -> None:
    assert_conforms(dr.resolve(name))


def test_every_model_names_only_registered_deployments() -> None:
    known = set(dr.known_deployments())
    for slot in mr.SLOTS:
        for model in mr.known_names(slot):
            assert set(mr.resolve(slot, model).deployments) <= known, f"{slot} {model}"


def test_the_conformance_check_rejects_a_secret_in_place_of_a_variable_name() -> None:
    with pytest.raises(AssertionError):
        assert_conforms(_deployment(env=(("sk-live-0123456789",),)))


def test_the_conformance_check_rejects_an_unbilled_network_deployment() -> None:
    with pytest.raises(AssertionError):
        assert_conforms(_deployment(billed=False))


def test_register_and_resolve(clean) -> None:
    d = _deployment("Acme")
    dr.register(d)
    assert dr.resolve(" ACME ") is d
    assert dr.known_deployments() == ["acme"]


def test_an_unknown_deployment_lists_the_known_names(clean) -> None:
    dr.register(_deployment())
    with pytest.raises(ValueError, match=r"unknown deployment: 'nope' \(known: \['acme'\]\)"):
        dr.resolve("nope")


def test_a_name_held_by_another_source_is_refused(clean) -> None:
    dr.register(_deployment())
    with pytest.raises(ValueError, match="already registered by 'builtin'"):
        dr.register(_deployment(source="acme-dist"))


def test_unset_names_the_required_groups_with_no_alternative_set(clean, monkeypatch: pytest.MonkeyPatch) -> None:
    d = _deployment(env=(("ACME_KEY", "ACME_ENDPOINTS"), ("ACME_REGION",)))
    monkeypatch.delenv("ACME_KEY", raising=False)
    monkeypatch.delenv("ACME_ENDPOINTS", raising=False)
    monkeypatch.setenv("ACME_REGION", "  ")  # blank is unset
    assert d.unset() == [("ACME_KEY", "ACME_ENDPOINTS"), ("ACME_REGION",)]
    monkeypatch.setenv("ACME_ENDPOINTS", "e1")
    monkeypatch.setenv("ACME_REGION", "ap-southeast-2")
    assert d.unset() == []


def test_a_plugin_deployment_carries_its_distribution_and_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dr, "_deployments", {})
    monkeypatch.setattr(dr, "_loaded", False)
    monkeypatch.setattr(dr, "_BUILTIN_MODULES", ())
    monkeypatch.setattr(dr, "plugin_objects", lambda group: iter([
        ("acme", _deployment(), "acme-dist", "1.2.3"),
        ("mismatch", _deployment("other"), "acme-dist", "1.2.3"),  # entry-point name must match
        ("not-one", SimpleNamespace(name="not-one"), "acme-dist", "1.2.3"),
    ]))
    assert dr.known_deployments() == ["acme"]
    assert dr.distribution_version(dr.resolve("acme")) == ("acme-dist", "1.2.3")


def test_a_built_in_reports_womblexs_own_version() -> None:
    from womblex import __version__

    assert dr.distribution_version(_deployment()) == ("womblex", __version__)


@pytest.mark.parametrize("billed", [True, False])
def test_a_pinging_model_without_a_deployment_is_given_one_built_from_its_ping(billed: bool) -> None:
    calls: list[str] = []

    class Reader:
        ping_is_billed = billed

        def ping(self) -> None:
            calls.append("ping")

    d = dr.deployment_for_pinging_model("ocr", "My-Reader", Reader())
    assert d.name == "ocr:my-reader" and d.billed is billed
    d.check(["any-model"])
    assert calls == ["ping"]
    assert_conforms(d)


def test_a_failing_ping_surfaces_as_a_failing_check() -> None:
    class Reader:
        def ping(self) -> None:
            raise ConnectionError("refused")

    with pytest.raises(ConnectionError):
        dr.deployment_for_pinging_model("ocr", "r", Reader()).check(["m"])


def test_a_model_registers_against_deployments_and_a_vendor_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mr, "_entries", {s: {} for s in mr.SLOTS})
    monkeypatch.setattr(mr, "_aliases", {s: {} for s in mr.SLOTS})
    monkeypatch.setattr(mr, "_loaded", set(mr.SLOTS))
    mr.register(mr.SLOT_EMBEDDING, "acme-embed", lambda **_: None,
                deployments=("acme", "acme-private"), vendor_model="acme-embed-v2")
    entry = mr.resolve(mr.SLOT_EMBEDDING, "acme-embed")
    assert entry.deployments == ("acme", "acme-private") and entry.vendor_model == "acme-embed-v2"


def test_a_plugin_factory_declares_its_deployments_as_it_declares_traits(monkeypatch: pytest.MonkeyPatch) -> None:
    def factory(**_):
        return None

    factory.womblex_deployments = ("acme",)  # type: ignore[attr-defined]
    factory.womblex_vendor_model = "acme-1"  # type: ignore[attr-defined]
    monkeypatch.setattr(mr, "_entries", {s: {} for s in mr.SLOTS})
    monkeypatch.setattr(mr, "_aliases", {s: {} for s in mr.SLOTS})
    monkeypatch.setattr(mr, "_loaded", set())
    monkeypatch.setattr(mr, "plugin_objects", lambda group: iter(
        [("acme-embed", factory, "acme-dist", "1.0")] if group.endswith(".embedding") else []))
    entry = mr.resolve(mr.SLOT_EMBEDDING, "acme-embed")
    assert (entry.deployments, entry.vendor_model, entry.source) == (("acme",), "acme-1", "acme-dist")


def test_a_process_builds_one_client_per_deployment_and_every_caller_gets_it(clean) -> None:
    built: list[list[str]] = []
    dr.register(_deployment(make_client=lambda models: built.append(list(models)) or object()))
    first = dr.client_for("acme", ["m1"])
    assert dr.client_for("acme", ["m1", "m2"]) is first  # a second stage, a second model
    assert built == [["m1"]]
    dr.reset_clients()
    assert dr.client_for("acme") is not first


def test_a_later_caller_asking_for_a_model_the_deployment_does_not_serve_is_refused(clean) -> None:
    dr.register(_deployment(serves=lambda model: model == "m1", make_client=lambda models: object()))
    dr.client_for("acme", ["m1"])
    with pytest.raises(ValueError, match="deployment 'acme' does not serve m2"):
        dr.client_for("acme", ["m1", "m2"])


def test_a_deployment_that_builds_no_client_says_so(clean) -> None:
    dr.register(_deployment())
    with pytest.raises(ValueError, match="builds no client"):
        dr.client_for("acme")


def test_a_client_that_fails_to_build_is_not_cached(clean) -> None:
    attempts: list[int] = []

    def make(models):
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionError("no route")
        return object()

    dr.register(_deployment(make_client=make))
    with pytest.raises(ConnectionError):
        dr.client_for("acme")
    assert dr.client_for("acme") is dr.client_for("acme") and len(attempts) == 2
