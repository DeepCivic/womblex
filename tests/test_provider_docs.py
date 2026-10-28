"""A model on a billed deployment is documented: a `docs/models.md` entry and a **Paid:** line naming it."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests import _fixture_vendor
from womblex.utils import deployment_registry as dr
from womblex.utils import model_registry as mr
from womblex.utils.model_check import _register_builtins

ROOT = Path(__file__).resolve().parent.parent
_register_builtins()


def _normal(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def models_md_names(text: str) -> set[str]:
    """Every backticked name in the first column of a `docs/models.md` table row."""
    names: set[str] = set()
    for line in text.splitlines():
        if line.startswith("| `"):
            names.update(re.findall(r"`([^`]+)`", line.split("|")[1]))
    return names


def paid_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("**Paid:**")]


def billed_models() -> list[tuple[str, str]]:
    return [(slot, name) for slot in mr.SLOTS for name in mr.known_names(slot)
            if any(dr.resolve(d).billed for d in dr.deployments_for(mr.resolve(slot, name)))]


def problems(slot: str, name: str, models_md: str, requirements: str) -> list[str]:
    """What is missing for the model registered as *name* in *slot*; empty when it is documented."""
    entry = mr.resolve(slot, name)
    labels = {entry.name, entry.vendor_model} - {""}
    out = []
    if not labels & models_md_names(models_md):
        out.append(f"{slot} model {entry.name!r} has no entry in docs/models.md (under {sorted(labels)})")
    wanted = {_normal(label) for label in labels}
    if not any(w in _normal(line) for w in wanted for line in paid_lines(requirements)):
        out.append(f"{slot} model {entry.name!r} is named by no **Paid:** line in docs/functional_requirements.md")
    return out


MODELS_MD = (ROOT / "docs" / "models.md").read_text()
REQUIREMENTS = (ROOT / "docs" / "functional_requirements.md").read_text()


def test_the_billed_built_ins_are_found() -> None:
    assert {name for _, name in billed_models()} >= {"kanon-2-enricher", "kanon-2-embedder", "mistral-ocr"}
    assert "ollama" not in {name for _, name in billed_models()}  # a local server bills nothing


@pytest.mark.parametrize(("slot", "name"), billed_models(), ids=[f"{s}:{n}" for s, n in billed_models()])
def test_a_model_on_a_billed_deployment_is_documented(slot: str, name: str) -> None:
    assert problems(slot, name, MODELS_MD, REQUIREMENTS) == []


def test_a_billed_model_with_no_entry_and_no_paid_line_fails(monkeypatch) -> None:
    _fixture_vendor.install(monkeypatch)
    ((slot, name),) = [(s, n) for s, n in _fixture_vendor.MODELS if s == mr.SLOT_EMBEDDING]
    assert (slot, name) in billed_models()
    found = problems(slot, name, MODELS_MD, REQUIREMENTS)
    assert len(found) == 2
    assert "has no entry in docs/models.md" in found[0] and "named by no **Paid:** line" in found[1]


def test_the_entry_and_the_paid_line_each_clear_their_own_problem(monkeypatch) -> None:
    _fixture_vendor.install(monkeypatch)
    slot, name = mr.SLOT_EMBEDDING, _fixture_vendor.EMBED
    entry = f"| `{_fixture_vendor.VENDOR_EMBED}` | Fixture Vendor | `x.py` | embeds | `embedding.model: {name}` |\n"
    assert len(problems(slot, name, MODELS_MD + entry, REQUIREMENTS)) == 1
    paid = f"**Paid:** Each request to `{name}` is billed.\n"
    assert problems(slot, name, MODELS_MD + entry, REQUIREMENTS + paid) == []
    assert len(problems(slot, name, MODELS_MD, REQUIREMENTS + paid)) == 1


def test_a_local_model_needs_neither(monkeypatch) -> None:
    assert ("ocr", "paddleocr") not in billed_models() and ("ocr", "ollama") not in billed_models()
