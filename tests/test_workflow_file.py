"""A workflow file: the master copy every workflow command reads (Phase R, D1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from womblex.cloud.workflow_file import WorkflowFileError, load_workflow_file, workflow_file_name

EXAMPLE = Path(__file__).resolve().parent.parent / "configs" / "workflows" / "nightly.yaml"


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_the_digest_ignores_formatting_and_comments(tmp_path: Path) -> None:
    plain = _write(tmp_path, "a.yaml", "name: nightly\npreset: DEFAULT-Isaacus\n"
                                       "steps: [{stage: normalise}, {stage: chunk}]\n")
    styled = _write(tmp_path, "b.yaml", (
        "# a comment\n"
        "name:   nightly    # trailing\n"
        "steps:\n"
        "  - stage: normalise\n"
        "  - {stage: chunk}\n"
        'preset: "DEFAULT-Isaacus"\n'
        "input_binding: {batch_size: 50}   # the default, spelt out\n"
    ))
    assert load_workflow_file(plain).digest == load_workflow_file(styled).digest


def test_a_changed_step_changes_the_digest(tmp_path: Path) -> None:
    one = _write(tmp_path, "a.yaml", "name: w\npreset: DEFAULT-Isaacus\nsteps: [{stage: normalise}]\n")
    two = _write(tmp_path, "b.yaml", "name: w\npreset: DEFAULT-Isaacus\nsteps: [{stage: chunk}]\n")
    assert load_workflow_file(one).digest != load_workflow_file(two).digest


def test_the_example_workflow_loads() -> None:
    stored = load_workflow_file(EXAMPLE)
    assert stored.id == "nightly" and stored.spec.triggers
    assert [s.stage for s in stored.spec.steps] == ["normalise", "enrich", "chunk", "embed"]


@pytest.mark.parametrize("text", [
    "preset: DEFAULT-Isaacus\nsteps: [{stage: chunk}]\n",                    # no name
    "name: ../escape\npreset: DEFAULT-Isaacus\nsteps: [{stage: chunk}]\n",   # unsafe name
    "name: w\npreset: DEFAULT-Isaacus\nsteps: [{stage: chunk}]\nwebhook: x\n",  # unknown key
    "- just a list\n",
])
def test_an_invalid_file_is_refused_as_readable(tmp_path: Path, text: str) -> None:
    with pytest.raises(WorkflowFileError) as e:
        load_workflow_file(_write(tmp_path, "w.yaml", text))
    assert e.value.readable


def test_an_unreadable_file_is_refused_as_unreadable(tmp_path: Path) -> None:
    for path in (tmp_path / "missing.yaml", _write(tmp_path, "bad.yaml", "name: [unclosed\n")):
        with pytest.raises(WorkflowFileError) as e:
            load_workflow_file(path)
        assert not e.value.readable


def test_the_name_is_read_without_validating_the_rest(tmp_path: Path) -> None:
    stale = _write(tmp_path, "w.yaml", "name: old\npreset: DEFAULT-Isaacus\nsteps: [{stage: no-such-stage}]\n")
    assert workflow_file_name(stale) == "old"
    with pytest.raises(WorkflowFileError):
        load_workflow_file(stale)
