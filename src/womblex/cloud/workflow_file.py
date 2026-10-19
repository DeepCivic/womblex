"""A workflow file: the master copy of a workflow (Phase R, D1).

A workflow is a YAML file naming itself (`name`) beside the fields of `StoredWorkflowSpec`.
Every workflow command reads one; only `schedule-workflow` stores a copy, which its firings
read. Unknown keys are refused, and the digest is over the parsed spec, so formatting and
comments do not change it.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from womblex.cloud.workflow_store import (
    StoredWorkflow,
    is_safe_workflow_id,
    validate_workflow_spec,
    workflow_digest,
)


class WorkflowFileError(ValueError):
    """A workflow file that could not be read (`readable` false) or is not a valid workflow."""

    def __init__(self, message: str, *, readable: bool):
        super().__init__(message)
        self.readable = readable


def load_workflow_file(path: Path) -> StoredWorkflow:
    """The workflow *path* holds, validated. Raises `WorkflowFileError`."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise WorkflowFileError(f"could not read workflow file {path}: {e}", readable=False) from e
    if not isinstance(raw, dict):
        raise WorkflowFileError(f"{path}: a workflow file is a mapping", readable=True)
    name = raw.pop("name", None)
    if not isinstance(name, str) or not is_safe_workflow_id(name):
        raise WorkflowFileError(
            f"{path}: `name` must be one safe token (letters, digits, `.`, `_`, `-`), got {name!r}",
            readable=True,
        )
    try:
        spec = validate_workflow_spec(raw)
    except (ValidationError, ValueError) as e:
        raise WorkflowFileError(f"{path}: not a valid workflow: {e}", readable=True) from e
    return StoredWorkflow(name, spec, workflow_digest(spec))


__all__ = ["WorkflowFileError", "load_workflow_file"]
