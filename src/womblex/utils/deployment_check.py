"""Pre-run checks of the deployments the configured models are served by.

A deployment (`utils/deployment_registry.py`) is checked once however many slots
use it: at ``load`` its free check is called once with every model the run uses on
it, and only if that fails is each model checked alone, so a failure is attributed
to the model that caused it. At ``smoke`` each model is sent one billed request,
warned first. A reader that declares no deployment but has ``ping()`` is wrapped in
one built from its ping.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from womblex.utils.deployment_registry import deployment_for_pinging_model, resolve
from womblex.utils.model_registry import SLOT_EMBEDDING, SLOT_ENRICHMENT

if TYPE_CHECKING:
    from womblex.utils.model_check import CheckLevel, ModelUse, SlotCheck

logger = logging.getLogger(__name__)


def _result(use: ModelUse, level: CheckLevel, source: str, start: float, error: Exception | None) -> SlotCheck:
    from womblex.utils.model_check import SlotCheck

    return SlotCheck(
        slot=use.slot, name=use.name, scopes=use.scopes, level=level, ok=error is None, source=source,
        reason="" if error is None else f"{type(error).__name__}: {error}", seconds=time.monotonic() - start,
    )


def _free_errors(check: Any, models: Sequence[str]) -> dict[str, Exception | None]:
    """Each model's outcome of one free check; a failure is retried model by model to name the cause."""
    try:
        check(list(models))
        return dict.fromkeys(models)
    except Exception as exc:  # the SDK, boto3 and httpx raise their own types
        if len(models) == 1:
            return {models[0]: exc}
    errors: dict[str, Exception | None] = {}
    for model in models:
        try:
            check([model])
            errors[model] = None
        except Exception as exc:
            errors[model] = exc
    return errors


def check_deployed(uses: Sequence[ModelUse], level: CheckLevel) -> list[SlotCheck]:
    """One `SlotCheck` per use, in order, each checked through the deployment its `deployment` names."""
    from womblex.utils.model_check import SCOPE_EMBED, CheckLevel

    out: dict[int, SlotCheck] = {}
    groups: dict[str, list[int]] = {}
    for i, use in enumerate(uses):
        groups.setdefault(use.deployment, []).append(i)
    for name, indexes in groups.items():
        start = time.monotonic()
        try:
            deployment = resolve(name)
        except ValueError as exc:
            for i in indexes:
                out[i] = _result(uses[i], level, "", start, exc)
            continue
        if level is CheckLevel.SMOKE and deployment.smoke is not None:
            for i in indexes:
                use = uses[i]
                start = time.monotonic()
                logger.warning(
                    "model check (smoke): sending one billed request to model %r on deployment %r to verify it; "
                    "processing.models_check=load checks it without a billed request", use.name, deployment.name)
                slot = SLOT_EMBEDDING if SCOPE_EMBED in use.scopes else SLOT_ENRICHMENT
                try:
                    deployment.smoke(slot, use.name)
                    error: Exception | None = None
                except Exception as exc:
                    error = exc
                out[i] = _result(use, level, deployment.name, start, error)
            continue
        errors = _free_errors(deployment.check, list(dict.fromkeys(uses[i].name for i in indexes)))
        for i in indexes:
            out[i] = _result(uses[i], level, deployment.name, start, errors[uses[i].name])
    return [out[i] for i in range(len(uses))]


def check_pinging_model(slot: str, name: str, model: Any) -> None:
    """Check a built reader through the deployment its ``ping()`` implies, warning first when it is billed.

    The reader carries the run's engine options (region, model, endpoint), so its own
    ping is the check; raises what the ping raises.
    """
    deployment = deployment_for_pinging_model(slot, name, model)
    if deployment.billed:
        logger.warning("model check: sending one billed request to %s %r to verify it", slot, name)
    deployment.check([name])
