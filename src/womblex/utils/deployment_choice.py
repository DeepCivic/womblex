"""Which deployment serves each model a config names.

Each slot's config takes an optional ``deployment:`` (AI chunking's is
``chunking.chunking_deployment``). Left unset, a model is served as it always was:
by SageMaker when ``ISAACUS_SAGEMAKER_ENDPOINTS`` is set, else the hosted API, for
an Isaacus model; by its only deployment otherwise. A model its chosen deployment
does not serve is refused, never routed to another deployment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from womblex.utils.deployment_registry import ISAACUS_DEPLOYMENTS, deployments_for, resolve
from womblex.utils.model_registry import SLOT_EMBEDDING, SLOT_ENRICHMENT, SLOT_OCR
from womblex.utils.model_registry import resolve as resolve_model

if TYPE_CHECKING:
    from womblex.config import WomblexConfig

#: The scope (``utils/model_check.py``) that uses each slot's model; AI chunking's is ``chunk``.
_SCOPE = {SLOT_OCR: "extract", SLOT_ENRICHMENT: "enrich", SLOT_EMBEDDING: "embed", "chunking": "chunk"}


def pick_deployment(deployments: tuple[str, ...], requested: str | None = None) -> str:
    """*requested* if named, else the environment's Isaacus default where the model may use it, else the first."""
    if requested:
        return requested.strip().lower()
    from womblex.providers.isaacus.deployment import default_deployment

    default = default_deployment()
    return default if default in deployments else deployments[0]


def deployment_of(slot: str, model: str, requested: str | None = None) -> str | None:
    """The deployment serving the model registered as *model* in *slot*; ``None`` if it runs locally."""
    deployments = deployments_for(resolve_model(slot, model))
    return pick_deployment(deployments, requested) if deployments else None


def _choices(config: WomblexConfig) -> list[tuple[str, str, str, str | None, tuple[str, ...] | None]]:
    """``(slot, scope-slot key, model, requested, deployments)`` for each model *config* names.

    ``deployments`` is ``None`` for a model that is not registered (its own check reports that).
    """
    out: list[tuple[str, str, str, str | None, tuple[str, ...] | None]] = []
    ocr, enr, emb, chunk = config.extraction.ocr, config.enrichment, config.embedding, config.chunking
    for slot, model, requested in (
        (SLOT_OCR, ocr.engine, ocr.deployment),
        (SLOT_ENRICHMENT, enr.model, enr.deployment),
        (SLOT_EMBEDDING, emb.model, emb.deployment),
    ):
        try:
            declared: tuple[str, ...] | None = deployments_for(resolve_model(slot, model))
        except ValueError:
            declared = None
        out.append((slot, slot, model, requested, declared))
    if chunk.chunking_model or chunk.chunking_deployment:
        out.append((SLOT_ENRICHMENT, "chunking", chunk.chunking_model or "", chunk.chunking_deployment, ISAACUS_DEPLOYMENTS))
    return out


def choice_problems(config: WomblexConfig) -> list[tuple[str, str, str, str]]:
    """``(scope, slot, model, message)`` for each ``deployment:`` setting the registry cannot honour.

    Static: it reads registrations and not the environment, so a host that carries none of
    the workers' models or endpoints can still refuse before any row is written.
    """
    problems: list[tuple[str, str, str, str]] = []
    for slot, key, model, requested, declared in _choices(config):
        if not requested or declared is None:
            continue
        scope = _SCOPE[key]
        name = requested.strip().lower()
        if key == "chunking" and not model:
            message = "chunking.chunking_deployment needs chunking.chunking_model"
        elif not declared:
            message = f"{slot} model {model!r} runs locally and takes no deployment (got {requested!r})"
        else:
            try:
                resolve(name)
            except ValueError as exc:
                problems.append((scope, slot, model, str(exc)))
                continue
            if name in declared:
                continue
            message = (f"{slot} model {model!r} is not served by deployment {requested!r}; "
                       f"it is served by: {', '.join(declared)}")
        problems.append((scope, slot, model, message))
    return problems
