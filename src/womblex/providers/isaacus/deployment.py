"""The two Isaacus deployments: the hosted API and a private SageMaker endpoint.

Both serve the Kanon-2 models through one SDK, so one adapter per slot
(`enrichment.py`, `embedding.py`) is served by either. The hosted API serves
every model; SageMaker serves what `ISAACUS_SAGEMAKER_ENDPOINTS` declares.
"""

from __future__ import annotations

from womblex.providers.isaacus import client
from womblex.providers.isaacus.client import API_KEY_ENV, ENDPOINTS_ENV, endpoints_from_env, is_model_served
from womblex.utils.deployment_registry import Deployment, client_for, register
from womblex.utils.model_registry import SLOT_EMBEDDING

API = "isaacus-api"
SAGEMAKER = "isaacus-sagemaker"

_PROBE_SENTENCE = "The applicant, Jane Citizen, signed the declaration on 5 March 2024."


def smoke_call(client: object, slot: str, model: str) -> None:
    """One minimal billed request to *model* through *client*; raises unless it answers sensibly.

    The client is the process's shared one, so it is left open.
    """
    if slot == SLOT_EMBEDDING:
        resp = client.embeddings.create(  # type: ignore[attr-defined]
            model=model, texts=[_PROBE_SENTENCE], task="retrieval/document",
        )
        if len(resp.embeddings) != 1 or not resp.embeddings[0].embedding:
            raise RuntimeError("returned no embedding for the probe sentence")
    else:
        resp = client.enrichments.create(model=model, texts=[_PROBE_SENTENCE])  # type: ignore[attr-defined]
        if len(resp.results) != 1:
            raise RuntimeError("returned no enrichment for the probe sentence")


register(Deployment(
    name=API,
    check=lambda models: client.check_api_reachable(),
    needs_network=True,
    billed=True,
    env=((API_KEY_ENV,),),
    env_optional=("ISAACUS_BASE_URL",),
    smoke=lambda slot, model: smoke_call(client_for(API, [model]), slot, model),
    make_client=lambda models: client.make_api_client(),
))

register(Deployment(
    name=SAGEMAKER,
    check=lambda models: client.check_sagemaker_reachable(*models),
    needs_network=True,
    billed=True,
    env=((ENDPOINTS_ENV,),),
    env_optional=("ISAACUS_SAGEMAKER_REGION", "ISAACUS_SAGEMAKER_PROFILE"),
    serves=lambda model: is_model_served(model, endpoints_from_env()),
    smoke=lambda slot, model: smoke_call(client_for(SAGEMAKER, [model]), slot, model),
    make_client=lambda models: client.make_sagemaker_client(endpoints_from_env(), models),
))


def default_deployment() -> str:
    """The deployment a model is served by with nothing chosen: SageMaker when its endpoints are declared, else hosted."""
    return SAGEMAKER if endpoints_from_env() else API
