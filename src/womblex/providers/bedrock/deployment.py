"""Amazon Bedrock, which serves the Mistral Pixtral OCR model.

Bedrock has no free reachability call, so its check is the reader's own
one-token Converse request, which is billed: the pre-run check warns before it.
Credentials and region come from the standard boto3 chain.
"""

from __future__ import annotations

from womblex.utils.deployment_registry import Deployment, register

NAME = "bedrock"


def _check(model: str) -> None:
    from womblex.providers.bedrock.ocr import MistralOCRReader

    MistralOCRReader(model=model).ping()


register(Deployment(
    name=NAME,
    check=_check,
    needs_network=True,
    billed=True,
    env_optional=("AWS_REGION", "AWS_DEFAULT_REGION", "MISTRAL_OCR_MODEL_ID"),
))
