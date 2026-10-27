"""A local Ollama server, which serves the multimodal OCR model on this machine.

Free to call and, being local, off the network: the check lists the server's
models and raises unless it has the one asked for.
"""

from __future__ import annotations

from womblex.utils.deployment_registry import Deployment, register

NAME = "ollama"


def _check(model: str) -> None:
    from womblex.providers.ollama.ocr import OllamaOCRReader

    OllamaOCRReader(model=model).ping()


register(Deployment(
    name=NAME,
    check=_check,
    needs_network=False,
    billed=False,
    env_optional=("OLLAMA_BASE_URL",),
))
