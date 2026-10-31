"""Mistral Pixtral Large on AWS Bedrock, an OCR engine that produces page-level markdown.

The engine returns page text with reading order already resolved, so
layout analysis and region sorting are skipped downstream — see
``OCRPageResult.reading_order_native``. The local Ollama engine is in
``providers/ollama/ocr.py``.

- ``MistralOCRReader`` — Mistral's Pixtral Large VLM inferenced via
  **AWS Bedrock** using the Converse API
  (``bedrock-runtime`` ``converse``). Region is taken from
  ``AWS_REGION`` / ``AWS_DEFAULT_REGION`` (default ``us-east-1``); the
  model id defaults to ``mistral.pixtral-large-2502-v1:0``. AWS
  credentials are resolved by the standard boto3 chain (env vars,
  shared config, instance/role profile). Model access must be enabled
  in the Bedrock Model Access console first.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from womblex.ingest.interfaces.protocols import OCRPageResult
from womblex.providers._images import img_to_png_bytes

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mistral (Pixtral Large) via AWS Bedrock Converse API
# ---------------------------------------------------------------------------

# The transcription instruction sent alongside each page image. Asks for a
# faithful markdown transcription in reading order — Pixtral resolves layout
# itself, so the output feeds the reading-order-native path downstream.
_MISTRAL_OCR_PROMPT = (
    "Transcribe all text in this document image to markdown, verbatim and "
    "in natural reading order. Preserve headings, lists and tables. Do not "
    "summarise, translate, or add commentary — output only the transcription."
)


class MistralOCRReader:
    """OCR reader backed by Mistral Pixtral Large on AWS Bedrock.

    Invokes the Pixtral Large VLM through the ``bedrock-runtime``
    Converse API. The page image is sent as raw PNG bytes (the Converse
    API takes bytes directly — no base64 wrapping) with a transcription
    prompt; the model returns markdown text in reading order.

    ``OCRPageResult.reading_order_native`` is True so downstream
    strategies bypass layout-model region sorting.

    AWS credentials and region come from the standard boto3 resolution
    chain. Override the region with ``AWS_REGION`` / ``AWS_DEFAULT_REGION``
    (default ``us-east-1``) and the model with the ``model`` kwarg or the
    ``MISTRAL_OCR_MODEL_ID`` env var. Model access must be enabled in the
    Bedrock Model Access console before first use.
    """

    DEFAULT_MODEL_ID = "mistral.pixtral-large-2502-v1:0"
    DEFAULT_REGION = "us-east-1"
    #: `ping` is a one-token Converse request, billed: the model check warns before it.
    ping_is_billed = True

    def __init__(
        self,
        model: str | None = None,
        region: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        timeout_s: float = 120.0,
    ) -> None:
        self.model_id = (
            model or os.environ.get("MISTRAL_OCR_MODEL_ID") or self.DEFAULT_MODEL_ID
        )
        self.region = (
            region
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or self.DEFAULT_REGION
        )
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_s = timeout_s
        self._client: object | None = None  # lazy boto3 client

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        import boto3
        from botocore.config import Config

        self._client = boto3.client(
            "bedrock-runtime",
            region_name=self.region,
            config=Config(
                read_timeout=self.timeout_s,
                connect_timeout=min(self.timeout_s, 10.0),
                retries={"max_attempts": 3, "mode": "adaptive"},
            ),
        )
        logger.info(
            "Mistral OCR (Bedrock Pixtral) reader ready: model=%s region=%s",
            self.model_id, self.region,
        )

    @staticmethod
    def _extract_text(response: dict) -> str:
        """Concatenate text blocks from a Bedrock Converse response."""
        content = (
            response.get("output", {}).get("message", {}).get("content", [])
        )
        parts = [
            block["text"] for block in content
            if isinstance(block, dict) and "text" in block
        ]
        return "\n".join(part for part in parts if part).strip()

    def read_page(self, img: np.ndarray) -> OCRPageResult:
        """Send the page image to Bedrock and return markdown text."""
        if img is None or img.size == 0:
            return OCRPageResult(reading_order_native=True, confidence=0.0)

        self._ensure_client()
        assert self._client is not None

        messages = [
            {
                "role": "user",
                "content": [
                    {"text": _MISTRAL_OCR_PROMPT},
                    {
                        "image": {
                            "format": "png",
                            "source": {"bytes": img_to_png_bytes(img)},
                        }
                    },
                ],
            }
        ]

        try:
            response = self._client.converse(  # type: ignore[attr-defined]
                modelId=self.model_id,
                messages=messages,
                inferenceConfig={
                    "maxTokens": self.max_tokens,
                    "temperature": self.temperature,
                },
            )
        except Exception as exc:
            logger.warning("Mistral OCR (Bedrock) request failed: %s", exc)
            return OCRPageResult(reading_order_native=True, confidence=0.0)

        try:
            text = self._extract_text(response)
        except (KeyError, TypeError, AttributeError) as exc:
            logger.warning("Mistral OCR (Bedrock) malformed response: %s", exc)
            return OCRPageResult(reading_order_native=True, confidence=0.0)

        # VLMs don't emit per-token confidence; treat a non-empty response
        # as high-confidence at the page level. Empty output → 0.
        confidence = 1.0 if text.strip() else 0.0
        return OCRPageResult(
            regions=[],
            markdown=text,
            reading_order_native=True,
            confidence=confidence,
        )

    def ping(self) -> None:
        """One-token text request: credentials, region and model access, or raise.

        The pre-run model check calls this; ``read_page`` swallows request
        errors so a single bad page does not stop a batch.
        """
        self._ensure_client()
        assert self._client is not None
        self._client.converse(  # type: ignore[attr-defined]
            modelId=self.model_id,
            messages=[{"role": "user", "content": [{"text": "ping"}]}],
            inferenceConfig={"maxTokens": 1},
        )

    def close(self) -> None:
        # boto3 clients hold no long-lived connection to close explicitly.
        self._client = None


# ------------------------------------------------------------------
# Module-level caches
# ------------------------------------------------------------------

_mistral_readers: dict[str, MistralOCRReader] = {}


def get_mistral_reader(
    model: str | None = None,
    region: str | None = None,
) -> MistralOCRReader:
    """Return a cached Mistral-OCR (Bedrock) reader keyed on (model, region)."""
    key = f"{model or ''}|{region or ''}"
    if key not in _mistral_readers:
        _mistral_readers[key] = MistralOCRReader(model=model, region=region)
    return _mistral_readers[key]
