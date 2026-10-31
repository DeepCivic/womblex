"""A multimodal LLM served by a local Ollama endpoint, an OCR engine that produces page-level markdown.

Reads ``OLLAMA_BASE_URL`` (default ``http://localhost:11435/v1``). The model
returns page text with reading order resolved, so layout analysis is skipped
downstream — see ``OCRPageResult.reading_order_native``. Retained for
fully-local runs.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import TYPE_CHECKING

import numpy as np

from womblex.ingest.interfaces.protocols import OCRPageResult
from womblex.providers._images import img_to_png_bytes

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)


def _img_to_data_url(img: np.ndarray) -> str:
    """Encode an RGB numpy image as a base64 PNG data URL (for Ollama)."""
    b64 = base64.b64encode(img_to_png_bytes(img)).decode("ascii")
    return f"data:image/png;base64,{b64}"


# Documented Ollama VLM prompts. "free" returns plain transcription; the
# markdown variant asks the model to preserve document structure.
OLLAMA_PROMPTS: dict[str, str] = {
    "free": "Transcribe all text in this image verbatim, preserving reading order.",
    "markdown": "Convert this document image to markdown, preserving headings, "
                "lists and tables in reading order.",
}


class OllamaOCRReader:
    """OCR reader backed by a multimodal LLM via a local Ollama endpoint.

    Default target is a local Ollama instance at
    ``http://localhost:11435/v1``. Override with ``OLLAMA_BASE_URL``.

    The model returns markdown/prose with native reading order, so
    ``OCRPageResult.reading_order_native`` is set to True and downstream
    strategies bypass layout-model region sorting.
    """

    DEFAULT_BASE_URL = "http://localhost:11435/v1"
    DEFAULT_MODEL = "llama3.2-vision"
    DEFAULT_PROMPT_KEY = "free"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        prompt: str | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        self.model = model or self.DEFAULT_MODEL
        env_base = os.environ.get("OLLAMA_BASE_URL")
        resolved_base = base_url or env_base or self.DEFAULT_BASE_URL
        self.base_url = resolved_base.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url = f"{self.base_url}/v1"
        self.prompt = prompt or OLLAMA_PROMPTS[self.DEFAULT_PROMPT_KEY]
        self.timeout_s = timeout_s
        self._client: httpx.Client | None = None  # lazy

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        import httpx

        self._client = httpx.Client(timeout=self.timeout_s)
        logger.info(
            "Ollama OCR reader ready: model=%s base_url=%s",
            self.model, self.base_url,
        )

    def read_page(self, img: np.ndarray) -> OCRPageResult:
        """Send the page image to the LLM and return markdown text."""
        if img is None or img.size == 0:
            return OCRPageResult(reading_order_native=True, confidence=0.0)

        self._ensure_client()
        assert self._client is not None

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": _img_to_data_url(img)},
                        },
                    ],
                }
            ],
            "temperature": 0.0,
            "stream": False,
        }

        url = f"{self.base_url}/chat/completions"
        try:
            resp = self._client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("Ollama OCR request failed: %s", exc)
            return OCRPageResult(reading_order_native=True, confidence=0.0)

        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            logger.warning("Ollama OCR malformed response: %s", exc)
            return OCRPageResult(reading_order_native=True, confidence=0.0)

        # LLMs don't emit per-token confidence; treat a non-empty response
        # as high-confidence at the page level. Empty output → 0.
        confidence = 1.0 if text.strip() else 0.0
        return OCRPageResult(
            regions=[],
            markdown=text,
            reading_order_native=True,
            confidence=confidence,
        )

    def ping(self) -> None:
        """Raise unless the server answers and has the model pulled (free, no inference)."""
        self._ensure_client()
        assert self._client is not None
        resp = self._client.get(f"{self.base_url}/models")
        resp.raise_for_status()
        served = {m.get("id", "") for m in resp.json().get("data", [])}
        if not served & {self.model, f"{self.model}:latest"}:
            raise RuntimeError(
                f"Ollama at {self.base_url} does not serve {self.model!r} "
                f"(has: {sorted(served)}); `ollama pull {self.model}`"
            )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


_ollama_readers: dict[str, OllamaOCRReader] = {}


def get_ollama_reader(
    model: str | None = None,
    base_url: str | None = None,
    prompt: str | None = None,
) -> OllamaOCRReader:
    """Return a cached Ollama OCR reader keyed on (model, base_url, prompt)."""
    key = f"{model or ''}|{base_url or ''}|{prompt or ''}"
    if key not in _ollama_readers:
        _ollama_readers[key] = OllamaOCRReader(
            model=model, base_url=base_url, prompt=prompt,
        )
    return _ollama_readers[key]
