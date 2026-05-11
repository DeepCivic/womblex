"""LLM-based OCR engines that produce page-level markdown.

These engines use multimodal LLMs (DeepSeek-OCR, etc.) served via
OpenAI-compatible HTTP APIs. They return page text with reading order
already resolved, so layout analysis and region sorting are skipped
downstream — see ``OCRPageResult.reading_order_native``.

The default ``DeepSeekOCRReader`` targets a local Ollama instance
(e.g. via the Alpaca GTK frontend) at ``http://localhost:11435``.
Override with the ``OLLAMA_BASE_URL`` env var.
"""

from __future__ import annotations

import base64
import io
import logging
import os

import numpy as np

from womblex.ingest.interfaces.protocols import OCRPageResult

logger = logging.getLogger(__name__)


# DeepSeek-OCR's documented prompts (see model card).
# "Free OCR." returns plain transcription. The grounding variants embed
# bbox tokens in the output for layout-aware downstream parsing.
DEEPSEEK_PROMPTS: dict[str, str] = {
    "free": "<image>\nFree OCR.",
    "markdown": "<image>\n<|grounding|>Convert the document to markdown.",
    "grounding": "<image>\n<|grounding|>OCR this image.",
    "figure": "<image>\nParse the figure.",
}


def _img_to_data_url(img: np.ndarray) -> str:
    """Encode an RGB numpy image as a base64 PNG data URL."""
    from PIL import Image

    if img.ndim == 2:
        pil = Image.fromarray(img, mode="L").convert("RGB")
    else:
        pil = Image.fromarray(img[:, :, :3], mode="RGB")

    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


class DeepSeekOCRReader:
    """OCR reader backed by DeepSeek-OCR via an OpenAI-compatible endpoint.

    Default target is Alpaca's managed Ollama instance at
    ``http://localhost:11435/v1``. Override with ``OLLAMA_BASE_URL``.

    The model returns markdown/prose with native reading order, so
    ``OCRPageResult.reading_order_native`` is set to True and downstream
    strategies bypass YOLO layout sorting.
    """

    DEFAULT_BASE_URL = "http://localhost:11435/v1"
    DEFAULT_MODEL = "deepseek-ocr:3b"
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
        self.prompt = prompt or DEEPSEEK_PROMPTS[self.DEFAULT_PROMPT_KEY]
        self.timeout_s = timeout_s
        self._client = None  # lazy httpx.Client

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        import httpx

        self._client = httpx.Client(timeout=self.timeout_s)
        logger.info(
            "DeepSeek-OCR reader ready: model=%s base_url=%s",
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
            logger.warning("DeepSeek-OCR request failed: %s", exc)
            return OCRPageResult(reading_order_native=True, confidence=0.0)

        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            logger.warning("DeepSeek-OCR malformed response: %s", exc)
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

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


# ------------------------------------------------------------------
# Module-level cache
# ------------------------------------------------------------------

_deepseek_readers: dict[str, DeepSeekOCRReader] = {}


def get_deepseek_reader(
    model: str | None = None,
    base_url: str | None = None,
    prompt: str | None = None,
) -> DeepSeekOCRReader:
    """Return a cached DeepSeek-OCR reader keyed on (model, base_url, prompt)."""
    key = f"{model or ''}|{base_url or ''}|{prompt or ''}"
    if key not in _deepseek_readers:
        _deepseek_readers[key] = DeepSeekOCRReader(
            model=model, base_url=base_url, prompt=prompt,
        )
    return _deepseek_readers[key]
