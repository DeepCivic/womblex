"""Re-export of the OCR readers in ``womblex.providers``; deprecated, removed in 0.7.0."""

import warnings

from womblex.providers.bedrock.ocr import MistralOCRReader, get_mistral_reader
from womblex.providers.ollama.ocr import OLLAMA_PROMPTS, OllamaOCRReader, get_ollama_reader

__all__ = [
    "OLLAMA_PROMPTS",
    "MistralOCRReader",
    "OllamaOCRReader",
    "get_mistral_reader",
    "get_ollama_reader",
]

warnings.warn(
    "womblex.ingest.llm_ocr is deprecated and will be removed in 0.7.0; import from "
    "womblex.providers.bedrock.ocr (MistralOCRReader, get_mistral_reader) or "
    "womblex.providers.ollama.ocr (OllamaOCRReader, get_ollama_reader, OLLAMA_PROMPTS)",
    DeprecationWarning,
    stacklevel=2,
)
