"""Re-export of the OCR readers in ``womblex.providers``, kept for one release."""

from womblex.providers.bedrock.ocr import MistralOCRReader, get_mistral_reader
from womblex.providers.ollama.ocr import OLLAMA_PROMPTS, OllamaOCRReader, get_ollama_reader

__all__ = [
    "OLLAMA_PROMPTS",
    "MistralOCRReader",
    "OllamaOCRReader",
    "get_mistral_reader",
    "get_ollama_reader",
]
