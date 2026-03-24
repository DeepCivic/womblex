"""PII cleaning pipeline stage.

Detects and replaces personally identifiable information using
regex pattern recognisers (Presidio-style) with Sentence Transformers
context validation.

Install the required extras before use:
    pip install womblex[pii]
"""

from womblex.pii.cleaner import PIICleaner

__all__ = ["PIICleaner"]
