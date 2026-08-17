# Womblex container image.
#
# Bundles the extraction pipeline + the [cloud] extra (object-storage staging
# and the Postgres job queue) so the same image serves `womblex run`, `worker`,
# `enqueue`, and the per-stage commands. Isaacus enrichment/embeddings and the
# Bedrock VLM OCR engine are in the base install (core deps), so no extra is
# needed for them — just a key / `ISAACUS_SAGEMAKER_ENDPOINTS` / an OCR engine
# choice at runtime. Override the installed extras at build time, e.g.
# --build-arg EXTRAS="cloud,ui".
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # The bundled models/ dir lives at the image root, not as a sibling of the
    # installed package, so point the resolver at it explicitly.
    WOMBLEX_MODELS_DIR=/app/models

# OpenCV (headless) + PyMuPDF need libglib/libGL at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

ARG EXTRAS=cloud
RUN pip install ".[${EXTRAS}]"

# Default to the CLI; compose / `docker run` override the subcommand.
ENTRYPOINT ["womblex"]
CMD ["--help"]
