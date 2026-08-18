# Womblex container image.
#
# Bundles the whole pipeline. Object-storage staging (fsspec + s3fs) and the
# Postgres job queue (psycopg) are now *core* dependencies, so the same image
# serves `womblex run`, `worker`, `enqueue`, and the per-stage commands against
# either a local dir or an `s3://` store with no extra install. Isaacus
# enrichment/embeddings and the Bedrock VLM OCR engine are core too — just a
# key / `ISAACUS_SAGEMAKER_ENDPOINTS` / an OCR engine choice at runtime. The
# only remaining extras are `[ui]` (the console) and `[dev]` (test/lint);
# override at build time, e.g. --build-arg EXTRAS="ui".
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

# `.` installs the core deps (S3 + queue included); no extra needed for a
# worker. Override to add the console/dev tools, e.g. --build-arg EXTRAS="ui".
ARG EXTRAS=""
RUN if [ -n "${EXTRAS}" ]; then pip install ".[${EXTRAS}]"; else pip install "."; fi

# Default to the CLI; compose / `docker run` override the subcommand.
ENTRYPOINT ["womblex"]
CMD ["--help"]
