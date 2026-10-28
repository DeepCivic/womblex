# Womblex container image.
#
# Bundles the whole pipeline. Object-storage staging (fsspec + s3fs) and the
# Postgres job queue (psycopg) are now *core* dependencies, so the same image
# serves `womblex run`, `worker`, `enqueue`, and the per-stage commands against
# either a local dir or an `s3://` store with no extra install. Isaacus
# enrichment/embeddings and the Bedrock VLM OCR engine are core too — just a
# key / `ISAACUS_SAGEMAKER_ENDPOINTS` / an OCR engine choice at runtime. The
# image installs `[api]` (the `womblex serve` service API), so the `api`
# compose service runs on the same image as the workers. The other extra is
# `[dev]` (test/lint); override at
# build time, e.g. --build-arg EXTRAS="api,dev".

# uv exports the lock at build time; it is mounted, never installed. A stage
# of its own so the pin is a FROM line that dependency updates can read.
FROM ghcr.io/astral-sh/uv:0.13.6@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0 AS uv

FROM python:3.12-slim@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159 AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # The large model artefacts live at the image root, not as a sibling of
    # the installed package, so name that root explicitly. It SUPPLEMENTS the
    # wheel-bundled womblex/_models/ (en_AU, kanon-2-tokenizer) rather than
    # replacing it: utils.models searches every root per artefact.
    WOMBLEX_MODELS_DIR=/app/models

# OpenCV (headless) needs libglib/libGL at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies install from `uv.lock`, hash-checked, rather than from a fresh
# resolve of pyproject.toml's ranges, so the image ships exactly the set CI
# tested and the dependency alerts name.
# Only the two files the export reads are copied first, so this layer stays
# cached until the lock changes. uv is mounted for the export, not installed.
# `.` installs the core deps (S3 + queue included); a worker needs nothing
# more. `[api]` is fastapi + uvicorn + python-multipart, for `womblex serve`.
# EXTRAS is comma-separated, e.g. "api,dev", or empty for the core alone.
ARG EXTRAS="api"
COPY pyproject.toml uv.lock /app/
RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    extras="" \
    && for e in $(echo "${EXTRAS}" | tr ',' ' '); do extras="${extras} --extra ${e}"; done \
    && uv export --frozen --no-dev ${extras} --no-emit-project -o /tmp/requirements.txt \
    && pip install --require-hashes -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

COPY . /app
RUN pip install --no-deps .

# The source commit, supplied rather than discovered: `.dockerignore` excludes
# `.git`, and a build context is a copy of a work tree rather than one, so a
# build that tried to read it would be reading nothing. Stamped into the
# *installed* package, which is the copy `store/build_info.py` asks — and
# after the install above, so the expensive dependency layer stays cached
# across commits while only this cheap layer changes per build.
ARG SOURCE_COMMIT=""
# Whether an unstamped build is a defect. The publish path sets it; a local
# a local build does not (those live in `docker-compose.local.override.yml`),
# and its image reports `unavailable` with a reason. A published image is a claim about what it holds, so one that cannot
# name its commit is a build failure rather than an honest answer.
ARG REQUIRE_STAMP=0
RUN if [ -n "${SOURCE_COMMIT}" ]; then \
        stamp="$(python -c 'import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec("womblex").origin).parent / "_build_stamp.py")')" \
        && printf 'COMMIT = "%s"\n' "${SOURCE_COMMIT}" > "${stamp}"; \
    elif [ "${REQUIRE_STAMP}" = "1" ]; then \
        echo "SOURCE_COMMIT is required when REQUIRE_STAMP=1" >&2 && exit 1; \
    fi

# The same fact as image metadata, readable with `docker inspect` without
# running the image. Standard OCI keys, so any registry or scanner reads them.
LABEL org.opencontainers.image.title="womblex" \
      org.opencontainers.image.source="https://github.com/DeepCivic/womblex" \
      org.opencontainers.image.revision="${SOURCE_COMMIT}"

# Default to the CLI; compose / `docker run` override the subcommand.
ENTRYPOINT ["womblex"]
CMD ["--help"]
