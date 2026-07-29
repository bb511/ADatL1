# syntax=docker/dockerfile:1.6

ARG PYTHON_IMAGE=python:3.10-slim-bookworm
ARG POETRY_VERSION=2.3.1

FROM ${PYTHON_IMAGE} AS builder
ARG POETRY_VERSION

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:{PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates openssh-client \
    && rm -rf /var/lib/apt/lists/*

RUN python -m  venv /opt/venv \
    && pip install --upgrade  pip setuptools wheel \
    && pip install "poetry==${POETRY_VERSION}"

WORKDIR /workspace
COPY pyproject.toml poetry.lock ./

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/pypoetry \
    poetry install --only main --no-root --no-ansi

FROM ${PYTHON_IMAGE} AS runtime

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH} \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    bash ca-certificates libgomp1 procps tini \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /workspace
COPY . .

RUN bash -n scripts/physics/runae.sh \
    && python -m compileall -q src

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash", "scripts/physics/runae.sh"]