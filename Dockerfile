# syntax=docker/dockerfile:1.6

ARG PYTHON_IMAGE=python:3.10-slim-bookworm
ARG POETRY_VERSION=2.3.1

############################
# builder
############################
FROM ${PYTHON_IMAGE} AS builder
ARG POETRY_VERSION

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git \
      ca-certificates \
      openssh-client \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv \
 && pip install --upgrade pip setuptools wheel \
 && pip install "poetry==${POETRY_VERSION}"

WORKDIR /deps
COPY pyproject.toml poetry.lock* /deps/

# If your lockfile expects CUDA wheels from the PyTorch index, uncomment and set appropriately:
# ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/pypoetry \
    poetry install --no-ansi --no-root

############################
# runtime
############################
FROM ${PYTHON_IMAGE} AS dev

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      bash \
      vim \
      less \
      procps \
      htop \
      tmux \
      openssh-client \
      git \
      rsync \
      curl \
      wget \
      tini \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /workspace
CMD ["bash"]
