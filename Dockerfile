# syntax=docker/dockerfile:1.6

ARG PYTHON_IMAGE=python:3.10-slim-bookworm
ARG UV_VERSION=0.9.13

############################
# builder
############################
FROM ${PYTHON_IMAGE} AS builder
ARG UV_VERSION

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git \
      ca-certificates \
      openssh-client \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv \
 && pip install --upgrade pip setuptools wheel \
 && pip install "uv==${UV_VERSION}"

WORKDIR /deps
COPY pyproject.toml uv.lock .python-version /deps/

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

############################
# runtime
############################
FROM ${PYTHON_IMAGE} AS dev

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
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
