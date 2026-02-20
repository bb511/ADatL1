# syntax=docker/dockerfile:1.6

############################
# Base: OS + Python + venv
############################
FROM registry.cern.ch/ngt/lxplus-like:9 AS base

USER root

RUN dnf install -y python3.10 python3.10-pip python3.10-devel \
 && dnf clean all

USER 1000

ENV VIRTUAL_ENV=/venv \
    PATH=/venv/bin:$PATH \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.3 \
    POETRY_NO_INTERACTION=1

RUN python3.10 -m venv /venv \
 && pip install --upgrade pip setuptools wheel

############################
# Builder: install deps
############################
FROM base AS builder

RUN dnf install -y git gcc gcc-c++ make cmake \
 && dnf clean all

RUN pip install "poetry==${POETRY_VERSION}"

WORKDIR /deps

COPY pyproject.toml poetry.lock* /deps/

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/pypoetry \
    poetry config virtualenvs.create false \
 && poetry install --no-ansi --no-root

############################
# Dev runtime: deps only, code mounted at /workspace
############################
FROM base AS dev

# Copy the fully-built venv
COPY --from=builder /venv /venv

# Optional: runtime niceties (git for quick pulls, etc.)
RUN dnf install -y git \
 && dnf clean all

WORKDIR /workspace

# Default: interactive shell. In NGT you’ll usually override command anyway.
CMD ["bash"]
