# syntax=docker/dockerfile:1.6

ARG BASE_IMAGE=registry.cern.ch/ngt/pytorch:2.3.1
ARG POETRY_VERSION=2.3.1

FROM ${BASE_IMAGE} AS dev
USER root
ARG POETRY_VERSION

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /deps
COPY pyproject.toml poetry.lock* /deps/

# Install poetry into the conda environment
RUN pip install --upgrade pip setuptools wheel \
 && pip install "poetry==${POETRY_VERSION}"

# Install deps into the conda env (no venv)
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/pypoetry \
    poetry install --no-ansi --no-root

WORKDIR /workspace
CMD ["bash"]
