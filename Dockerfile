# syntax=docker/dockerfile:1.6

ARG PYTHON_VERSION=3.10.19
ARG POETRY_VERSION=2.3.1

############################
# 1) Build a venv with python
############################
FROM rockylinux:9 AS python-builder
USER root
ARG PYTHON_VERSION

RUN dnf -y --allowerasing install \
      gcc gcc-c++ make \
      curl tar xz \
      openssl-devel bzip2-devel libffi-devel zlib-devel \
      readline-devel sqlite-devel tk-devel xz-devel findutils \
 && dnf clean all

WORKDIR /tmp/python-build

RUN curl -fsSLO "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz" \
 && tar -xzf "Python-${PYTHON_VERSION}.tgz" \
 && cd "Python-${PYTHON_VERSION}" \
 && ./configure \
      --prefix="/opt/python/${PYTHON_VERSION}" \
      --enable-optimizations \
      --with-lto \
      --with-ensurepip=install \
 && make -j"$(nproc)" \
 && make install

RUN "/opt/python/${PYTHON_VERSION}/bin/python3.10" -m pip install --upgrade pip setuptools wheel

############################
# 2) Install the dependencies using poetry
############################
FROM registry.cern.ch/ngt/lxplus-like:9 AS deps-installer
USER root
ARG PYTHON_VERSION
ARG POETRY_VERSION

# Bring in Python
COPY --from=python-builder "/opt/python/${PYTHON_VERSION}" "/opt/python/${PYTHON_VERSION}"

ENV PATH="/opt/python/${PYTHON_VERSION}/bin:$PATH" \
    VIRTUAL_ENV=/venv \
    PATH="/venv/bin:/opt/python/${PYTHON_VERSION}/bin:$PATH" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_NO_INTERACTION=1

# Create venv + install Poetry
RUN python3.10 -m venv /venv \
 && pip install --upgrade pip setuptools wheel \
 && pip install "poetry==${POETRY_VERSION}"

WORKDIR /deps
COPY pyproject.toml poetry.lock* /deps/

# ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu126

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/pypoetry \
    poetry config virtualenvs.create false \
 && poetry install --no-ansi --no-root

############################
# 3) Instantiate env and mount code
############################
FROM registry.cern.ch/ngt/lxplus-like:9 AS dev
USER root
ARG PYTHON_VERSION

COPY --from=deps-installer "/opt/python/${PYTHON_VERSION}" "/opt/python/${PYTHON_VERSION}"
COPY --from=deps-installer /venv /venv

ENV VIRTUAL_ENV=/venv \
    PATH="/venv/bin:/opt/python/${PYTHON_VERSION}/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /workspace
CMD ["bash"]
