#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."
uv sync --group dev
