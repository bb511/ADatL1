#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

uv run python src/train.py experiment=synthetic/vae_smoke "$@"
