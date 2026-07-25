#!/usr/bin/env bash
# Recreate the native Clariden uv/Python environment after a scratch cleanup.

set -euo pipefail

readonly REPOSITORY_ROOT=/users/vjimenez/adatl1
readonly SCRATCH_ROOT=/iopsstor/scratch/cscs/vjimenez/adatl1
readonly TOOL_DIR="${SCRATCH_ROOT}/tools/uv-0.11.32"
readonly PROJECT_ENVIRONMENT="${SCRATCH_ROOT}/.venv-clariden"
readonly REPOSITORY_ENVIRONMENT="${REPOSITORY_ROOT}/.venv"
SOURCE_UV="$(command -v uv)"
SOURCE_UVX="$(command -v uvx)"
readonly SOURCE_UV SOURCE_UVX

case "$(uname -m)" in
  aarch64) ;;
  *)
    echo "Clariden bootstrap requires aarch64; found $(uname -m)." >&2
    exit 1
    ;;
esac

case "$(file -b "${SOURCE_UV}")" in
  *"ARM aarch64"*) ;;
  *)
    echo "The uv on PATH is not an ARM64 executable: ${SOURCE_UV}" >&2
    exit 1
    ;;
esac

mkdir -p \
  "${TOOL_DIR}" \
  "${SCRATCH_ROOT}/python" \
  "${SCRATCH_ROOT}/uv-cache-clariden-arm64"
install -m 0755 "${SOURCE_UV}" "${TOOL_DIR}/uv"
install -m 0755 "${SOURCE_UVX}" "${TOOL_DIR}/uvx"

export UV_PYTHON_INSTALL_DIR="${SCRATCH_ROOT}/python"
export UV_CACHE_DIR="${SCRATCH_ROOT}/uv-cache-clariden-arm64"
export UV_PROJECT_ENVIRONMENT="${PROJECT_ENVIRONMENT}"
export UV_MANAGED_PYTHON=1

cd "${REPOSITORY_ROOT}"
"${TOOL_DIR}/uv" python install 3.10
"${TOOL_DIR}/uv" sync --locked --group dev
"${TOOL_DIR}/uv" lock --check

if [[ -L "${REPOSITORY_ENVIRONMENT}" || ! -e "${REPOSITORY_ENVIRONMENT}" ]]; then
  ln -sfn "${PROJECT_ENVIRONMENT}" "${REPOSITORY_ENVIRONMENT}"
else
  echo "Refusing to replace non-symlink ${REPOSITORY_ENVIRONMENT}." >&2
  exit 1
fi

"${TOOL_DIR}/uv" run --no-sync python - <<'PY'
import platform
import sys

assert sys.version_info[:2] == (3, 10), sys.version
assert platform.machine() == "aarch64", platform.machine()
print(f"Clariden environment ready: Python {sys.version.split()[0]} ({platform.machine()})")
PY
