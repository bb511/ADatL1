#!/usr/bin/env bash
#SBATCH --job-name=cch-theorem-canary
#SBATCH --account=a0166
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --output=/iopsstor/scratch/cscs/vjimenez/adatl1/logs/slurm/%x-%j.out

set -euo pipefail

readonly REPOSITORY_ROOT=/users/vjimenez/adatl1
readonly SCRATCH_ROOT=/iopsstor/scratch/cscs/vjimenez/adatl1
readonly AUDIT_ROOT="${SCRATCH_ROOT}/audits/cchamber_real_20260725_63b941a_candidate_rank_d1a3017"
readonly OUTPUT_ROOT="${SCRATCH_ROOT}/audits/cchamber_theorem_bridge_20260726/canary"
readonly UV_BIN="${SCRATCH_ROOT}/tools/uv-0.11.32/uv"

export UV_PROJECT_ENVIRONMENT="${SCRATCH_ROOT}/.venv-clariden"
export UV_CACHE_DIR="${SCRATCH_ROOT}/uv-cache-clariden-arm64"
export UV_PYTHON_INSTALL_DIR="${SCRATCH_ROOT}/python"
export UV_MANAGED_PYTHON=1
export HYDRA_FULL_ERROR=1

cd "${REPOSITORY_ROOT}"
"${UV_BIN}" run --frozen --no-sync python scripts/cchamber_theorem_bridge.py \
  extract-scores \
  --audit-root "${AUDIT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --trajectory-index 144
