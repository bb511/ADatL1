#!/usr/bin/env bash
#SBATCH --job-name=phys-bgpair-pilot
#SBATCH --account=a0166
#SBATCH --partition=debug
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --output=/iopsstor/scratch/cscs/vjimenez/adatl1/logs/slurm/%x-%j.out

set -euo pipefail

readonly SCRATCH_ROOT=/iopsstor/scratch/cscs/vjimenez/adatl1
readonly REPOSITORY_ROOT="${SCRATCH_ROOT}/worktrees/physics_background_pairing_20260803"
readonly CAMPAIGN_ROOT="${SCRATCH_ROOT}/campaigns/physics_background_pairing_20260803"
readonly UV_BIN="${SCRATCH_ROOT}/tools/uv-0.11.32/uv"
: "${PILOT_MODEL:=ae}"
: "${PILOT_SCORE:=residual_oas}"
: "${PILOT_METRIC:=cap_mapping}"
: "${PILOT_STRATEGY:=physics_summary}"

export UV_PROJECT_ENVIRONMENT="${SCRATCH_ROOT}/.venv-clariden"
export UV_CACHE_DIR="${SCRATCH_ROOT}/uv-cache-clariden-arm64"
export UV_PYTHON_INSTALL_DIR="${SCRATCH_ROOT}/python"
export UV_MANAGED_PYTHON=1
export HYDRA_FULL_ERROR=1

cd "${REPOSITORY_ROOT}"
"${UV_BIN}" run --frozen --no-sync python scripts/physics_background_pairing_campaign.py \
  --root "${CAMPAIGN_ROOT}" pilot --metric "${PILOT_METRIC}" \
  --model "${PILOT_MODEL}" --score "${PILOT_SCORE}" --strategy "${PILOT_STRATEGY}"
