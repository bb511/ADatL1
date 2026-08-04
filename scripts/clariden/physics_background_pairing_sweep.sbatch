#!/usr/bin/env bash
#SBATCH --job-name=phys-bgpair-sweep
#SBATCH --account=a0166
#SBATCH --partition=xfer
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/iopsstor/scratch/cscs/vjimenez/adatl1/logs/slurm/%x-%j.out

set -euo pipefail
: "${STUDY_MODEL:?STUDY_MODEL must identify one of the six model studies}"
: "${CHUNK_TRIALS:=48}"
: "${CHUNK_JOBS:=48}"
: "${TIMEOUT_MIN:=720}"

readonly SCRATCH_ROOT=/iopsstor/scratch/cscs/vjimenez/adatl1
readonly REPOSITORY_ROOT="${PHYSICS_BGPAIR_REPOSITORY_ROOT:-${SCRATCH_ROOT}/worktrees/physics_background_shared_20260804}"
readonly CAMPAIGN_ROOT="${PHYSICS_BGPAIR_CAMPAIGN_ROOT:-${SCRATCH_ROOT}/campaigns/physics_background_shared_20260804}"
readonly UV_BIN="${SCRATCH_ROOT}/tools/uv-0.11.32/uv"

export UV_PROJECT_ENVIRONMENT="${SCRATCH_ROOT}/.venv-clariden"
export UV_CACHE_DIR="${SCRATCH_ROOT}/uv-cache-clariden-arm64"
export UV_PYTHON_INSTALL_DIR="${SCRATCH_ROOT}/python"
export UV_MANAGED_PYTHON=1
export HYDRA_FULL_ERROR=1

cd "${REPOSITORY_ROOT}"
"${UV_BIN}" run --frozen --no-sync python scripts/physics_background_pairing_campaign.py \
  --root "${CAMPAIGN_ROOT}" sweep --model "${STUDY_MODEL}" \
  --trials "${CHUNK_TRIALS}" --jobs "${CHUNK_JOBS}" \
  --timeout-min "${TIMEOUT_MIN}"
