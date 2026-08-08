#!/usr/bin/env bash
#SBATCH --job-name=phys-bgpair-sweep
#SBATCH --account=a0166
# The project environment is built for the ARM GPU nodes; xfer nodes are x86-64.
# This account has normal QOS only. One allocation packs four independent trials
# per node and the Hydra/Optuna controller runs in the enclosing batch step.
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=384G
#SBATCH --output=/iopsstor/scratch/cscs/vjimenez/adatl1/logs/slurm/%x-%j.out

set -euo pipefail
: "${STUDY_MODEL:?STUDY_MODEL must identify one of the six model studies}"
: "${CHUNK_TRIALS:=48}"
: "${CHUNK_JOBS:=48}"
: "${TIMEOUT_MIN:=720}"
: "${CANARY_MODE:=0}"

readonly SCRATCH_ROOT=/iopsstor/scratch/cscs/vjimenez/adatl1
: "${PHYSICS_BGPAIR_REPOSITORY_ROOT:?PHYSICS_BGPAIR_REPOSITORY_ROOT is required}"
: "${PHYSICS_BGPAIR_CAMPAIGN_ROOT:?PHYSICS_BGPAIR_CAMPAIGN_ROOT is required}"
readonly REPOSITORY_ROOT="${PHYSICS_BGPAIR_REPOSITORY_ROOT}"
readonly CAMPAIGN_ROOT="${PHYSICS_BGPAIR_CAMPAIGN_ROOT}"
readonly UV_BIN="${SCRATCH_ROOT}/tools/uv-0.11.32/uv"

export UV_PROJECT_ENVIRONMENT="${SCRATCH_ROOT}/.venv-clariden"
export UV_CACHE_DIR="${SCRATCH_ROOT}/uv-cache-clariden-arm64"
export UV_PYTHON_INSTALL_DIR="${SCRATCH_ROOT}/python"
export UV_MANAGED_PYTHON=1
export HYDRA_FULL_ERROR=1
export MLFLOW_DISABLE_TELEMETRY=true

cd "${REPOSITORY_ROOT}"
if [[ "${CANARY_MODE}" == "1" ]]; then
  : "${CANARY_ATTEMPT:?CANARY_ATTEMPT is required for production canaries}"
  "${UV_BIN}" run --frozen --no-sync python scripts/physics_background_pairing_campaign.py \
    --root "${CAMPAIGN_ROOT}" production-canary --model "${STUDY_MODEL}" \
    --attempt "${CANARY_ATTEMPT}"
else
  "${UV_BIN}" run --frozen --no-sync python scripts/physics_background_pairing_campaign.py \
    --root "${CAMPAIGN_ROOT}" sweep --model "${STUDY_MODEL}" \
    --trials "${CHUNK_TRIALS}" --jobs "${CHUNK_JOBS}" \
    --timeout-min "${TIMEOUT_MIN}"
fi
