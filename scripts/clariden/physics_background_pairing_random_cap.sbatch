#!/usr/bin/env bash
#SBATCH --job-name=phys-bgpair-random-cap
#SBATCH --account=a0166
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --open-mode=append
#SBATCH --output=/iopsstor/scratch/cscs/vjimenez/adatl1/logs/slurm/%x-%A_%a.out

set -euo pipefail

readonly SCRATCH_ROOT=/iopsstor/scratch/cscs/vjimenez/adatl1
readonly REPOSITORY="${PHYSICS_BGPAIR_REPOSITORY_ROOT:-/users/vjimenez/adatl1}"
readonly CAMPAIGN_ROOT="${PHYSICS_BGPAIR_CAMPAIGN_ROOT:?Set PHYSICS_BGPAIR_CAMPAIGN_ROOT}"
readonly EVALUATION_MODE="${PHYSICS_BGPAIR_RANDOM_CAP_MODE:?Set discovery or confirmation mode}"
export UV_PROJECT_ENVIRONMENT="${SCRATCH_ROOT}/.venv-clariden"
export UV_CACHE_DIR="${SCRATCH_ROOT}/uv-cache-clariden-arm64"
export UV_PYTHON_INSTALL_DIR="${SCRATCH_ROOT}/python"
export UV_MANAGED_PYTHON=1
export HYDRA_FULL_ERROR=1
export MLFLOW_DISABLE_TELEMETRY=true

cd "${REPOSITORY}"
srun "${UV_PROJECT_ENVIRONMENT}/bin/python" \
  scripts/physics_background_pairing_random_cap_evaluation.py \
  --root "${CAMPAIGN_ROOT}" evaluate \
  --mode "${EVALUATION_MODE}" \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --attempt-id "slurm-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}-${SLURM_RESTART_COUNT:-0}"
