#!/usr/bin/env bash
#SBATCH --job-name=phys-bgpair-retrain
#SBATCH --account=a0166
#SBATCH --partition=normal
#SBATCH --time=12:00:00
# Requeue is reserved for Lightning's emergency USR1 checkpoint path. Normal 40-epoch
# boundaries exit successfully and continue in a newly submitted mapped array.
#SBATCH --requeue
# Do not use Slurm's B: prefix: it signals only the batch shell, not Lightning.
#SBATCH --signal=USR1@900
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --output=/iopsstor/scratch/cscs/vjimenez/adatl1/logs/slurm/%x-%A_%a.out

set -euo pipefail

readonly SCRATCH_ROOT=/iopsstor/scratch/cscs/vjimenez/adatl1
readonly REPOSITORY_ROOT="${PHYSICS_BGPAIR_REPOSITORY_ROOT:-${SCRATCH_ROOT}/worktrees/physics_background_shared_20260804}"
readonly CAMPAIGN_ROOT="${PHYSICS_BGPAIR_CAMPAIGN_ROOT:-${SCRATCH_ROOT}/campaigns/physics_background_shared_20260804}"
export UV_PROJECT_ENVIRONMENT="${SCRATCH_ROOT}/.venv-clariden"
export UV_CACHE_DIR="${SCRATCH_ROOT}/uv-cache-clariden-arm64"
export UV_PYTHON_INSTALL_DIR="${SCRATCH_ROOT}/python"
export UV_MANAGED_PYTHON=1
export HYDRA_FULL_ERROR=1
export MLFLOW_DISABLE_TELEMETRY=true

cd "${REPOSITORY_ROOT}"
readonly RETRAIN_TASK_MAP="${PHYSICS_BGPAIR_RETRAIN_TASK_MAP:?missing retrain task map}"
readonly RETRAIN_TASK_MAP_SHA256="${PHYSICS_BGPAIR_RETRAIN_TASK_MAP_SHA256:?missing task-map SHA-256}"
export PHYSICS_BGPAIR_RETRAIN_LOCAL_TASK_ID="${SLURM_ARRAY_TASK_ID}"
PHYSICS_BGPAIR_RETRAIN_INDEX="$(
  "${UV_PROJECT_ENVIRONMENT}/bin/python" scripts/physics_background_pairing_campaign.py \
    --root "${CAMPAIGN_ROOT}" resolve-retrain-index \
    --task-map "${RETRAIN_TASK_MAP}" \
    --task-map-sha256 "${RETRAIN_TASK_MAP_SHA256}" \
    --local-task-id "${SLURM_ARRAY_TASK_ID}"
)"
readonly PHYSICS_BGPAIR_RETRAIN_INDEX
export PHYSICS_BGPAIR_RETRAIN_INDEX
srun "${UV_PROJECT_ENVIRONMENT}/bin/python" scripts/physics_background_pairing_campaign.py \
  --root "${CAMPAIGN_ROOT}" retrain \
  --index "${PHYSICS_BGPAIR_RETRAIN_INDEX}" \
  --task-map "${RETRAIN_TASK_MAP}" \
  --task-map-sha256 "${RETRAIN_TASK_MAP_SHA256}"
