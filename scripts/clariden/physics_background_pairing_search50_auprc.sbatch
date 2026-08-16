#!/usr/bin/env bash
#SBATCH --job-name=phys-bgpair-search50-auprc
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
readonly ORCHESTRATOR_REPOSITORY="${PHYSICS_BGPAIR_ORCHESTRATOR_REPOSITORY:-/users/vjimenez/adatl1}"
readonly CAMPAIGN_ROOT="${PHYSICS_BGPAIR_CAMPAIGN_ROOT:-${SCRATCH_ROOT}/campaigns/physics_background_nsgaii_05b649c}"
export UV_PROJECT_ENVIRONMENT="${SCRATCH_ROOT}/.venv-clariden"
export UV_CACHE_DIR="${SCRATCH_ROOT}/uv-cache-clariden-arm64"
export UV_PYTHON_INSTALL_DIR="${SCRATCH_ROOT}/python"
export UV_MANAGED_PYTHON=1
export HYDRA_FULL_ERROR=1
export MLFLOW_DISABLE_TELEMETRY=true

cd "${ORCHESTRATOR_REPOSITORY}"
srun "${UV_PROJECT_ENVIRONMENT}/bin/python" scripts/physics_background_pairing_search50_auprc.py \
  --root "${CAMPAIGN_ROOT}" evaluate \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --attempt-id "slurm-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}-${SLURM_RESTART_COUNT:-0}"
