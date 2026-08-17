#!/usr/bin/env bash
#SBATCH --job-name=phys-bgpair-confirm-canary
#SBATCH --account=a0166
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --output=/iopsstor/scratch/cscs/vjimenez/adatl1/logs/slurm/%x-%A_%a.out

set -euo pipefail

readonly SCRATCH_ROOT=/iopsstor/scratch/cscs/vjimenez/adatl1
readonly REPOSITORY_ROOT="${PHYSICS_BGPAIR_REPOSITORY_ROOT:?missing execution repository}"
readonly CAMPAIGN_ROOT="${PHYSICS_BGPAIR_CAMPAIGN_ROOT:?missing confirmation root}"
readonly UV_ENVIRONMENT="${SCRATCH_ROOT}/.venv-clariden"
readonly MODELS=(ae vae dsae dsvae svdd realnvp)
readonly MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"

export UV_PROJECT_ENVIRONMENT="${UV_ENVIRONMENT}"
export UV_CACHE_DIR="${SCRATCH_ROOT}/uv-cache-clariden-arm64"
export UV_PYTHON_INSTALL_DIR="${SCRATCH_ROOT}/python"
export UV_MANAGED_PYTHON=1
export HYDRA_FULL_ERROR=1
export MLFLOW_DISABLE_TELEMETRY=true
export WANDB_MODE=offline
export PROJECT_ROOT="${REPOSITORY_ROOT}"
export LOG_DIR="${CAMPAIGN_ROOT}/canaries/runtime/logs"
export OUTPUT_DIR="${CAMPAIGN_ROOT}/canaries/runtime/outputs"
export CHECKPOINT_DIR="${CAMPAIGN_ROOT}/canaries/runtime/checkpoints"

cd "${REPOSITORY_ROOT}"
srun "${UV_ENVIRONMENT}/bin/python" \
  scripts/physics_background_pairing_confirmation_campaign.py \
  --root "${CAMPAIGN_ROOT}" \
  --execution-repository "${REPOSITORY_ROOT}" \
  run-canary --model "${MODEL}"
