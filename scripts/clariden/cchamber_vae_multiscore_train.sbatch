#!/usr/bin/env bash
#SBATCH --job-name=cch-vae-multi-train
#SBATCH --account=a0166
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=4
#SBATCH --mem=440G
#SBATCH --array=0-11%12
#SBATCH --output=/iopsstor/scratch/cscs/vjimenez/adatl1/logs/slurm/%x-%A_%a.out

set -euo pipefail

readonly REPOSITORY_ROOT=/users/vjimenez/adatl1
readonly SCRATCH_ROOT=/iopsstor/scratch/cscs/vjimenez/adatl1
readonly CAMPAIGN_ROOT="${SCRATCH_ROOT}/campaigns/cchamber_vae_multiscore_20260803_auto"
readonly UV_BIN="${SCRATCH_ROOT}/tools/uv-0.11.32/uv"

export UV_PROJECT_ENVIRONMENT="${SCRATCH_ROOT}/.venv-clariden"
export UV_CACHE_DIR="${SCRATCH_ROOT}/uv-cache-clariden-arm64"
export UV_PYTHON_INSTALL_DIR="${SCRATCH_ROOT}/python"
export UV_MANAGED_PYTHON=1
export HYDRA_FULL_ERROR=1

cd "${REPOSITORY_ROOT}"
pids=()
for slot in 0 1 2 3; do
  trajectory_index=$((SLURM_ARRAY_TASK_ID * 4 + slot))
  srun \
    --exclusive \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=72 \
    --gpus-per-node=1 \
    --mem=110G \
    "${UV_BIN}" run --frozen --no-sync python scripts/cchamber_vae_multiscore_campaign.py \
      train --root "${CAMPAIGN_ROOT}" --trajectory-index "${trajectory_index}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
exit "${status}"
