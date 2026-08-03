#!/usr/bin/env bash
#SBATCH --account=a0166
#SBATCH --partition=debug
#SBATCH --time=00:20:00
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=4
#SBATCH --mem=440G

set -euo pipefail

readonly FROZEN_ROOT=/iopsstor/scratch/cscs/vjimenez/adatl1/worktrees/cchamber_vae_multiscore_9b6f14b
readonly SCRATCH_ROOT=/iopsstor/scratch/cscs/vjimenez/adatl1
readonly CAMPAIGN_ROOT=${SCRATCH_ROOT}/campaigns/cchamber_vae_multiscore_20260803_auto
readonly UV_BIN=${SCRATCH_ROOT}/tools/uv-0.11.32/uv
: "${START_INDEX:?START_INDEX is required}"
: "${END_INDEX:?END_INDEX is required}"

export UV_PROJECT_ENVIRONMENT=${SCRATCH_ROOT}/.venv-clariden
export UV_CACHE_DIR=${SCRATCH_ROOT}/uv-cache-clariden-arm64
export UV_PYTHON_INSTALL_DIR=${SCRATCH_ROOT}/python
export UV_MANAGED_PYTHON=1
export HYDRA_FULL_ERROR=1

cd "${FROZEN_ROOT}"
pids=()
for trajectory_index in $(seq "${START_INDEX}" "${END_INDEX}"); do
  srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=72 --gpus-per-node=1 --mem=110G \
    "${UV_BIN}" run --frozen --no-sync python \
    scripts/cchamber_vae_multiscore_campaign.py train \
    --root "${CAMPAIGN_ROOT}" --trajectory-index "${trajectory_index}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
exit "${status}"
