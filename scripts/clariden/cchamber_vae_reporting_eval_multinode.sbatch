#!/usr/bin/env bash
#SBATCH --account=a0166
#SBATCH --partition=debug
#SBATCH --nodes=4
#SBATCH --time=01:30:00
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=4
#SBATCH --mem=440G

set -euo pipefail

readonly REPOSITORY=/users/vjimenez/adatl1
readonly SCRATCH_ROOT=/iopsstor/scratch/cscs/vjimenez/adatl1
readonly CAMPAIGN_ROOT=${SCRATCH_ROOT}/campaigns/cchamber_vae_reporting_20260803_auto
readonly UV_BIN=${SCRATCH_ROOT}/tools/uv-0.11.32/uv
: "${N_TRAJECTORIES:?N_TRAJECTORIES is required}"

export UV_PROJECT_ENVIRONMENT=${SCRATCH_ROOT}/.venv-clariden
export UV_CACHE_DIR=${SCRATCH_ROOT}/uv-cache-clariden-arm64
export UV_PYTHON_INSTALL_DIR=${SCRATCH_ROOT}/python
export UV_MANAGED_PYTHON=1
export HYDRA_FULL_ERROR=1

cd "${REPOSITORY}"
pids=()
for worker in $(seq 0 15); do
  srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=72 --gpus-per-node=1 --mem=110G \
    bash -c '
      set -euo pipefail
      worker=$1
      n_trajectories=$2
      for ((trajectory_index = worker; trajectory_index < n_trajectories; trajectory_index += 16)); do
        "$3" run --frozen --no-sync python \
          scripts/cchamber_vae_reporting_campaign.py evaluate \
          --root "$4" --trajectory-index "${trajectory_index}"
      done
    ' _ "${worker}" "${N_TRAJECTORIES}" "${UV_BIN}" "${CAMPAIGN_ROOT}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
exit "${status}"
