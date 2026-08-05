#!/usr/bin/env bash
#SBATCH --job-name=phys-bgpair-restart-canary
#SBATCH --account=a0166
#SBATCH --partition=debug
#SBATCH --time=00:05:00
#SBATCH --requeue
# Do not use B:: the production path relies on Slurm signalling the srun child.
#SBATCH --signal=USR1@120
# JobFileAppend is disabled cluster-wide, so this is part of the tested contract.
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --output=/iopsstor/scratch/cscs/vjimenez/adatl1/logs/slurm/%x-%j.out

set -euo pipefail

readonly SCRATCH_ROOT=/iopsstor/scratch/cscs/vjimenez/adatl1
: "${PHYSICS_BGPAIR_REPOSITORY_ROOT:?set the immutable execution worktree}"
: "${PHYSICS_BGPAIR_RESTART_CANARY_ROOT:?set a fresh isolated canary root}"
: "${PHYSICS_BGPAIR_EXPECTED_COMMIT:?set the execution commit}"

export UV_PROJECT_ENVIRONMENT="${SCRATCH_ROOT}/.venv-clariden"
export UV_CACHE_DIR="${SCRATCH_ROOT}/uv-cache-clariden-arm64"
export UV_PYTHON_INSTALL_DIR="${SCRATCH_ROOT}/python"
export UV_MANAGED_PYTHON=1
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

readonly RESTART_COUNT="${SLURM_RESTART_COUNT:-0}"
readonly SCHEDULER_LOG="${SCRATCH_ROOT}/logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
printf 'CANARY_ATTEMPT_START restart=%s job=%s timestamp=%s\n' \
  "${RESTART_COUNT}" "${SLURM_JOB_ID}" "$(date --iso-8601=seconds)"

cd "${PHYSICS_BGPAIR_REPOSITORY_ROOT}"
srun "${UV_PROJECT_ENVIRONMENT}/bin/python" \
  scripts/physics_background_pairing_restart_canary.py \
  --root "${PHYSICS_BGPAIR_RESTART_CANARY_ROOT}" \
  --scheduler-log "${SCHEDULER_LOG}" \
  --expected-commit "${PHYSICS_BGPAIR_EXPECTED_COMMIT}"

printf 'CANARY_ATTEMPT_FINISH restart=%s job=%s timestamp=%s\n' \
  "${RESTART_COUNT}" "${SLURM_JOB_ID}" "$(date --iso-8601=seconds)"
