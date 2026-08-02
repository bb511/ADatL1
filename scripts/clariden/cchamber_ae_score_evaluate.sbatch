#!/usr/bin/env bash
#SBATCH --job-name=cch-ae-score-eval
#SBATCH --account=a0166
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=4
#SBATCH --mem=440G
#SBATCH --array=0-11%8
#SBATCH --output=/iopsstor/scratch/cscs/vjimenez/adatl1/logs/slurm/%x-%A_%a.out

set -euo pipefail

readonly REPOSITORY_ROOT=/users/vjimenez/adatl1
readonly SCRATCH_ROOT=/iopsstor/scratch/cscs/vjimenez/adatl1
readonly AUDIT_ROOT="${SCRATCH_ROOT}/audits/cchamber_real_20260801_3789655_candidate_rank_3789655"
readonly OUTPUT_ROOT="${SCRATCH_ROOT}/audits/cchamber_ae_score_audit_20260802"
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
    "${UV_BIN}" run --frozen --no-sync python scripts/cchamber_ae_score_audit.py \
      evaluate-outcomes \
      --audit-root "${AUDIT_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --trajectory-index "${trajectory_index}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
exit "${status}"
