#!/usr/bin/env bash
#SBATCH --job-name=adatl1-cchamber-canary
#SBATCH --account=a0166
#SBATCH --partition=debug
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --output=/iopsstor/scratch/cscs/vjimenez/adatl1/logs/slurm/%x-%j.out

set -euo pipefail

readonly REPOSITORY_ROOT=/users/vjimenez/adatl1
readonly SCRATCH_ROOT=/iopsstor/scratch/cscs/vjimenez/adatl1
readonly JOB_ROOT="${SCRATCH_ROOT}/canaries/${SLURM_JOB_ID}/cchamber"
readonly UV_BIN="${SCRATCH_ROOT}/tools/uv-0.11.32/uv"

export PROJECT_ROOT="${REPOSITORY_ROOT}"
export DATA_DIR="${SCRATCH_ROOT}/data"
export RAW_DATA_DIR="${JOB_ROOT}/raw"
export LOG_DIR="${JOB_ROOT}/logs"
export OUTPUT_DIR="${JOB_ROOT}/outputs"
export CHECKPOINT_DIR="${JOB_ROOT}/checkpoints"
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export UV_PYTHON_INSTALL_DIR="${SCRATCH_ROOT}/python"
export UV_CACHE_DIR="${SCRATCH_ROOT}/uv-cache-clariden-arm64"
export UV_PROJECT_ENVIRONMENT="${SCRATCH_ROOT}/.venv-clariden"
export UV_MANAGED_PYTHON=1

mkdir -p \
  "${RAW_DATA_DIR}" \
  "${LOG_DIR}" \
  "${OUTPUT_DIR}" \
  "${CHECKPOINT_DIR}" \
  "${JOB_ROOT}/results"
cd "${REPOSITORY_ROOT}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing to run a canary from a dirty Git worktree." >&2
  git status --short >&2
  exit 1
fi
echo "Repository commit: $(git rev-parse HEAD)"

"${UV_BIN}" run --no-sync python - <<'PY'
import torch

assert torch.cuda.is_available(), "Slurm allocation does not expose CUDA"
print(
    {
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
)
PY

"${UV_BIN}" run --no-sync python scripts/cchamber_pairing_smoke.py \
  --trainer gpu \
  --device cuda \
  --checkpoints-dir "${CHECKPOINT_DIR}" \
  --output-dir "${JOB_ROOT}/results/pairing"

echo "Causal Chamber canary passed: ${JOB_ROOT}"
