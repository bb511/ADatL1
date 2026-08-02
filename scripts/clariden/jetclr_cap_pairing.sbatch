#!/usr/bin/env bash
#SBATCH --job-name=jetclr-cap-pair
#SBATCH --account=a0166
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --mem=180G
#SBATCH --array=0-1%2
#SBATCH --output=/iopsstor/scratch/cscs/vjimenez/jetclr/logs/slurm/%x-%A_%a.out

set -euo pipefail

readonly REPOSITORY=/users/vjimenez/train-jetclr
readonly UV_BIN=/iopsstor/scratch/cscs/vjimenez/adatl1/tools/uv-0.11.32/uv
readonly PROJECT_ENV=/iopsstor/scratch/cscs/vjimenez/adatl1/.venv-clariden
readonly CACHE_ROOT=/iopsstor/scratch/cscs/podagiu/data/data_2025E+G/mlready/eminimalTauFET_pdefaultTauFET_default/robust
readonly PAIRING_ROOT=/iopsstor/scratch/cscs/vjimenez/adatl1/data/data_2025E+G/pairing/ZB_run396102_to_ZB_run398183
readonly CHECKPOINT=/iopsstor/scratch/cscs/vjimenez/jetclr/final/jetclr_aad_best_20260801/jetclr_aad_seed456_epoch1.ckpt
readonly -a STAGES=(validate test)
readonly STAGE="${STAGES[${SLURM_ARRAY_TASK_ID}]}"

export UV_PROJECT_ENVIRONMENT="${PROJECT_ENV}"
export PYTHONPATH="${REPOSITORY}"
export PROJECT_ROOT="${REPOSITORY}"
export DATA_DIR=/iopsstor/scratch/cscs/vjimenez/adatl1/data/data_2025E+G
export RAW_DATA_DIR=/iopsstor/scratch/cscs/podagiu/data
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1

cd "${REPOSITORY}"
"${UV_BIN}" run --frozen --no-sync python -m src.utils.pairing.physics_tables \
  --cache-root "${CACHE_ROOT}" \
  --source-metadata-dir "${PAIRING_ROOT}/sources" \
  --out-dir "${PAIRING_ROOT}" \
  --stage "${STAGE}" \
  --strategy jetclr \
  --events 163840 \
  --backend faiss_hnsw \
  --device cuda:0 \
  --jetclr-batch-size 8192 \
  --initial-k 64 \
  --max-k 256 \
  --audit-events 20000 \
  --jetclr-checkpoint "${CHECKPOINT}"
