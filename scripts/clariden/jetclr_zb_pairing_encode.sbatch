#!/usr/bin/env bash
#SBATCH --job-name=jetclr-zb-encode
#SBATCH --account=a0166
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --mem=180G
#SBATCH --array=0-1%2
#SBATCH --output=/iopsstor/scratch/cscs/vjimenez/jetclr/logs/slurm/%x-%A_%a.out

set -euo pipefail

: "${PAIR_DEPLOYMENT:?PAIR_DEPLOYMENT is required}"
: "${PAIR_OUTPUT_ROOT:?PAIR_OUTPUT_ROOT is required}"

readonly UV_BIN=/iopsstor/scratch/cscs/vjimenez/adatl1/tools/uv-0.11.32/uv
readonly PROJECT_ENV=/iopsstor/scratch/cscs/vjimenez/adatl1/.venv-clariden
readonly TRAINING_CACHE=/iopsstor/scratch/cscs/podagiu/data/data_2025E+G/mlready/eminimalTauFET_pdefaultTauFET_default/robust
readonly PROCESSED_ROOT=/iopsstor/scratch/cscs/podagiu/data/data_2025E+G/processed/eminimalTauFET_pdefaultTauFET/zerobias
readonly CHECKPOINT=/iopsstor/scratch/cscs/vjimenez/jetclr/final/jetclr_aad_best_20260801/jetclr_aad_seed456_epoch1.ckpt

readonly -a RUNS=(ZB_run396102 ZB_run398183)
readonly RUN_NAME="${RUNS[${SLURM_ARRAY_TASK_ID}]}"

export UV_PROJECT_ENVIRONMENT="${PROJECT_ENV}"
export PYTHONPATH="${PAIR_DEPLOYMENT}"
export PROJECT_ROOT="${PAIR_DEPLOYMENT}"
export DATA_DIR=/iopsstor/scratch/cscs/podagiu/data/data_2025E+G
export RAW_DATA_DIR=/iopsstor/scratch/cscs/podagiu/data
export LOG_DIR="${PAIR_OUTPUT_ROOT}/logs"
export OUTPUT_DIR="${PAIR_OUTPUT_ROOT}/outputs"
export CHECKPOINT_DIR="${PAIR_OUTPUT_ROOT}/checkpoints"
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1

mkdir -p "${PAIR_OUTPUT_ROOT}/embeddings" "${LOG_DIR}" "${OUTPUT_DIR}" "${CHECKPOINT_DIR}"
cd "${PAIR_DEPLOYMENT}"

arguments=(
  encode
  --run-dir "${PROCESSED_ROOT}/${RUN_NAME}"
  --training-cache "${TRAINING_CACHE}"
  --checkpoint "${CHECKPOINT}"
  --output "${PAIR_OUTPUT_ROOT}/embeddings/${RUN_NAME}.npy"
  --batch-size "${PAIR_BATCH_SIZE:-8192}"
  --device cuda:0
)
if [[ -n "${PAIR_MAX_EVENTS:-}" ]]; then
  arguments+=(--max-events "${PAIR_MAX_EVENTS}")
fi
arguments+=(experiment=physics/jetclr_aad_best)

"${UV_BIN}" run --frozen --no-sync python -m src.utils.pairing.jetclr_run_tables \
  "${arguments[@]}"
