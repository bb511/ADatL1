#!/usr/bin/env bash
#SBATCH --job-name=jetclr-zb-match
#SBATCH --account=a0166
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=800G
#SBATCH --exclusive
#SBATCH --output=/iopsstor/scratch/cscs/vjimenez/jetclr/logs/slurm/%x-%j.out

set -euo pipefail

: "${PAIR_DEPLOYMENT:?PAIR_DEPLOYMENT is required}"
: "${PAIR_OUTPUT_ROOT:?PAIR_OUTPUT_ROOT is required}"

readonly UV_BIN=/iopsstor/scratch/cscs/vjimenez/adatl1/tools/uv-0.11.32/uv
readonly PROJECT_ENV=/iopsstor/scratch/cscs/vjimenez/adatl1/.venv-clariden

export UV_PROJECT_ENVIRONMENT="${PROJECT_ENV}"
export PYTHONPATH="${PAIR_DEPLOYMENT}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
mkdir -p "${PAIR_OUTPUT_ROOT}/pairing"
cd "${PAIR_DEPLOYMENT}"

"${UV_BIN}" run --frozen --no-sync python -m src.utils.pairing.jetclr_run_tables \
  match \
  --target-embeddings "${PAIR_OUTPUT_ROOT}/embeddings/ZB_run396102.npy" \
  --reference-embeddings "${PAIR_OUTPUT_ROOT}/embeddings/ZB_run398183.npy" \
  --output "${PAIR_OUTPUT_ROOT}/pairing/ZB_run396102_to_ZB_run398183.pt" \
  --backend "${PAIR_BACKEND:-ivf_flat}" \
  --k "${PAIR_K:-8}" \
  --nlist "${PAIR_NLIST:-16384}" \
  --nprobe "${PAIR_NPROBE:-32}" \
  --train-events "${PAIR_TRAIN_EVENTS:-262144}" \
  --train-iterations "${PAIR_TRAIN_ITERATIONS:-12}" \
  --search-batch-size "${PAIR_SEARCH_BATCH_SIZE:-131072}" \
  --add-batch-size "${PAIR_ADD_BATCH_SIZE:-262144}" \
  --threads "${SLURM_CPUS_PER_TASK}" \
  --seed 271828
