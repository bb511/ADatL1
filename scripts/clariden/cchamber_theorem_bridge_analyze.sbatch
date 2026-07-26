#!/usr/bin/env bash
#SBATCH --job-name=cch-theorem-analyze
#SBATCH --account=a0166
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --output=/iopsstor/scratch/cscs/vjimenez/adatl1/logs/slurm/%x-%j.out

set -euo pipefail

readonly REPOSITORY_ROOT=/users/vjimenez/adatl1
readonly SCRATCH_ROOT=/iopsstor/scratch/cscs/vjimenez/adatl1
readonly AUDIT_ROOT="${SCRATCH_ROOT}/audits/cchamber_real_20260725_63b941a_candidate_rank_d1a3017"
readonly BRIDGE_ROOT="${SCRATCH_ROOT}/audits/cchamber_theorem_bridge_20260726"
readonly TARGET_CATALOG="${SCRATCH_ROOT}/campaigns/cchamber_real_20260725_63b941a/design/physical_intervention_catalog_v1.json"
readonly CONFIRMATORY_RESULTS="${SCRATCH_ROOT}/audits/cchamber_real_20260725_63b941a_threshold_d1a3017/results/threshold_safe_results.csv"
readonly OUTPUT_DIR="${BRIDGE_ROOT}/analysis_v4"
readonly UV_BIN="${SCRATCH_ROOT}/tools/uv-0.11.32/uv"

export UV_PROJECT_ENVIRONMENT="${SCRATCH_ROOT}/.venv-clariden"
export UV_CACHE_DIR="${SCRATCH_ROOT}/uv-cache-clariden-arm64"
export UV_PYTHON_INSTALL_DIR="${SCRATCH_ROOT}/python"
export UV_MANAGED_PYTHON=1

cd "${REPOSITORY_ROOT}"
"${UV_BIN}" run --frozen --no-sync python scripts/cchamber_theorem_bridge.py \
  analyze \
  --audit-root "${AUDIT_ROOT}" \
  --scores-root "${BRIDGE_ROOT}/scores" \
  --target-catalog "${TARGET_CATALOG}" \
  --confirmatory-results "${CONFIRMATORY_RESULTS}" \
  --output-dir "${OUTPUT_DIR}"
