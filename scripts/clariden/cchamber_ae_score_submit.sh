#!/usr/bin/env bash
set -euo pipefail

readonly REPOSITORY_ROOT=/users/vjimenez/adatl1
readonly SCRATCH_ROOT=/iopsstor/scratch/cscs/vjimenez/adatl1
readonly AUDIT_ROOT="${SCRATCH_ROOT}/audits/cchamber_real_20260801_3789655_candidate_rank_3789655"
readonly OUTPUT_ROOT="${SCRATCH_ROOT}/audits/cchamber_ae_score_audit_20260802"
readonly UV_BIN="${SCRATCH_ROOT}/tools/uv-0.11.32/uv"
readonly SCRIPT_DIR="${REPOSITORY_ROOT}/scripts/clariden"

export UV_PROJECT_ENVIRONMENT="${SCRATCH_ROOT}/.venv-clariden"
export UV_CACHE_DIR="${SCRATCH_ROOT}/uv-cache-clariden-arm64"
export UV_PYTHON_INSTALL_DIR="${SCRATCH_ROOT}/python"
export UV_MANAGED_PYTHON=1

cd "${REPOSITORY_ROOT}"
"${UV_BIN}" run --frozen --no-sync python scripts/cchamber_ae_score_audit.py \
  design \
  --audit-root "${AUDIT_ROOT}" \
  --output-root "${OUTPUT_ROOT}"

extract_job=$(sbatch --parsable "${SCRIPT_DIR}/cchamber_ae_score_extract.sbatch")
freeze_job=$(sbatch \
  --parsable \
  --dependency="afterok:${extract_job}" \
  "${SCRIPT_DIR}/cchamber_ae_score_freeze.sbatch")
evaluate_job=$(sbatch \
  --parsable \
  --dependency="afterok:${freeze_job}" \
  "${SCRIPT_DIR}/cchamber_ae_score_evaluate.sbatch")
analysis_job=$(sbatch \
  --parsable \
  --dependency="afterok:${evaluate_job}" \
  "${SCRIPT_DIR}/cchamber_ae_score_analyze.sbatch")

printf 'extract=%s\nfreeze=%s\nevaluate=%s\nanalysis=%s\n' \
  "${extract_job}" "${freeze_job}" "${evaluate_job}" "${analysis_job}"
