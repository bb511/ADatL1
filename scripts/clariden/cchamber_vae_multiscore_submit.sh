#!/usr/bin/env bash
set -euo pipefail

readonly REPOSITORY_ROOT=/users/vjimenez/adatl1
readonly SCRATCH_ROOT=/iopsstor/scratch/cscs/vjimenez/adatl1
readonly CAMPAIGN_ROOT="${SCRATCH_ROOT}/campaigns/cchamber_vae_multiscore_20260803_auto"
readonly UV_BIN="${SCRATCH_ROOT}/tools/uv-0.11.32/uv"
readonly SCRIPT_DIR="${REPOSITORY_ROOT}/scripts/clariden"

export UV_PROJECT_ENVIRONMENT="${SCRATCH_ROOT}/.venv-clariden"
export UV_CACHE_DIR="${SCRATCH_ROOT}/uv-cache-clariden-arm64"
export UV_PYTHON_INSTALL_DIR="${SCRATCH_ROOT}/python"
export UV_MANAGED_PYTHON=1

cd "${REPOSITORY_ROOT}"
"${UV_BIN}" run --frozen --no-sync python scripts/cchamber_vae_multiscore_campaign.py \
  init --root "${CAMPAIGN_ROOT}"

canary_job=$(sbatch --parsable "${SCRIPT_DIR}/cchamber_vae_multiscore_canary.sbatch")
train_job=$(sbatch \
  --parsable \
  --dependency="afterok:${canary_job}" \
  "${SCRIPT_DIR}/cchamber_vae_multiscore_train.sbatch")
freeze_job=$(sbatch \
  --parsable \
  --dependency="afterok:${train_job}" \
  "${SCRIPT_DIR}/cchamber_vae_multiscore_freeze.sbatch")
evaluate_job=$(sbatch \
  --parsable \
  --dependency="afterok:${freeze_job}" \
  "${SCRIPT_DIR}/cchamber_vae_multiscore_evaluate.sbatch")
analysis_job=$(sbatch \
  --parsable \
  --dependency="afterok:${evaluate_job}" \
  "${SCRIPT_DIR}/cchamber_vae_multiscore_analyze.sbatch")

printf 'canary=%s\ntrain=%s\nfreeze=%s\nevaluate=%s\nanalysis=%s\n' \
  "${canary_job}" "${train_job}" "${freeze_job}" "${evaluate_job}" "${analysis_job}"
