#!/usr/bin/env bash
# =============================================================================
# HTCondor batch wrapper – AE training on LXPLUS
# =============================================================================
# Configures the LXPLUS / EOS environment and delegates to
# scripts/physics/runae.sh.
#
# Submit (from the ADatL1 repo root):
#   condor_submit batch/runae.sub
#
# Override any parameter at submit time, e.g.:
#   condor_submit batch/runae.sub -append 'environment = "RUN_NAME=My_Run_02 MAX_EPOCHS=10"'
# =============================================================================
set -euo pipefail

echo "========================================"
echo "AE Training — LXPLUS HTCondor batch"
echo "========================================"
echo "Date:     $(date)"
echo "Hostname: $(hostname)"
echo "User:     $(whoami)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo ""

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "--- nvidia-smi ---"
  nvidia-smi
  echo "------------------"
  echo ""
fi

# ── Paths (LXPLUS / EOS) ─────────────────────────────────────────────────────
# EOS base directory that holds the project data, code and outputs.
# Mirrors the layout used in test_container.sh.
EOS_BASE="/eos/user/l/lbehrens/adatl1"

# Location of the ADatL1 git repository on EOS (same as in test_container.sh).
export CODE_DIR="${CODE_DIR:-${EOS_BASE}/ADatL1}"

# Root of the staged physics data; runae.sh expects
#   ${PROJECT_ROOT}/data/data_2025E+G/{extracted,processed,mlready}
export PROJECT_ROOT="${PROJECT_ROOT:-${EOS_BASE}}"

# EOS directory where MLflow artefacts, checkpoints and plots are written.
export ADL1T_OUTPUT_ROOT="${ADL1T_OUTPUT_ROOT:-${EOS_BASE}/output}"

# Matplotlib config cache – must be writable inside the container.
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

# ── Training hyper-parameters ─────────────────────────────────────────────────
export RUN_NAME="${RUN_NAME:-AE_LXPLUS_Run01}"
export MAX_EPOCHS="${MAX_EPOCHS:-3}"
export MI_GAMMA="${MI_GAMMA:-0.1}"
export MI_NUM_BINS="${MI_NUM_BINS:-50}"
export DATA_WORKERS="${DATA_WORKERS:-6}"
export CKPT_PATH="${CKPT_PATH:-}"

echo "Configuration:"
echo "  CODE_DIR:           $CODE_DIR"
echo "  PROJECT_ROOT:       $PROJECT_ROOT"
echo "  ADL1T_OUTPUT_ROOT:  $ADL1T_OUTPUT_ROOT"
echo "  RUN_NAME:           $RUN_NAME"
echo "  MAX_EPOCHS:         $MAX_EPOCHS"
echo "  MI_GAMMA:           $MI_GAMMA"
echo "  MI_NUM_BINS:        $MI_NUM_BINS"
echo "  DATA_WORKERS:       $DATA_WORKERS"
echo "  CKPT_PATH:          ${CKPT_PATH:-(none)}"
echo ""

# ── Delegate ──────────────────────────────────────────────────────────────────
RUNAE_SCRIPT="${CODE_DIR}/scripts/physics/runae.sh"

if [[ ! -f "$RUNAE_SCRIPT" ]]; then
  echo "ERROR: runae.sh not found at $RUNAE_SCRIPT"
  exit 1
fi

echo "Handing off to $RUNAE_SCRIPT ..."
exec bash "$RUNAE_SCRIPT"
