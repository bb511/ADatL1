#!/bin/bash

# ===========================================================================
# Full FET.Et validation Pareto study on LXPLUS HTCondor.
# ===========================================================================
# Sibling of batch/runae.sh. That script trains one AE; this one runs the
# complete study end to end in a single job:
#
#   Phase 1  train every grid point x paired seed
#   Phase 2  scripts/collect_pareto_study.py   -> pareto_metrics.{csv,parquet}
#   Phase 3  scripts/select_pareto_front.py    -> pareto_front.csv + selection
#
# The grid is NOT defined here. scripts/physics/run_pareto_fet_ngt.sh reads it
# from configs/experiment/physics/pareto_fet.yaml, which is the predeclared
# contract. Check the size before submitting:
#
#   bash scripts/physics/run_pareto_fet_ngt.sh --plan
#
# PARALLEL MODE (what runpareto.sub does by default)
# --------------------------------------------------
# One job per shard, each training an interleaved slice of the grid, so the
# study spreads over the fleet instead of occupying one machine for two days.
# Set PARETO_SHARD=<i>/<n>; the submit file derives i from $(ProcId).
#
# Phase 2 and Phase 3 are NOT run by a shard - they need every run's artifacts
# in one place, and each shard only has its own. After all shards finish and
# their outputs/ trees have merged under the submit directory, run the analysis
# once on lxplus (it is CPU-only and takes minutes):
#
#   cd /eos/user/l/lbehrens/adatl1/ADatL1
#   ADL1T_OUTPUT_ROOT=$PWD/outputs \
#   PROJECT_ROOT=/eos/user/l/lbehrens/adl1t-stage \
#     bash scripts/physics/run_pareto_fet_ngt.sh --collect
#
# --collect rewrites the study map from the manifest first, so it picks up the
# merged location rather than the sandbox paths recorded during training.
#
# To run everything in a single job instead, set PARETO_SHARD= (empty) and
# PARETO_ACTION=--all, and raise +MaxRuntime accordingly.
#
set -euo pipefail

echo "========================================"
echo "FET.Et Pareto study (GPU) on HTCondor"
echo "========================================"
echo "Date:     $(date)"
echo "Hostname: $(hostname)"
echo "PWD:      $(pwd)"
echo "User:     $(whoami)"

# Dependencies live in a venv inside the image and are not on PATH by default;
# python3 would otherwise resolve to the system interpreter without torch.
if [[ -d /opt/venv/bin ]]; then
  export PATH="/opt/venv/bin:$PATH"
fi

echo
echo "Python:"
command -v python3
python3 --version
python3 -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'

echo
echo "GPU:"
nvidia-smi || echo "nvidia-smi unavailable"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Code and staged data on EOS, exactly as batch/runae.sh uses them.
: "${CODE_DIR:=/eos/user/l/lbehrens/adatl1/ADatL1}"
: "${PROJECT_ROOT:=/eos/user/l/lbehrens/adl1t-stage}"
: "${RAW_DATA_DIR:=${PROJECT_ROOT}/raw/parquet_files}"

# Outputs. paths.output_root resolves to ADL1T_OUTPUT_ROOT, and the runner now
# derives its study root and checkpoint paths from the same variable, so the
# study map points where training actually wrote.
#
# Default is the job sandbox, transferred back by transfer_output_files. Do not
# point this at /eos unless you accept the Kerberos caveat in the runbook: a
# worker writing to /eos depends on the job's credentials surviving the run.
# Pointing it at EOS does make the study resumable, because the runner skips
# runs that already have a complete reportable artifact set.
SCRATCH="${_CONDOR_SCRATCH_DIR:-$PWD}"
: "${ADL1T_OUTPUT_ROOT:=${SCRATCH}/outputs}"
: "${MPLCONFIGDIR:=${SCRATCH}/matplotlib}"

# Threading. Keep at or below request_cpus.
: "${DATA_WORKERS:=3}"

# Epoch budget per run. The manifest inherits trainer.max_epochs=200 from
# physics/ae, which no batch job will finish; 30 is the agreed ceiling.
: "${MAX_EPOCHS:=30}"

# The study runs from a checkout that may legitimately differ from HEAD while
# iterating. The runner records the commit and dirty status either way.
: "${ALLOW_DIRTY_GIT:=1}"

: "${PARETO_ACTION:=--run}"

# <i>/<n>, or empty for the whole grid in this one job.
: "${PARETO_SHARD:=}"

export CODE_DIR PROJECT_ROOT RAW_DATA_DIR ADL1T_OUTPUT_ROOT MPLCONFIGDIR
export DATA_WORKERS MAX_EPOCHS ALLOW_DIRTY_GIT

mkdir -p "$ADL1T_OUTPUT_ROOT" "$MPLCONFIGDIR"

echo
echo "CODE_DIR:          $CODE_DIR"
echo "PROJECT_ROOT:      $PROJECT_ROOT"
echo "RAW_DATA_DIR:      $RAW_DATA_DIR"
echo "ADL1T_OUTPUT_ROOT: $ADL1T_OUTPUT_ROOT"
echo "DATA_WORKERS:      $DATA_WORKERS"
echo "MAX_EPOCHS:        $MAX_EPOCHS"
echo "PARETO_ACTION:     $PARETO_ACTION"
echo "PARETO_SHARD:      ${PARETO_SHARD:-<whole grid>}"

echo
echo "Study plan:"
bash "${CODE_DIR}/scripts/physics/run_pareto_fet_ngt.sh" --plan

runner_args=("${PARETO_ACTION}")
if [[ -n "$PARETO_SHARD" ]]; then
  runner_args+=(--shard "$PARETO_SHARD")
fi

echo
echo "Running scripts/physics/run_pareto_fet_ngt.sh ${runner_args[*]} ..."
exec bash "${CODE_DIR}/scripts/physics/run_pareto_fet_ngt.sh" "${runner_args[@]}"
