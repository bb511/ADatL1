#!/usr/bin/env bash
# ===========================================================================
# TEST evaluation of a trained run: loss_total.ckpt on the held-out test split
# ===========================================================================
# Replays the TEST split (the zero-bias test part plus the test part of every
# simulated signal/background set) through
#
#   ${ADL1T_OUTPUT_ROOT}/checkpoints/<EXPERIMENT_NAME>/<RUN_NAME>/loss_total.ckpt
#
# and runs the same evaluation callbacks as the validation pass after training,
# writing the same artifacts under .../<RUN_NAME>/plots/test/ instead of plots/val/:
#
#   loss_total/eff/                       per-signal efficiency plots, eff_summary.json
#   loss_total/correlation_matrix/normal/ Pearson/ and Spearman/ (input,
#                                         reconstruction, self_improvement/) and
#                                         mean_correlations.json
#   loss_total/auroc/auroc_summary.json
#   loss_total/latent_collapse/collapse_summary.json
#   loss_total/{reco,ascore_operational,thres_drift}/
#   {eff,ascore_operational,thres_drift,wasserstein}_summary/
#
# Efficiencies use the operating threshold stored in loss_total.ckpt, i.e. the one
# fixed on validation data during training. Nothing under plots/val, the run
# manifest or the Pareto-study outputs is touched; the record of this step is
# stage_status/metrics_test.yaml, and the plots are logged into the run's
# existing MLflow run under test/.
#
# Not part of this script:
#   - the leakage probes on test: stage 2 with evaluation.leakage_probes.mode=final_test
#   - the comparison with the gamma = 0 run: scripts/physics/runae_test_comparison.sh
#
# Use it only for the configuration(s) finally selected on validation; test
# results must never feed back into the selection
# (docs/evaluation/leakage_probe_contract.md, section 5.1).
#
# The run is identified by EXPERIMENT_NAME and RUN_NAME alone. What the config
# needs to rebuild the trained model is read from the run's resolved_config.yaml:
#   - Pareto-study runs (tag "pareto"): experiment=physics/pareto_fet plus the
#     run's pareto_study.candidate (seed, gamma, bins, architecture, encoder nodes);
#   - other runs: experiment=physics/ae plus the run's algorithm hyperparameters.
# The fingerprint check in src/run_eval_metrics.py (run_manifest.yaml) then stops the job before any
# data is loaded if the composed model is not the one in the checkpoint. Runs
# without run_manifest.yaml (trained before 2026-09-18) are refused.
#
# Usage:
#   EXPERIMENT_NAME=Pareto-Front-261002 \
#   RUN_NAME=Seed180524_Gamma_0.1_Bins_40_architecture_h64_32_Run01 \
#     bash scripts/physics/runae_test.sh
#
#   DRY_RUN=1 ...                    print the command instead of running it
#   TEST_EXPERIMENT=physics/<name>   use this experiment instead of the one above
#
# On HTCondor: batch/runae_test.sub (one job per run name).

set -euo pipefail

: "${RUN_NAME:?Set RUN_NAME to the run to test, e.g. RUN_NAME=Seed180524_Gamma_0.1_Bins_40_architecture_h64_32_Run01}"
: "${EXPERIMENT_NAME:?Set EXPERIMENT_NAME to its checkpoint folder, e.g. EXPERIMENT_NAME=Pareto-Front-261002}"

# Same defaults as _stage_common.sh, needed here already to find the run.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${PROJECT_ROOT:=${REPO_ROOT}}"
: "${ADL1T_OUTPUT_ROOT:=${PROJECT_ROOT}}"

RUN_DIR="${ADL1T_OUTPUT_ROOT}/checkpoints/${EXPERIMENT_NAME}/${RUN_NAME}"
for required in loss_total.ckpt run_manifest.yaml resolved_config.yaml; do
  [[ -s "${RUN_DIR}/${required}" ]] || {
    echo "FATAL: missing ${RUN_DIR}/${required}" >&2
    exit 2
  }
done

# --- rebuild the stage-1 config from the run's own record ---------------------
# Values are written back exactly as the resolved config stores them (1 stays 1,
# 0.1 stays 0.1): the fingerprint hashes the resolved algorithm config, so an int
# that comes back as a float would no longer match.
RUN_SETTINGS="$(python3 - "$RUN_DIR" "$EXPERIMENT_NAME" "$RUN_NAME" <<'PY'
import shlex
import sys
from pathlib import Path

import yaml

run_dir, experiment_name, run_name = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
manifest = yaml.safe_load((run_dir / "run_manifest.yaml").read_text()) or {}
config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text()) or {}


def fmt(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(fmt(item) for item in value) + "]"
    return repr(value) if isinstance(value, float) else str(value)


def emit(name, value):
    print(f"{name}={shlex.quote(fmt(value))}")


for key, expected in (("experiment_name", experiment_name), ("run_name", run_name)):
    if str(manifest.get(key)) != expected:
        print(f"echo {shlex.quote(f'WARNING: run_manifest.yaml has {key}={manifest.get(key)!r}, not {expected!r}')} >&2")

algorithm = config.get("algorithm") or {}
pareto = "pareto" in (config.get("tags") or [])
emit("MANIFEST_EXPERIMENT", "physics/pareto_fet" if pareto else "physics/ae")
emit("MANIFEST_PARETO", int(pareto))
if pareto:
    candidate = (config.get("pareto_study") or {}).get("candidate") or {}
    for name, key in (("SEED", "autoencoder_seed"), ("MI_GAMMA", "mi_gamma"),
                      ("MI_NUM_BINS", "mi_sensitive_num_bins"),
                      ("ARCHITECTURE_ID", "architecture_id"), ("ENCODER_NODES", "encoder_nodes")):
        if candidate.get(key) is None:
            sys.exit(f"resolved_config.yaml has no pareto_study.candidate.{key}")
        emit(name, candidate[key])
else:
    optimizer = algorithm.get("optimizer") or {}
    values = {
        "SEED": config.get("seed"),
        "LR": optimizer.get("lr"),
        "WEIGHT_DECAY": optimizer.get("weight_decay"),
        "BETAS": optimizer.get("betas"),
        "DELTA": algorithm.get("delta"),
        "MI_GAMMA": algorithm.get("mi_gamma"),
        "MI_TEMPERATURE": algorithm.get("mi_temperature"),
        "MI_NUM_BINS": algorithm.get("mi_sensitive_num_bins"),
        "ENCODER_NODES": (algorithm.get("encoder") or {}).get("nodes"),
        "INPUT_NOISE_STD": algorithm.get("input_noise_std"),
    }
    for name, value in values.items():
        if value is not None:
            emit(name, value)
PY
)" || {
  echo "FATAL: could not read the configuration of $RUN_DIR" >&2
  exit 2
}
eval "$RUN_SETTINGS"

EXPERIMENT="${TEST_EXPERIMENT:-$MANIFEST_EXPERIMENT}"
PARETO_CANDIDATE="$MANIFEST_PARETO"
export EXPERIMENT EXPERIMENT_NAME PARETO_CANDIDATE PROJECT_ROOT ADL1T_OUTPUT_ROOT

# Builds COMMON_ARGS and ALGO_ARGS (from SEED, MI_GAMMA, ... set above) and cd's
# into the checkout.
source "${REPO_ROOT}/scripts/physics/_stage_common.sh"

stage_banner "TEST EVALUATION  (loss_total.ckpt on the test split)"
echo " run dir:     $RUN_DIR"
echo " output:      $RUN_DIR/plots/test/"
echo "==============================================================="

command=(
  python3 src/run_eval_metrics.py
  "${COMMON_ARGS[@]}"
  "${ALGO_ARGS[@]}"
  eval_split=test
)

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${command[@]}"
  echo
  exit 0
fi
exec "${command[@]}"
