#!/usr/bin/env bash
# ===========================================================================
# STAGE 3 of 4 -- the remaining Pareto metrics
# ===========================================================================
# Replays the validation split through the loss_total.ckpt that stage 1 wrote
# and lets the evaluation callbacks write their summary artifacts, under
# checkpoints/<experiment_name>/<RUN_NAME>/plots/val/loss_total/:
#
#   eff/eff_summary.json                        per-signal and summary efficiency
#   correlation_matrix/normal/
#                    mean_correlations.json     objective E (residual correlation)
#   latent_collapse/collapse_summary.json       the feasibility constraint
#   auroc/auroc_summary.json                    diagnostic
#
# This is the memory-hungry stage: it holds the validation split plus all 21
# auxiliary signal datasets at once. It is independent of stage 2.
#
# EXPERIMENT defaults to physics/ae_metrics here rather than physics/ae, because
# the plain AE experiment deliberately leaves the correlation matrix disabled and
# defines no auroc or latent_collapse callback. ae_metrics inherits physics/ae
# unchanged and only switches those four on, so the architecture still matches
# the trained checkpoint exactly.
#
# Usage:
#   RUN_NAME=AE_30ep bash scripts/physics/runmetrics.sh
#
# For a Pareto-study run use the study's own experiment instead:
#   RUN_NAME=<study run> EXPERIMENT=physics/pareto_fet bash scripts/physics/runmetrics.sh

: "${EXPERIMENT:=physics/ae_metrics}"
export EXPERIMENT

source "$(dirname "${BASH_SOURCE[0]}")/_stage_common.sh"

# Labels the architecture when pairing a candidate with its gamma=0 baseline.
# Change it whenever you compare several encoder shapes.
: "${ARCHITECTURE_ID:=ae_standalone}"

architecture_args=()
if [[ "$EXPERIMENT" == "physics/ae_metrics" ]]; then
  architecture_args=(
    "evaluation.callbacks.latent_collapse.architecture_id=$ARCHITECTURE_ID"
  )
fi

stage_banner "STAGE 3/4  EVALUATION METRICS"

exec python3 src/run_eval_metrics.py \
  "${COMMON_ARGS[@]}" \
  "${ALGO_ARGS[@]}" \
  "${architecture_args[@]}"
