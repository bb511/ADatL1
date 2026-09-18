#!/usr/bin/env bash
# ===========================================================================
# STAGE 2 of 4 -- the four leakage probes
# ===========================================================================
# Loads the loss_total.ckpt that stage 1 wrote and measures objective L: the
# maximum clipped held-out R^2 over {MLP, linear} x {z_logits, reconstruction}
# when predicting the sensitive variable from the model's representations.
#
# Writes, under checkpoints/<experiment_name>/<RUN_NAME>/plots/val/loss_total/probes/:
#   leakage_probes.json           the scientific record
#   leakage_probes_summary.json   the compact form
#   leakage_probes_loss_plots/    one PNG per probe
#
# This is the slow stage: ~27 min per run, independent of epoch count, dominated
# by the two MLP probes. It is independent of stage 3 and may run alongside it.
#
# Usage:
#   RUN_NAME=AE_30ep bash scripts/physics/runprobes.sh
#
# Exit codes: 0 valid, 2 bad arguments, 3 the probes ran but the protocol
# rejected the result (an invalid result IS written to disk -- the run is
# finished, it just must not be reported).

source "$(dirname "${BASH_SOURCE[0]}")/_stage_common.sh"

# validation or final_test. Keep validation for everything except the single
# selected configuration at the very end of the study.
: "${PROBE_MODE:=validation}"
# Shuffled-target controls are a guardrail, not a measurement; off by default
# because they roughly double the stage's runtime.
: "${PROBE_SHUFFLED_CONTROLS:=false}"

stage_banner "STAGE 2/4  LEAKAGE PROBES  (mode=$PROBE_MODE)"

exec python3 src/run_probes.py \
  "${COMMON_ARGS[@]}" \
  "${ALGO_ARGS[@]}" \
  evaluation.leakage_probes.enabled=true \
  evaluation.leakage_probes.mode="$PROBE_MODE" \
  evaluation.leakage_probes.run_shuffled_target_controls="$PROBE_SHUFFLED_CONTROLS" \
  evaluation.leakage_probes.smoke_test.enabled=false
