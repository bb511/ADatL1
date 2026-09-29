#!/bin/bash
# BACKFILL for runs trained before stage 1 produced the full evaluation.
#
# Replays the validation split through an existing loss_total.ckpt and runs
# only the evaluation callbacks whose plots the old stage 1 did not write:
#
#   correlation_matrix  -> plots/val/loss_total/correlation_matrix/normal/{Pearson,Spearman}/*.png|csv
#   anomaly_efficiency  -> plots/val/loss_total/eff/... per-signal plots (+ eff_summary.json, same numbers)
#
# Everything else stage 1 already wrote (reco, ascore_operational, thres_drift,
# wasserstein, auroc, latent_collapse) is switched off here, so nothing is
# recomputed or overwritten. No training, no probes. Runs trained with the
# current configs do not need this: their stage 1 already writes all of it.
#
# One job per run, same queue list as stage 1. Submit with:
#   condor_submit batch/runplots_pareto.sub

STAGE_LABEL="Pareto backfill: evaluation plots for an existing checkpoint"
export STAGE_LABEL

: "${EXPERIMENT:=physics/pareto_fet}"
: "${PARETO_CANDIDATE:=1}"
export EXPERIMENT PARETO_CANDIDATE

# Read and write the merged tree on EOS directly, like stage 3: the checkpoint
# is already there and the plots belong beside it.
: "${ADL1T_OUTPUT_ROOT:=/eos/user/l/lbehrens/adatl1/ADatL1/outputs}"
export ADL1T_OUTPUT_ROOT

: "${RUN_NAME:?Set RUN_NAME via the queue statement}"
: "${SEED:?Set SEED via the queue statement}"
: "${MI_GAMMA:?Set MI_GAMMA via the queue statement}"
: "${MI_NUM_BINS:?Set MI_NUM_BINS via the queue statement}"
: "${ARCHITECTURE_ID:?Set ARCHITECTURE_ID via the queue statement}"
: "${ENCODER_NODES_US:?Set ENCODER_NODES_US via the queue statement}"
ENCODER_NODES="[${ENCODER_NODES_US//_/,}]"
export SEED MI_GAMMA MI_NUM_BINS ARCHITECTURE_ID ENCODER_NODES

: "${CODE_DIR:=/eos/user/l/lbehrens/adatl1/ADatL1}"
export CODE_DIR
STAGE_ENV="${CODE_DIR}/batch/_stage_env.sh"
[[ -r "$STAGE_ENV" ]] || { echo "FATAL: cannot read $STAGE_ENV" >&2; exit 2; }
source "$STAGE_ENV"

: "${EXPERIMENT_NAME:=$PARETO_EXPERIMENT_NAME}"
export EXPERIMENT_NAME

CKPT="${ADL1T_OUTPUT_ROOT}/checkpoints/${EXPERIMENT_NAME}/${RUN_NAME}/loss_total.ckpt"
[[ -s "$CKPT" ]] || { echo "FATAL: no checkpoint at $CKPT" >&2; exit 2; }
echo "Checkpoint:        $CKPT"

# Builds COMMON_ARGS / ALGO_ARGS from the environment above (cd's into CODE_DIR).
source "${CODE_DIR}/scripts/physics/_stage_common.sh"

# optimized_metric_config=null: the evaluator's Optuna bookkeeping looks up the
# ascore_operational callback as its secondary metric. That callback is switched
# off below, so without this the job wrote every plot and then exited 1 with
# "Callback ascore_operational not available" (cluster 354142, 2026-09-29).
exec python3 src/run_eval_metrics.py \
  "${COMMON_ARGS[@]}" \
  "${ALGO_ARGS[@]}" \
  optimized_metric_config=null \
  evaluation.callbacks.reco=null \
  evaluation.callbacks.ascore_operational=null \
  evaluation.callbacks.thres_drift=null \
  evaluation.callbacks.wasserstein=null \
  evaluation.callbacks.anomaly_auroc=null \
  evaluation.callbacks.latent_collapse=null \
  evaluation.callbacks.anomaly_efficiency.write_plots=true \
  evaluation.callbacks.correlation_matrix.enabled=true \
  evaluation.callbacks.correlation_matrix.write_details=true \
  evaluation.callbacks.correlation_matrix.write_source_tables=false
