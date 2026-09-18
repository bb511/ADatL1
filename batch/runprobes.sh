#!/bin/bash
# STAGE 2 of 4 on HTCondor -- the four leakage probes.
#
# One job per run. RUN_NAME comes from the queue statement in batch/runprobes.sub;
# it must name a run whose stage-1 job already wrote loss_total.ckpt into
# ADL1T_OUTPUT_ROOT/checkpoints/<experiment_name>/<RUN_NAME>/.
#
# Unlike stage 1 this job READS an existing checkpoint tree, so ADL1T_OUTPUT_ROOT
# must point at the merged stage-1 output on EOS, not at the empty job scratch
# dir. That is the one setting that differs from batch/runae.sh, and getting it
# wrong surfaces immediately as "Missing loss_total.ckpt" rather than as a
# silently wrong result.

STAGE_LABEL="AE leakage probes (stage 2/4)"
export STAGE_LABEL

: "${ADL1T_OUTPUT_ROOT:=/eos/user/l/lbehrens/adatl1/outputs}"
export ADL1T_OUTPUT_ROOT

source "$(dirname "$0")/_stage_env.sh"

: "${PROBE_MODE:=validation}"
: "${PROBE_SHUFFLED_CONTROLS:=false}"
export PROBE_MODE PROBE_SHUFFLED_CONTROLS

echo "PROBE_MODE:        $PROBE_MODE"
echo
echo "Running scripts/physics/runprobes.sh ..."
exec bash "${CODE_DIR}/scripts/physics/runprobes.sh"
