#!/usr/bin/env bash
# ===========================================================================
# Pareto study -- write the stage-1 job list from the grid config
# ===========================================================================
# Expands configs/pareto_study/fet_et.yaml into one line per training run,
#
#   SEED,GAMMA,BINS,ARCH,NODES,RUN_NAME
#
# the list batch/runae_pareto.sub queues from (default: batch/pareto_runs.txt).
# The logic lives in src/evaluation/pareto/make_grid.py; this wrapper only runs
# it from the repository root, so it works from any directory.
#
# Usage:
#   bash scripts/physics/runae_pareto_makegrid.sh                     # whole grid
#   bash scripts/physics/runae_pareto_makegrid.sh --attempt 02 --only Gamma_0.1_
#   bash scripts/physics/runae_pareto_makegrid.sh --exclude batch/pareto_runs.txt \
#        --output batch/pareto_runs_refine.txt                        # only new points
#   bash scripts/physics/runae_pareto_makegrid.sh --help
#
# Needs python3 with omegaconf (PYTHON overrides the interpreter). No torch, no data.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
exec "${PYTHON:-python3}" -m src.evaluation.pareto.make_grid "$@"
