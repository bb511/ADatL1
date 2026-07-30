#!/usr/bin/env bash
# ========================================================================
# DTE -- default parameters, robustad
# ========================================================================
# One ImageDTE training at the config defaults, two epochs. Run it on olqti.
#
# Besides being a sanity run, this materialises the decoded image tensors and
# the channel stats for the `pcb` subset -- the only subset any experiment
# config uses. The evaluator runs too (test=true, the config default), which is
# what pulls in every shifted normal and shifted anomaly split.
#
# Afterwards, copy this to clariden, keeping the path relative to the repo root,
# and launch scripts/robustad/rundte_search.sh:
#
#   data/robustad                                                      ~18G
set -eu

python3 src/train.py \
    experiment=robustad/dte \
    experiment_name=dte_default \
    logger=none \
    trainer=gpu \
    trainer.devices=[0] \
    trainer.max_epochs=2
