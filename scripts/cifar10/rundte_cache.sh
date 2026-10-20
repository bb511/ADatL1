#!/usr/bin/env bash
# ========================================================================
# DTE -- default parameters, cifar10
# ========================================================================
# One ImageDTE training at the config defaults, two epochs. Run it on olqti.
#
# Besides being a sanity run, this materialises the CIFAR-10 tensors and the
# per-channel normalisation stats the sweeps read. The experiment sets test: false
# and has no evaluator, but the score dump builds and scores every test set
# (stored_sets: null), which is what pulls in every anomaly class.
#
# Afterwards, copy this to clariden, keeping the path relative to the repo root,
# and launch scripts/cifar10/rundte_search.sh:
#
#   data/cifar10                                                       ~341M
set -eu

taskset -c 0-2 \
python3 src/train.py \
    experiment=cifar10/dte \
    experiment_name=dte_default \
    logger=none \
    trainer=gpu \
    trainer.devices=[0] \
    trainer.max_epochs=2
