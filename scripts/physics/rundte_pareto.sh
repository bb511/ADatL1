#!/usr/bin/env bash
# ========================================================================
# DTE -- default parameters, physics
# ========================================================================
# One DTE training at the config defaults, two epochs. Run it on olqti.
#
# Besides being a sanity run, this is what builds the caches the sweeps need.
# configs/experiment/physics/dte*.yaml override /data/data_normalizer to
# `standard`, and nothing else in the repo uses that normalizer, so its cache
# tree has to be built once from scratch. Doing it here, up front, is what keeps
# the six concurrent sweep drivers on clariden from racing on the same directory
# and leaving partial parquet behind.
#
# The evaluator runs too (test=true, the config default) and that is deliberate:
# the test split and the ~20 auxiliary signal datasets only get their torch
# caches written when something actually reads them.
#
# This one run covers all four physics configs -- physics/dte,
# physics/dte_agnostic and both q99 variants read the same cache tree, because
# target_rate/base_rate change the metric, not the data.
#
# Afterwards, copy these to clariden, keeping the paths relative to the repo
# root, and launch scripts/physics/rundte_search.sh:
#
#   data/data_2025E+G/extracted                                        ~3.0G
#   data/data_2025E+G/processed                                        ~3.1G
#   data/data_2025E+G/mlready/eminimalTauFET_pdefaultTauFET_default/standard  ~24G
#
# All three are needed. prepare_data() walks extract -> process -> mlready on
# every run; each stage skips when its output exists, but each still lists its
# input directory, so a missing `extracted` or `processed` sends it back to the
# raw ntuples. The sibling mlready/.../robust tree belongs to the other models
# and does not need to travel.
set -eu

taskset -c 0-2 \
python3 src/train.py \
    experiment=physics/dte \
    paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
    experiment_name=dte_default \
    logger=none \
    trainer=gpu \
    trainer.devices=[0] \
    trainer.max_epochs=2
