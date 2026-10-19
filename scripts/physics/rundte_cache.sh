#!/usr/bin/env bash
# ========================================================================
# DTE -- default parameters, physics, raw route
# ========================================================================
# One DTE training at the config defaults, two epochs, reading the raw parquet
# files through data=basis, the experiment's default. Run it on olqti.
#
# Besides being a sanity run, this builds the raw-route caches DTE needs.
# configs/experiment/physics/dte.yaml overrides /data/data_normalizer to
# `standard`, and no other physics experiment uses that normalizer, so its cache
# tree has to be built once from scratch. Doing it here, before concurrent DTE
# runs on the raw route, keeps them from racing on the same directory and
# leaving partial parquet behind.
#
# The evaluator runs too (test=true, the config default) and that is deliberate:
# the test split and the ~20 auxiliary signal datasets only get their torch
# caches written when something actually reads them.
#
# The one cache tree serves both rate tiers of physics/dte, because
# target_rate/base_rate change the metric, not the data. The pool drivers in
# scripts/physics/rundte_search.sh read the published record instead
# (data=basis_hf), whose caches sit under data/hf/ and are not built here.
#
# A machine that runs DTE on the raw route needs these trees, at the same paths
# relative to the repo root:
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
