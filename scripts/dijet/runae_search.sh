#!/usr/bin/env bash
# ========================================================================
# AE LATTICE COMMANDS (DIJET)
# ========================================================================
# - Random lattice of dijet/ae: 200 trials of 200 epochs, in the Optuna
#   study `pool` over configs/hparams_search/dijet/ae_optuna.yaml.
# - Four drivers of 50 trials each.
# - Each trial dumps its scores and weights at every epoch and selects nothing;
#   src/eval.py evaluates the dumps offline.
# - Timeout per trial: 60 min, ceil(1.5 x 200 x 11.9 s + 19 s), the epoch
#   interval of the space's largest configuration on one V100S (11 Nov 2026,
#   olgpu under a load of about 70 on 64 cores).
# - A driver that stops early is relaunched with a fresh seed, which
#   scripts/cluster/sweep_status.py prints.
# - The blocks name no launcher; scripts/cluster/submit_search.sh appends the
#   venue's to every driver, e.g. one process per trial on a GPU host:
#     bash scripts/cluster/submit_search.sh scripts/dijet/runae_search.sh \
#         hydra/launcher=submitit_local hydra.launcher.gpus_per_node=1 \
#         hydra.launcher.timeout_min=60


# ========================================================================
# LATTICE DRIVERS
# ========================================================================
# ------------------------------------------------------------------------
# Driver 1 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=dijet/ae \
#     logger=mlflow \
#     hparams_search=dijet/ae_optuna \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=6011 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 2 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=dijet/ae \
#     logger=mlflow \
#     hparams_search=dijet/ae_optuna \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=6012 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 3 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=dijet/ae \
#     logger=mlflow \
#     hparams_search=dijet/ae_optuna \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=6013 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 4 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=dijet/ae \
#     logger=mlflow \
#     hparams_search=dijet/ae_optuna \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=6014 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]
