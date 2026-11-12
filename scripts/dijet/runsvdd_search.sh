#!/usr/bin/env bash
# ========================================================================
# SVDD LATTICE COMMANDS (DIJET)
# ========================================================================
# - Random lattice of dijet/svdd: 200 trials of 200 epochs, in the Optuna
#   study `pool` over configs/hparams_search/dijet/svdd_optuna.yaml.
# - Four drivers of 50 trials each.
# - Each trial dumps its scores and weights at every epoch and selects nothing;
#   src/eval.py evaluates the dumps offline.
# - Timeout per trial: 33 min, ceil(1.5 x 200 x 6.4 s + 20 s), the epoch
#   interval of the space's largest configuration on one V100S (11 Nov 2026,
#   olgpu under a load of about 70 on 64 cores).
# - A driver that stops early is relaunched with a fresh seed, which
#   scripts/cluster/sweep_status.py prints.
# - The blocks name no launcher; scripts/cluster/submit_search.sh appends the
#   venue's to every driver, e.g. one process per trial on a GPU host:
#     bash scripts/cluster/submit_search.sh scripts/dijet/runsvdd_search.sh \
#         hydra/launcher=submitit_local hydra.launcher.gpus_per_node=1 \
#         hydra.launcher.timeout_min=33


# ========================================================================
# LATTICE DRIVERS
# ========================================================================
# ------------------------------------------------------------------------
# Driver 1 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=dijet/svdd \
#     logger=mlflow \
#     hparams_search=dijet/svdd_optuna \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=6041 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 2 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=dijet/svdd \
#     logger=mlflow \
#     hparams_search=dijet/svdd_optuna \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=6042 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 3 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=dijet/svdd \
#     logger=mlflow \
#     hparams_search=dijet/svdd_optuna \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=6043 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 4 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=dijet/svdd \
#     logger=mlflow \
#     hparams_search=dijet/svdd_optuna \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=6044 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]
