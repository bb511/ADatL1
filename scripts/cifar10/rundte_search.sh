#!/usr/bin/env bash
# ========================================================================
# DTE POOL COMMANDS
# ========================================================================
# The random pool of cifar10/dte: 110 trials, drawn by four drivers that
# share the Optuna study `pool` and differ only in their sampler seed. Every
# trial is a full 200-epoch training, evaluated at the checkpoint of each
# validation strategy, and the strategies then select from the pool.
#
# The random sampler replays its draws from the seed, so a driver that stops
# early is relaunched with a fresh seed, not its own, which would repeat the
# configurations already trained. scripts/cluster/sweep_status.py prints
# such a relaunch.
#
# Each driver keeps six trials queued as slurm jobs
# (hydra.sweeper.n_jobs=6).
#
# BEFORE THE FIRST LAUNCH the processed data must already exist where the
# trials run. Build it with rundte_cache.sh in this directory and copy the
# trees listed there.
#
# The search space is configs/hparams_search/cifar10/dte_optuna.yaml and
# holds the CIFAR-10 ranges only. Its RobustAD counterpart is a separate
# file under configs/hparams_search/robustad/.


# ========================================================================
# POOL DRIVERS
# ========================================================================
# ------------------------------------------------------------------------
# Driver 1 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_clariden \
#     hydra.launcher.timeout_min=240 \
#     hydra.sweeper.n_jobs=6 \
#     experiment=cifar10/dte \
#     logger=mlflow \
#     hparams_search=cifar10/dte_optuna \
#     hydra.sweeper.n_trials=28 \
#     hydra.sweeper.sampler.seed=2061 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 2 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_clariden \
#     hydra.launcher.timeout_min=240 \
#     hydra.sweeper.n_jobs=6 \
#     experiment=cifar10/dte \
#     logger=mlflow \
#     hparams_search=cifar10/dte_optuna \
#     hydra.sweeper.n_trials=28 \
#     hydra.sweeper.sampler.seed=2062 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 3 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_clariden \
#     hydra.launcher.timeout_min=240 \
#     hydra.sweeper.n_jobs=6 \
#     experiment=cifar10/dte \
#     logger=mlflow \
#     hparams_search=cifar10/dte_optuna \
#     hydra.sweeper.n_trials=27 \
#     hydra.sweeper.sampler.seed=2063 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 4 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_clariden \
#     hydra.launcher.timeout_min=240 \
#     hydra.sweeper.n_jobs=6 \
#     experiment=cifar10/dte \
#     logger=mlflow \
#     hparams_search=cifar10/dte_optuna \
#     hydra.sweeper.n_trials=27 \
#     hydra.sweeper.sampler.seed=2064 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]
