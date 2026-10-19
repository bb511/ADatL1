#!/usr/bin/env bash
# ========================================================================
# SVDD POOL COMMANDS
# ========================================================================
# The random pool of cifar10/svdd: 90 trials, drawn by four drivers that
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


# ========================================================================
# POOL DRIVERS
# ========================================================================
# ------------------------------------------------------------------------
# Driver 1 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_clariden \
#     hydra.launcher.timeout_min=200 \
#     hydra.sweeper.n_jobs=6 \
#     experiment=cifar10/svdd \
#     logger=mlflow \
#     hparams_search=cifar10/svdd_optuna \
#     hydra.sweeper.n_trials=23 \
#     hydra.sweeper.sampler.seed=2041 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 2 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_clariden \
#     hydra.launcher.timeout_min=200 \
#     hydra.sweeper.n_jobs=6 \
#     experiment=cifar10/svdd \
#     logger=mlflow \
#     hparams_search=cifar10/svdd_optuna \
#     hydra.sweeper.n_trials=23 \
#     hydra.sweeper.sampler.seed=2042 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 3 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_clariden \
#     hydra.launcher.timeout_min=200 \
#     hydra.sweeper.n_jobs=6 \
#     experiment=cifar10/svdd \
#     logger=mlflow \
#     hparams_search=cifar10/svdd_optuna \
#     hydra.sweeper.n_trials=22 \
#     hydra.sweeper.sampler.seed=2043 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 4 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_clariden \
#     hydra.launcher.timeout_min=200 \
#     hydra.sweeper.n_jobs=6 \
#     experiment=cifar10/svdd \
#     logger=mlflow \
#     hparams_search=cifar10/svdd_optuna \
#     hydra.sweeper.n_trials=22 \
#     hydra.sweeper.sampler.seed=2044 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]
