#!/usr/bin/env bash
# ========================================================================
# DTE POOL COMMANDS
# ========================================================================
# The random pool of physics/dte: 90 trials, drawn by four drivers that
# share the Optuna study `pool` and differ only in their sampler seed. Every
# trial is a full 200-epoch training, evaluated at the checkpoint of each
# validation strategy, at both background-rate tiers, and the strategies
# then select from the pool.
#
# The random sampler replays its draws from the seed, so a driver that stops
# early is relaunched with a fresh seed, not its own, which would repeat the
# configurations already trained. scripts/cluster/sweep_status.py prints
# such a relaunch.
#
# Each driver keeps six trials queued as slurm jobs
# (hydra.sweeper.n_jobs=6). The clariden launcher defaults to
# timeout_min=60, far too short for physics, hence the explicit 720, which
# is exactly the `normal` partition cap, so a trial that needs longer is
# killed rather than requeued.
#
# BEFORE THE FIRST LAUNCH build the data cache: experiment=physics/dte
# selects the `standard` normaliser, whose cache tree under data/hf/ the
# other physics models do not use. One plain training of
# experiment=physics/dte data=basis_hf builds it; drivers starting against a
# missing cache race on the same directory and leave partial files behind.


# ========================================================================
# POOL DRIVERS
# ========================================================================
# ------------------------------------------------------------------------
# Driver 1 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_clariden \
#     hydra.launcher.timeout_min=720 \
#     hydra.sweeper.n_jobs=6 \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.n_trials=23 \
#     hydra.sweeper.sampler.seed=1061 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 2 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_clariden \
#     hydra.launcher.timeout_min=720 \
#     hydra.sweeper.n_jobs=6 \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.n_trials=23 \
#     hydra.sweeper.sampler.seed=1062 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 3 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_clariden \
#     hydra.launcher.timeout_min=720 \
#     hydra.sweeper.n_jobs=6 \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.n_trials=22 \
#     hydra.sweeper.sampler.seed=1063 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 4 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_clariden \
#     hydra.launcher.timeout_min=720 \
#     hydra.sweeper.n_jobs=6 \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.n_trials=22 \
#     hydra.sweeper.sampler.seed=1064 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]
