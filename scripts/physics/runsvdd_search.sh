#!/usr/bin/env bash
# ========================================================================
# SVDD POOL COMMANDS
# ========================================================================
# The random pool of physics/svdd: 160 trials, drawn by four drivers that
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
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.n_trials=40 \
#     hydra.sweeper.sampler.seed=1041 \
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
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.n_trials=40 \
#     hydra.sweeper.sampler.seed=1042 \
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
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.n_trials=40 \
#     hydra.sweeper.sampler.seed=1043 \
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
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.n_trials=40 \
#     hydra.sweeper.sampler.seed=1044 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]
