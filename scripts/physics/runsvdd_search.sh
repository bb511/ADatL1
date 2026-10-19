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
# The blocks name no launcher: the venue appends one at submission, with a
# timeout of 60 minutes per trial for this model (200 epochs at 11 s each on
# an H100, plus about 3 minutes of evaluation, and half again as margin).
#
# On clariden, each driver keeps six trials queued as slurm jobs:
#   bash scripts/cluster/submit_search.sh scripts/physics/runsvdd_search.sh \
#       hydra/launcher=submitit_slurm_clariden hydra.sweeper.n_jobs=6 \
#       hydra.launcher.timeout_min=60
#
# On a local GPU host or an NGT pod, Hydra's basic launcher runs a driver's
# trials in-process, one after another:
#   bash scripts/cluster/submit_search.sh scripts/physics/runsvdd_search.sh
# or, with each trial in its own process and under the timeout:
#   bash scripts/cluster/submit_search.sh scripts/physics/runsvdd_search.sh \
#       hydra/launcher=submitit_local hydra.launcher.gpus_per_node=1 \
#       hydra.launcher.timeout_min=60
# submitit_local shows a trial GPUs 0 to gpus_per_node-1 only, none when it
# is unset, and trainer.devices picks among them.


# ========================================================================
# POOL DRIVERS
# ========================================================================
# ------------------------------------------------------------------------
# Driver 1 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
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
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.n_trials=40 \
#     hydra.sweeper.sampler.seed=1044 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]
