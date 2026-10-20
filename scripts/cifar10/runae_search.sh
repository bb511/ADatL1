#!/usr/bin/env bash
# ========================================================================
# AE POOL COMMANDS
# ========================================================================
# The random pool of cifar10/ae: 90 trials, drawn by four drivers that share
# the Optuna study `pool` and differ only in their sampler seed. Every trial
# is a full 200-epoch training that dumps its scores and weights at every
# epoch and selects nothing; the criteria are evaluated on the dumps offline.
#
# The random sampler replays its draws from the seed, so a driver that stops
# early is relaunched with a fresh seed, not its own, which would repeat the
# configurations already trained. scripts/cluster/sweep_status.py prints
# such a relaunch.
#
# The blocks name no launcher: the venue appends one at submission, with a
# timeout of 27 minutes per trial for this model (200 epochs at up to 4.5 s
# each on an H100, plus about 3 minutes of evaluation, and half again as
# margin). These times were measured before the score dump: the end-of-run
# evaluation no longer runs, and the dump's per-epoch pass over the test,
# auxiliary and training sets is not in them.
#
# On clariden, each driver keeps six trials queued as slurm jobs:
#   bash scripts/cluster/submit_search.sh scripts/cifar10/runae_search.sh \
#       hydra/launcher=submitit_slurm_clariden hydra.sweeper.n_jobs=6 \
#       hydra.launcher.timeout_min=27
#
# On a local GPU host or an NGT pod, Hydra's basic launcher runs a driver's
# trials in-process, one after another:
#   bash scripts/cluster/submit_search.sh scripts/cifar10/runae_search.sh
# or, with each trial in its own process and under the timeout:
#   bash scripts/cluster/submit_search.sh scripts/cifar10/runae_search.sh \
#       hydra/launcher=submitit_local hydra.launcher.gpus_per_node=1 \
#       hydra.launcher.timeout_min=27
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
#     experiment=cifar10/ae \
#     logger=mlflow \
#     hparams_search=cifar10/ae_optuna \
#     hydra.sweeper.n_trials=23 \
#     hydra.sweeper.sampler.seed=2011 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 2 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=cifar10/ae \
#     logger=mlflow \
#     hparams_search=cifar10/ae_optuna \
#     hydra.sweeper.n_trials=23 \
#     hydra.sweeper.sampler.seed=2012 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 3 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=cifar10/ae \
#     logger=mlflow \
#     hparams_search=cifar10/ae_optuna \
#     hydra.sweeper.n_trials=22 \
#     hydra.sweeper.sampler.seed=2013 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 4 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=cifar10/ae \
#     logger=mlflow \
#     hparams_search=cifar10/ae_optuna \
#     hydra.sweeper.n_trials=22 \
#     hydra.sweeper.sampler.seed=2014 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]
