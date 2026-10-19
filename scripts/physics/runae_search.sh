#!/usr/bin/env bash
# ========================================================================
# AE POOL COMMANDS
# ========================================================================
# The random pool of physics/ae: 160 trials, drawn by four drivers that
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
# Each driver runs one trial at a time through submitit_local, whose default
# timeout of 60 minutes is shorter than 200 epochs of the slower models,
# hence timeout_min=200.


# ========================================================================
# POOL DRIVERS
# ========================================================================
# ------------------------------------------------------------------------
# Driver 1 of 4
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.timeout_min=200 \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.n_trials=40 \
#     hydra.sweeper.sampler.seed=1011 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 2 of 4
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.timeout_min=200 \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.n_trials=40 \
#     hydra.sweeper.sampler.seed=1012 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 3 of 4
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.timeout_min=200 \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.n_trials=40 \
#     hydra.sweeper.sampler.seed=1013 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 4 of 4
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.timeout_min=200 \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.n_trials=40 \
#     hydra.sweeper.sampler.seed=1014 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]
