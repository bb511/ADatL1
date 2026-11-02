#!/usr/bin/env bash
# ========================================================================
# DDL POOL COMMANDS
# ========================================================================
# - Random pool of cifar10/ddl: 90 trials of 200 epochs, in the Optuna
#   study `pool` over configs/hparams_search/cifar10/ddl_optuna.yaml.
# - Four drivers of 23, 23, 22 and 22 trials.
# - Each trial dumps its scores and weights at every epoch and selects nothing;
#   src/eval.py evaluates the dumps offline.
# - Timeout per trial: 80 min. The space's largest configuration, U-Net [16, 32, 64]
#   with time_dim 64 at the fixed 4 timesteps, took 10.3 s per epoch and 111 s to
#   start, the SDS pseudo-sets' build (about 72 s) included, on an idle V100S (1 Nov),
#   so ceil(1.5 x 200 x t_epoch + startup) = 54 min; the timeout stays at 80 min,
#   since trials sharing a GPU ran 3-4 times slower in validation (1 Nov). That build
#   raises a trial's peak host memory from 2.9 GB to 6.7 GB.
# - A driver that stops early is relaunched with a fresh seed, which
#   scripts/cluster/sweep_status.py prints.
# - The blocks name no launcher; scripts/cluster/submit_search.sh appends the
#   venue's to every driver, e.g. one process per trial on a GPU host:
#     bash scripts/cluster/submit_search.sh scripts/cifar10/runddl_search.sh \
#         hydra/launcher=submitit_local hydra.launcher.gpus_per_node=1 \
#         hydra.launcher.timeout_min=80
# - Dormant `## python3` blocks repeat the live sampler seeds under a second
#   training seed (456).
# - To run a dormant block, delete one `#` and submit it with --only <index>.


# ========================================================================
# POOL DRIVERS
# ========================================================================
# ------------------------------------------------------------------------
# Driver 1 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=cifar10/ddl \
#     logger=mlflow \
#     hparams_search=cifar10/ddl_optuna \
#     hydra.sweeper.n_trials=23 \
#     hydra.sweeper.sampler.seed=2091 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 2 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=cifar10/ddl \
#     logger=mlflow \
#     hparams_search=cifar10/ddl_optuna \
#     hydra.sweeper.n_trials=23 \
#     hydra.sweeper.sampler.seed=2092 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 3 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=cifar10/ddl \
#     logger=mlflow \
#     hparams_search=cifar10/ddl_optuna \
#     hydra.sweeper.n_trials=22 \
#     hydra.sweeper.sampler.seed=2093 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 4 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=cifar10/ddl \
#     logger=mlflow \
#     hparams_search=cifar10/ddl_optuna \
#     hydra.sweeper.n_trials=22 \
#     hydra.sweeper.sampler.seed=2094 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ========================================================================
# DORMANT: SECOND TRAINING SEED
# ========================================================================
# ------------------------------------------------------------------------
# Seed 456, driver 1 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cifar10/ddl \
#     logger=mlflow \
#     hparams_search=cifar10/ddl_optuna \
#     hydra.sweeper.study_name=pool_seed456 \
#     hydra.sweeper.n_trials=23 \
#     hydra.sweeper.sampler.seed=2091 \
#     seed=456 \
#     experiment_name=cifar10_ddl_pool_seed456 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Seed 456, driver 2 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cifar10/ddl \
#     logger=mlflow \
#     hparams_search=cifar10/ddl_optuna \
#     hydra.sweeper.study_name=pool_seed456 \
#     hydra.sweeper.n_trials=23 \
#     hydra.sweeper.sampler.seed=2092 \
#     seed=456 \
#     experiment_name=cifar10_ddl_pool_seed456 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Seed 456, driver 3 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cifar10/ddl \
#     logger=mlflow \
#     hparams_search=cifar10/ddl_optuna \
#     hydra.sweeper.study_name=pool_seed456 \
#     hydra.sweeper.n_trials=22 \
#     hydra.sweeper.sampler.seed=2093 \
#     seed=456 \
#     experiment_name=cifar10_ddl_pool_seed456 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Seed 456, driver 4 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cifar10/ddl \
#     logger=mlflow \
#     hparams_search=cifar10/ddl_optuna \
#     hydra.sweeper.study_name=pool_seed456 \
#     hydra.sweeper.n_trials=22 \
#     hydra.sweeper.sampler.seed=2094 \
#     seed=456 \
#     experiment_name=cifar10_ddl_pool_seed456 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]
