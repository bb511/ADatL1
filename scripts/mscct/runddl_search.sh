#!/usr/bin/env bash
# ========================================================================
# DDL POOL COMMANDS
# ========================================================================
# - Random pool of mscct/ddl: 90 trials of 200 epochs, in the Optuna
#   study `pool` over configs/hparams_search/mscct/ddl_optuna.yaml.
# - Four drivers of 23, 23, 22 and 22 trials.
# - Each trial dumps its scores and weights at every epoch and selects nothing;
#   src/eval.py evaluates the dumps offline.
# - Timeout per trial: 156 min. The space's largest configuration, U-Net [16, 32, 64]
#   with time_dim 64 at the fixed 4 timesteps, took 19.3 s per epoch and 950 s to
#   start, the SDS pseudo-sets' build (about 15 min on one core, from 10,000 of the
#   31,238 training patches) included, on an idle V100S (1 Nov), so
#   ceil(1.5 x 200 x t_epoch + startup) = 113 min; the timeout stays at 156 min, since
#   trials sharing a GPU ran 3-4 times slower in validation (1 Nov). That build raises
#   a trial's peak host memory from 4.3 GB to 14.7 GB.
# - A driver that stops early is relaunched with a fresh seed, which
#   scripts/cluster/sweep_status.py prints.
# - The blocks name no launcher; scripts/cluster/submit_search.sh appends the
#   venue's to every driver, e.g. one process per trial on a GPU host:
#     bash scripts/cluster/submit_search.sh scripts/mscct/runddl_search.sh \
#         hydra/launcher=submitit_local hydra.launcher.gpus_per_node=1 \
#         hydra.launcher.timeout_min=156
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
#     experiment=mscct/ddl \
#     logger=mlflow \
#     hparams_search=mscct/ddl_optuna \
#     hydra.sweeper.n_trials=23 \
#     hydra.sweeper.sampler.seed=3091 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 2 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=mscct/ddl \
#     logger=mlflow \
#     hparams_search=mscct/ddl_optuna \
#     hydra.sweeper.n_trials=23 \
#     hydra.sweeper.sampler.seed=3092 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 3 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=mscct/ddl \
#     logger=mlflow \
#     hparams_search=mscct/ddl_optuna \
#     hydra.sweeper.n_trials=22 \
#     hydra.sweeper.sampler.seed=3093 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 4 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=mscct/ddl \
#     logger=mlflow \
#     hparams_search=mscct/ddl_optuna \
#     hydra.sweeper.n_trials=22 \
#     hydra.sweeper.sampler.seed=3094 \
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
#     experiment=mscct/ddl \
#     logger=mlflow \
#     hparams_search=mscct/ddl_optuna \
#     hydra.sweeper.study_name=pool_seed456 \
#     hydra.sweeper.n_trials=23 \
#     hydra.sweeper.sampler.seed=3091 \
#     seed=456 \
#     experiment_name=mscct_ddl_pool_seed456 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Seed 456, driver 2 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=mscct/ddl \
#     logger=mlflow \
#     hparams_search=mscct/ddl_optuna \
#     hydra.sweeper.study_name=pool_seed456 \
#     hydra.sweeper.n_trials=23 \
#     hydra.sweeper.sampler.seed=3092 \
#     seed=456 \
#     experiment_name=mscct_ddl_pool_seed456 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Seed 456, driver 3 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=mscct/ddl \
#     logger=mlflow \
#     hparams_search=mscct/ddl_optuna \
#     hydra.sweeper.study_name=pool_seed456 \
#     hydra.sweeper.n_trials=22 \
#     hydra.sweeper.sampler.seed=3093 \
#     seed=456 \
#     experiment_name=mscct_ddl_pool_seed456 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Seed 456, driver 4 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=mscct/ddl \
#     logger=mlflow \
#     hparams_search=mscct/ddl_optuna \
#     hydra.sweeper.study_name=pool_seed456 \
#     hydra.sweeper.n_trials=22 \
#     hydra.sweeper.sampler.seed=3094 \
#     seed=456 \
#     experiment_name=mscct_ddl_pool_seed456 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]
