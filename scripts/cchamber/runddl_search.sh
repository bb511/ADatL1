#!/usr/bin/env bash
# ========================================================================
# DDL LATTICE COMMANDS (LIGHT TUNNEL)
# ========================================================================
# - Random lattice of cchamber/ddl: 200 trials of 150 epochs, in the Optuna
#   study `pool` over configs/hparams_search/cchamber/ddl_optuna.yaml.
# - Four drivers of 50 trials each.
# - Each trial dumps its scores and weights at every epoch and selects nothing;
#   src/eval.py evaluates the dumps offline.
# - Timeout per trial: 30 min, the margin of the other light-tunnel catalogues. The
#   space's largest configuration took 2.5 s per epoch and about 8 s to start on a
#   V100S (1 Nov), so ceil(1.5 x 150 x t_epoch + startup) = 10 min; the margin is for
#   contended venues.
# - A driver that stops early is relaunched with a fresh seed, which
#   scripts/cluster/sweep_status.py prints.
# - The blocks name no launcher; scripts/cluster/submit_search.sh appends the
#   venue's to every driver, e.g. one process per trial on a GPU host:
#     bash scripts/cluster/submit_search.sh scripts/cchamber/runddl_search.sh \
#         hydra/launcher=submitit_local hydra.launcher.gpus_per_node=1 \
#         hydra.launcher.timeout_min=30
# - Dormant `## python3` blocks repeat the live sampler seeds under a second
#   training seed (456) and on the two training halves (half 1 at seed 456).
# - To run a dormant block, delete one `#` and submit it with --only <index>.


# ========================================================================
# LATTICE DRIVERS
# ========================================================================
# ------------------------------------------------------------------------
# Driver 1 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=cchamber/ddl \
#     logger=mlflow \
#     hparams_search=cchamber/ddl_optuna \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=4091 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 2 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=cchamber/ddl \
#     logger=mlflow \
#     hparams_search=cchamber/ddl_optuna \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=4092 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 3 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=cchamber/ddl \
#     logger=mlflow \
#     hparams_search=cchamber/ddl_optuna \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=4093 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 4 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=cchamber/ddl \
#     logger=mlflow \
#     hparams_search=cchamber/ddl_optuna \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=4094 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ========================================================================
# DORMANT: SECOND TRAINING SEED
# ========================================================================
# ------------------------------------------------------------------------
# Seed 456, driver 1 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber/ddl \
#     logger=mlflow \
#     hparams_search=cchamber/ddl_optuna \
#     hydra.sweeper.study_name=pool_seed456 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=4091 \
#     seed=456 \
#     experiment_name=cchamber_ddl_pool_seed456 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Seed 456, driver 2 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber/ddl \
#     logger=mlflow \
#     hparams_search=cchamber/ddl_optuna \
#     hydra.sweeper.study_name=pool_seed456 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=4092 \
#     seed=456 \
#     experiment_name=cchamber_ddl_pool_seed456 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Seed 456, driver 3 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber/ddl \
#     logger=mlflow \
#     hparams_search=cchamber/ddl_optuna \
#     hydra.sweeper.study_name=pool_seed456 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=4093 \
#     seed=456 \
#     experiment_name=cchamber_ddl_pool_seed456 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Seed 456, driver 4 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber/ddl \
#     logger=mlflow \
#     hparams_search=cchamber/ddl_optuna \
#     hydra.sweeper.study_name=pool_seed456 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=4094 \
#     seed=456 \
#     experiment_name=cchamber_ddl_pool_seed456 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ========================================================================
# DORMANT: TRAINING HALVES
# ========================================================================
# ------------------------------------------------------------------------
# Half 0, driver 1 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber/ddl \
#     logger=mlflow \
#     hparams_search=cchamber/ddl_optuna \
#     hydra.sweeper.study_name=half0 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=4091 \
#     data.train_fold=0 \
#     experiment_name=cchamber_ddl_half0 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Half 0, driver 2 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber/ddl \
#     logger=mlflow \
#     hparams_search=cchamber/ddl_optuna \
#     hydra.sweeper.study_name=half0 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=4092 \
#     data.train_fold=0 \
#     experiment_name=cchamber_ddl_half0 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Half 0, driver 3 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber/ddl \
#     logger=mlflow \
#     hparams_search=cchamber/ddl_optuna \
#     hydra.sweeper.study_name=half0 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=4093 \
#     data.train_fold=0 \
#     experiment_name=cchamber_ddl_half0 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Half 0, driver 4 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber/ddl \
#     logger=mlflow \
#     hparams_search=cchamber/ddl_optuna \
#     hydra.sweeper.study_name=half0 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=4094 \
#     data.train_fold=0 \
#     experiment_name=cchamber_ddl_half0 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Half 1, driver 1 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber/ddl \
#     logger=mlflow \
#     hparams_search=cchamber/ddl_optuna \
#     hydra.sweeper.study_name=half1 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=4091 \
#     data.train_fold=1 \
#     seed=456 \
#     experiment_name=cchamber_ddl_half1 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Half 1, driver 2 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber/ddl \
#     logger=mlflow \
#     hparams_search=cchamber/ddl_optuna \
#     hydra.sweeper.study_name=half1 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=4092 \
#     data.train_fold=1 \
#     seed=456 \
#     experiment_name=cchamber_ddl_half1 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Half 1, driver 3 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber/ddl \
#     logger=mlflow \
#     hparams_search=cchamber/ddl_optuna \
#     hydra.sweeper.study_name=half1 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=4093 \
#     data.train_fold=1 \
#     seed=456 \
#     experiment_name=cchamber_ddl_half1 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Half 1, driver 4 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber/ddl \
#     logger=mlflow \
#     hparams_search=cchamber/ddl_optuna \
#     hydra.sweeper.study_name=half1 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=4094 \
#     data.train_fold=1 \
#     seed=456 \
#     experiment_name=cchamber_ddl_half1 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]
