#!/usr/bin/env bash
# ========================================================================
# AE LATTICE COMMANDS (WIND TUNNEL)
# ========================================================================
# - Random lattice of cchamber_wt/ae: 200 trials of 150 epochs, in the Optuna
#   study `pool` over configs/hparams_search/cchamber_wt/ae_optuna.yaml.
# - Four drivers of 50 trials each.
# - Each trial dumps its scores and weights at every epoch and selects nothing;
#   src/eval.py evaluates the dumps offline.
# - Timeout per trial: 45 min, an estimate with a wide margin (the 19 Oct pools took
#   about 11 min per fit, at 16 concurrent fits on two H100s).
# - A driver that stops early is relaunched with a fresh seed, which
#   scripts/cluster/sweep_status.py prints.
# - The blocks name no launcher; scripts/cluster/submit_search.sh appends the
#   venue's to every driver, e.g. one process per trial on a GPU host:
#     bash scripts/cluster/submit_search.sh scripts/cchamber_wt/runae_search.sh \
#         hydra/launcher=submitit_local hydra.launcher.gpus_per_node=1 \
#         hydra.launcher.timeout_min=45
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
#     experiment=cchamber_wt/ae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/ae_optuna \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5011 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 2 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/ae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/ae_optuna \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5012 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 3 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/ae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/ae_optuna \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5013 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 4 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/ae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/ae_optuna \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5014 \
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
#     experiment=cchamber_wt/ae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/ae_optuna \
#     hydra.sweeper.study_name=pool_seed456 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5011 \
#     seed=456 \
#     experiment_name=cchamber_wt_ae_pool_seed456 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Seed 456, driver 2 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/ae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/ae_optuna \
#     hydra.sweeper.study_name=pool_seed456 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5012 \
#     seed=456 \
#     experiment_name=cchamber_wt_ae_pool_seed456 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Seed 456, driver 3 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/ae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/ae_optuna \
#     hydra.sweeper.study_name=pool_seed456 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5013 \
#     seed=456 \
#     experiment_name=cchamber_wt_ae_pool_seed456 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Seed 456, driver 4 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/ae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/ae_optuna \
#     hydra.sweeper.study_name=pool_seed456 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5014 \
#     seed=456 \
#     experiment_name=cchamber_wt_ae_pool_seed456 \
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
#     experiment=cchamber_wt/ae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/ae_optuna \
#     hydra.sweeper.study_name=half0 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5011 \
#     data.train_fold=0 \
#     experiment_name=cchamber_wt_ae_half0 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Half 0, driver 2 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/ae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/ae_optuna \
#     hydra.sweeper.study_name=half0 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5012 \
#     data.train_fold=0 \
#     experiment_name=cchamber_wt_ae_half0 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Half 0, driver 3 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/ae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/ae_optuna \
#     hydra.sweeper.study_name=half0 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5013 \
#     data.train_fold=0 \
#     experiment_name=cchamber_wt_ae_half0 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Half 0, driver 4 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/ae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/ae_optuna \
#     hydra.sweeper.study_name=half0 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5014 \
#     data.train_fold=0 \
#     experiment_name=cchamber_wt_ae_half0 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Half 1, driver 1 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/ae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/ae_optuna \
#     hydra.sweeper.study_name=half1 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5011 \
#     data.train_fold=1 \
#     seed=456 \
#     experiment_name=cchamber_wt_ae_half1 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Half 1, driver 2 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/ae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/ae_optuna \
#     hydra.sweeper.study_name=half1 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5012 \
#     data.train_fold=1 \
#     seed=456 \
#     experiment_name=cchamber_wt_ae_half1 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Half 1, driver 3 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/ae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/ae_optuna \
#     hydra.sweeper.study_name=half1 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5013 \
#     data.train_fold=1 \
#     seed=456 \
#     experiment_name=cchamber_wt_ae_half1 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Half 1, driver 4 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/ae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/ae_optuna \
#     hydra.sweeper.study_name=half1 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5014 \
#     data.train_fold=1 \
#     seed=456 \
#     experiment_name=cchamber_wt_ae_half1 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]
