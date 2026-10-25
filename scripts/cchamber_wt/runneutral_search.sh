#!/usr/bin/env bash
# ========================================================================
# NEUTRAL AD LATTICE COMMANDS (WIND TUNNEL)
# ========================================================================
# - Random lattice of cchamber_wt/neutral: 200 trials of 150 epochs, in the Optuna
#   study `pool` over configs/hparams_search/cchamber_wt/neutral_optuna.yaml.
# - Four drivers of 50 trials each.
# - Each trial dumps its scores and weights at every epoch and selects nothing;
#   src/eval.py evaluates the dumps offline.
# - Timeout per trial: 45 min. The largest configuration (K = 16, encoder
#   [64, 32, 16]) alone on a V100S takes 2.08 s per epoch and 10 s to start, so
#   ceil(1.5 x 150 x t_epoch + startup) = 8 min; the margin covers 8-16 concurrent
#   fits per GPU, as in the other chamber catalogues.
# - A driver that stops early is relaunched with a fresh seed, which
#   scripts/cluster/sweep_status.py prints.
# - The blocks name no launcher; scripts/cluster/submit_search.sh appends the
#   venue's to every driver, e.g. one process per trial on a GPU host:
#     bash scripts/cluster/submit_search.sh scripts/cchamber_wt/runneutral_search.sh \
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
#     experiment=cchamber_wt/neutral \
#     logger=mlflow \
#     hparams_search=cchamber_wt/neutral_optuna \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5071 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 2 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/neutral \
#     logger=mlflow \
#     hparams_search=cchamber_wt/neutral_optuna \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5072 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 3 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/neutral \
#     logger=mlflow \
#     hparams_search=cchamber_wt/neutral_optuna \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5073 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 4 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/neutral \
#     logger=mlflow \
#     hparams_search=cchamber_wt/neutral_optuna \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5074 \
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
#     experiment=cchamber_wt/neutral \
#     logger=mlflow \
#     hparams_search=cchamber_wt/neutral_optuna \
#     hydra.sweeper.study_name=pool_seed456 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5071 \
#     seed=456 \
#     experiment_name=cchamber_wt_neutral_pool_seed456 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Seed 456, driver 2 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/neutral \
#     logger=mlflow \
#     hparams_search=cchamber_wt/neutral_optuna \
#     hydra.sweeper.study_name=pool_seed456 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5072 \
#     seed=456 \
#     experiment_name=cchamber_wt_neutral_pool_seed456 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Seed 456, driver 3 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/neutral \
#     logger=mlflow \
#     hparams_search=cchamber_wt/neutral_optuna \
#     hydra.sweeper.study_name=pool_seed456 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5073 \
#     seed=456 \
#     experiment_name=cchamber_wt_neutral_pool_seed456 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Seed 456, driver 4 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/neutral \
#     logger=mlflow \
#     hparams_search=cchamber_wt/neutral_optuna \
#     hydra.sweeper.study_name=pool_seed456 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5074 \
#     seed=456 \
#     experiment_name=cchamber_wt_neutral_pool_seed456 \
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
#     experiment=cchamber_wt/neutral \
#     logger=mlflow \
#     hparams_search=cchamber_wt/neutral_optuna \
#     hydra.sweeper.study_name=half0 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5071 \
#     data.train_fold=0 \
#     experiment_name=cchamber_wt_neutral_half0 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Half 0, driver 2 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/neutral \
#     logger=mlflow \
#     hparams_search=cchamber_wt/neutral_optuna \
#     hydra.sweeper.study_name=half0 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5072 \
#     data.train_fold=0 \
#     experiment_name=cchamber_wt_neutral_half0 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Half 0, driver 3 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/neutral \
#     logger=mlflow \
#     hparams_search=cchamber_wt/neutral_optuna \
#     hydra.sweeper.study_name=half0 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5073 \
#     data.train_fold=0 \
#     experiment_name=cchamber_wt_neutral_half0 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Half 0, driver 4 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/neutral \
#     logger=mlflow \
#     hparams_search=cchamber_wt/neutral_optuna \
#     hydra.sweeper.study_name=half0 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5074 \
#     data.train_fold=0 \
#     experiment_name=cchamber_wt_neutral_half0 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Half 1, driver 1 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/neutral \
#     logger=mlflow \
#     hparams_search=cchamber_wt/neutral_optuna \
#     hydra.sweeper.study_name=half1 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5071 \
#     data.train_fold=1 \
#     seed=456 \
#     experiment_name=cchamber_wt_neutral_half1 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Half 1, driver 2 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/neutral \
#     logger=mlflow \
#     hparams_search=cchamber_wt/neutral_optuna \
#     hydra.sweeper.study_name=half1 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5072 \
#     data.train_fold=1 \
#     seed=456 \
#     experiment_name=cchamber_wt_neutral_half1 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Half 1, driver 3 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/neutral \
#     logger=mlflow \
#     hparams_search=cchamber_wt/neutral_optuna \
#     hydra.sweeper.study_name=half1 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5073 \
#     data.train_fold=1 \
#     seed=456 \
#     experiment_name=cchamber_wt_neutral_half1 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Half 1, driver 4 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/neutral \
#     logger=mlflow \
#     hparams_search=cchamber_wt/neutral_optuna \
#     hydra.sweeper.study_name=half1 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5074 \
#     data.train_fold=1 \
#     seed=456 \
#     experiment_name=cchamber_wt_neutral_half1 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]
