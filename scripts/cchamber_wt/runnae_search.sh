#!/usr/bin/env bash
# ========================================================================
# NAE LATTICE COMMANDS (WIND TUNNEL)
# ========================================================================
# - Not for launch (25 Oct): the sampler failed the label-free check here. Of 30
#   full-length calibration runs, unadjusted or Metropolis-adjusted, one met the
#   criteria. The others failed mostly because a classifier told the negatives from
#   validation normal (MLP AUROC up to 0.96): they cross the floors at which
#   signal_1, signal_2 and mic sit. NAE runs on the light tunnel alone.
# - Random lattice of cchamber_wt/nae: 200 trials of 150 epochs, in the Optuna
#   study `pool` over configs/hparams_search/cchamber_wt/nae_optuna.yaml.
# - Four drivers of 50 trials each.
# - Each trial dumps its scores and weights at every epoch and selects nothing;
#   src/eval.py evaluates the dumps offline.
# - Timeout per trial: 30 min, a margin over 19 min, 1.5 times the measured fit plus
#   start-up (25 Oct, one V100, one run: 40 plain-AE epochs of 1.4-2.0 s and 110
#   energy epochs of 5.0-6.1 s).
# - A driver that stops early is relaunched with a fresh seed, which
#   scripts/cluster/sweep_status.py prints.
# - The blocks name no launcher; scripts/cluster/submit_search.sh appends the
#   venue's to every driver, e.g. one process per trial on a GPU host:
#     bash scripts/cluster/submit_search.sh scripts/cchamber_wt/runnae_search.sh \
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
#     experiment=cchamber_wt/nae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/nae_optuna \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5081 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 2 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/nae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/nae_optuna \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5082 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 3 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/nae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/nae_optuna \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5083 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 4 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/nae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/nae_optuna \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5084 \
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
#     experiment=cchamber_wt/nae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/nae_optuna \
#     hydra.sweeper.study_name=pool_seed456 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5081 \
#     seed=456 \
#     experiment_name=cchamber_wt_nae_pool_seed456 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Seed 456, driver 2 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/nae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/nae_optuna \
#     hydra.sweeper.study_name=pool_seed456 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5082 \
#     seed=456 \
#     experiment_name=cchamber_wt_nae_pool_seed456 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Seed 456, driver 3 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/nae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/nae_optuna \
#     hydra.sweeper.study_name=pool_seed456 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5083 \
#     seed=456 \
#     experiment_name=cchamber_wt_nae_pool_seed456 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Seed 456, driver 4 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/nae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/nae_optuna \
#     hydra.sweeper.study_name=pool_seed456 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5084 \
#     seed=456 \
#     experiment_name=cchamber_wt_nae_pool_seed456 \
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
#     experiment=cchamber_wt/nae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/nae_optuna \
#     hydra.sweeper.study_name=half0 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5081 \
#     data.train_fold=0 \
#     experiment_name=cchamber_wt_nae_half0 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Half 0, driver 2 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/nae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/nae_optuna \
#     hydra.sweeper.study_name=half0 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5082 \
#     data.train_fold=0 \
#     experiment_name=cchamber_wt_nae_half0 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Half 0, driver 3 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/nae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/nae_optuna \
#     hydra.sweeper.study_name=half0 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5083 \
#     data.train_fold=0 \
#     experiment_name=cchamber_wt_nae_half0 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Half 0, driver 4 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/nae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/nae_optuna \
#     hydra.sweeper.study_name=half0 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5084 \
#     data.train_fold=0 \
#     experiment_name=cchamber_wt_nae_half0 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Half 1, driver 1 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/nae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/nae_optuna \
#     hydra.sweeper.study_name=half1 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5081 \
#     data.train_fold=1 \
#     seed=456 \
#     experiment_name=cchamber_wt_nae_half1 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Half 1, driver 2 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/nae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/nae_optuna \
#     hydra.sweeper.study_name=half1 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5082 \
#     data.train_fold=1 \
#     seed=456 \
#     experiment_name=cchamber_wt_nae_half1 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Half 1, driver 3 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/nae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/nae_optuna \
#     hydra.sweeper.study_name=half1 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5083 \
#     data.train_fold=1 \
#     seed=456 \
#     experiment_name=cchamber_wt_nae_half1 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Half 1, driver 4 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/nae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/nae_optuna \
#     hydra.sweeper.study_name=half1 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5084 \
#     data.train_fold=1 \
#     seed=456 \
#     experiment_name=cchamber_wt_nae_half1 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]
