#!/usr/bin/env bash
# ========================================================================
# VAE LATTICE COMMANDS (WIND TUNNEL)
# ========================================================================
# The random lattice of cchamber_wt/vae has 200 points, drawn by
# four drivers of 50 trials that share the Optuna study `pool` and differ only
# in their sampler seed. Every trial is a full 150-epoch training that dumps its
# scores and weights at every epoch and selects nothing; the criteria are
# evaluated on the dumps offline. The space is
#   configs/hparams_search/cchamber_wt/vae_optuna.yaml
#
# The random sampler replays its draws from the seed, so a driver that stops
# early is relaunched with a fresh seed, not its own, which would repeat the
# configurations already trained. scripts/cluster/sweep_status.py prints
# such a relaunch.
#
# The blocks name no launcher: the venue appends one at submission, with a
# timeout of 45 minutes per trial. The timeout is an estimate from the pools of
# 19 Oct, whose 204 wind-tunnel fits took 145 minutes at 16 concurrent fits on
# two H100s (about 11 minutes per fit on average), and leaves a wide margin.
#
# On clariden, each driver keeps six trials queued as slurm jobs:
#   bash scripts/cluster/submit_search.sh scripts/cchamber_wt/runvae_search.sh \
#       hydra/launcher=submitit_slurm_clariden hydra.sweeper.n_jobs=6 \
#       hydra.launcher.timeout_min=45
#
# On a local GPU host or an NGT pod, Hydra's basic launcher runs a driver's
# trials in-process, one after another:
#   bash scripts/cluster/submit_search.sh scripts/cchamber_wt/runvae_search.sh
# or, with each trial in its own process and under the timeout:
#   bash scripts/cluster/submit_search.sh scripts/cchamber_wt/runvae_search.sh \
#       hydra/launcher=submitit_local hydra.launcher.gpus_per_node=1 \
#       hydra.launcher.timeout_min=45
# submitit_local shows a trial GPUs 0 to gpus_per_node-1 only, none when it
# is unset, and trainer.devices picks among them.
#
# The dormant blocks after the drivers are further instances of the same
# lattice, run only on the user's allocation and only if the theory asks for
# pairings across training seeds or data instances. They repeat the four
# sampler seeds and therefore the points of every live driver that ran without
# a relaunch. The second-training-seed blocks train every point again from
# another initialisation (seed=456, study pool_seed456); the training-halves
# blocks train it on one half of the training rows each, half 0 with the
# experiment's seed and half 1 with seed 456, so that the two halves differ in
# seed and rows alike (studies half0 and half1, data.train_fold 0 and 1). A
# dormant block opens with `## python3`, which scripts/cluster/lib.sh does not
# parse, and removing the extra `#` of that line activates it. Submit an
# activated block with --only and its index, since a submission without --only
# launches the four live drivers again.


# ========================================================================
# LATTICE DRIVERS
# ========================================================================
# ------------------------------------------------------------------------
# Driver 1 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/vae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/vae_optuna \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5021 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 2 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/vae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/vae_optuna \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5022 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 3 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/vae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/vae_optuna \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5023 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 4 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/vae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/vae_optuna \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5024 \
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
#     experiment=cchamber_wt/vae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/vae_optuna \
#     hydra.sweeper.study_name=pool_seed456 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5021 \
#     seed=456 \
#     experiment_name=cchamber_wt_vae_pool_seed456 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Seed 456, driver 2 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/vae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/vae_optuna \
#     hydra.sweeper.study_name=pool_seed456 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5022 \
#     seed=456 \
#     experiment_name=cchamber_wt_vae_pool_seed456 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Seed 456, driver 3 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/vae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/vae_optuna \
#     hydra.sweeper.study_name=pool_seed456 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5023 \
#     seed=456 \
#     experiment_name=cchamber_wt_vae_pool_seed456 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Seed 456, driver 4 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/vae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/vae_optuna \
#     hydra.sweeper.study_name=pool_seed456 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5024 \
#     seed=456 \
#     experiment_name=cchamber_wt_vae_pool_seed456 \
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
#     experiment=cchamber_wt/vae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/vae_optuna \
#     hydra.sweeper.study_name=half0 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5021 \
#     data.train_fold=0 \
#     experiment_name=cchamber_wt_vae_half0 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Half 0, driver 2 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/vae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/vae_optuna \
#     hydra.sweeper.study_name=half0 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5022 \
#     data.train_fold=0 \
#     experiment_name=cchamber_wt_vae_half0 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Half 0, driver 3 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/vae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/vae_optuna \
#     hydra.sweeper.study_name=half0 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5023 \
#     data.train_fold=0 \
#     experiment_name=cchamber_wt_vae_half0 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Half 0, driver 4 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/vae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/vae_optuna \
#     hydra.sweeper.study_name=half0 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5024 \
#     data.train_fold=0 \
#     experiment_name=cchamber_wt_vae_half0 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Half 1, driver 1 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/vae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/vae_optuna \
#     hydra.sweeper.study_name=half1 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5021 \
#     data.train_fold=1 \
#     seed=456 \
#     experiment_name=cchamber_wt_vae_half1 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Half 1, driver 2 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/vae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/vae_optuna \
#     hydra.sweeper.study_name=half1 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5022 \
#     data.train_fold=1 \
#     seed=456 \
#     experiment_name=cchamber_wt_vae_half1 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Half 1, driver 3 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/vae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/vae_optuna \
#     hydra.sweeper.study_name=half1 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5023 \
#     data.train_fold=1 \
#     seed=456 \
#     experiment_name=cchamber_wt_vae_half1 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Half 1, driver 4 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=cchamber_wt/vae \
#     logger=mlflow \
#     hparams_search=cchamber_wt/vae_optuna \
#     hydra.sweeper.study_name=half1 \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=5024 \
#     data.train_fold=1 \
#     seed=456 \
#     experiment_name=cchamber_wt_vae_half1 \
#     trainer=gpu \
#     trainer.max_epochs=150 \
#     trainer.devices=[0]
