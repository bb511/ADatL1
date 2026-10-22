#!/usr/bin/env bash
# ========================================================================
# VAE POOL COMMANDS
# ========================================================================
# - Random pool of robustad/vae: 90 trials of 200 epochs, in the Optuna
#   study `pool` over configs/hparams_search/robustad/vae_optuna.yaml.
# - Four drivers of 23, 23, 22 and 22 trials.
# - Each trial dumps its scores and weights at every epoch and selects nothing;
#   src/eval.py evaluates the dumps offline.
# - Timeout per trial: 10 min, measured on an idle V100 on 22 Oct (half again 200
#   epochs, plus start-up); epochs ran 4.5 times slower on a host at load 125.
# - A driver that stops early is relaunched with a fresh seed, which
#   scripts/cluster/sweep_status.py prints.
# - The blocks name no launcher; scripts/cluster/submit_search.sh appends the
#   venue's to every driver, e.g. one process per trial on a GPU host:
#     bash scripts/cluster/submit_search.sh scripts/robustad/runvae_search.sh \
#         hydra/launcher=submitit_local hydra.launcher.gpus_per_node=1 \
#         hydra.launcher.timeout_min=10
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
#     experiment=robustad/vae \
#     logger=mlflow \
#     hparams_search=robustad/vae_optuna \
#     hydra.sweeper.n_trials=23 \
#     hydra.sweeper.sampler.seed=3021 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 2 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=robustad/vae \
#     logger=mlflow \
#     hparams_search=robustad/vae_optuna \
#     hydra.sweeper.n_trials=23 \
#     hydra.sweeper.sampler.seed=3022 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 3 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=robustad/vae \
#     logger=mlflow \
#     hparams_search=robustad/vae_optuna \
#     hydra.sweeper.n_trials=22 \
#     hydra.sweeper.sampler.seed=3023 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 4 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=robustad/vae \
#     logger=mlflow \
#     hparams_search=robustad/vae_optuna \
#     hydra.sweeper.n_trials=22 \
#     hydra.sweeper.sampler.seed=3024 \
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
#     experiment=robustad/vae \
#     logger=mlflow \
#     hparams_search=robustad/vae_optuna \
#     hydra.sweeper.study_name=pool_seed456 \
#     hydra.sweeper.n_trials=23 \
#     hydra.sweeper.sampler.seed=3021 \
#     seed=456 \
#     experiment_name=robustad_vae_pool_seed456 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Seed 456, driver 2 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=robustad/vae \
#     logger=mlflow \
#     hparams_search=robustad/vae_optuna \
#     hydra.sweeper.study_name=pool_seed456 \
#     hydra.sweeper.n_trials=23 \
#     hydra.sweeper.sampler.seed=3022 \
#     seed=456 \
#     experiment_name=robustad_vae_pool_seed456 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Seed 456, driver 3 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=robustad/vae \
#     logger=mlflow \
#     hparams_search=robustad/vae_optuna \
#     hydra.sweeper.study_name=pool_seed456 \
#     hydra.sweeper.n_trials=22 \
#     hydra.sweeper.sampler.seed=3023 \
#     seed=456 \
#     experiment_name=robustad_vae_pool_seed456 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Seed 456, driver 4 of 4
# ------------------------------------------------------------------------
## python3 src/train.py \
#     -m \
#     experiment=robustad/vae \
#     logger=mlflow \
#     hparams_search=robustad/vae_optuna \
#     hydra.sweeper.study_name=pool_seed456 \
#     hydra.sweeper.n_trials=22 \
#     hydra.sweeper.sampler.seed=3024 \
#     seed=456 \
#     experiment_name=robustad_vae_pool_seed456 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]
