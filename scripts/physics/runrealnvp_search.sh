#!/usr/bin/env bash
# ========================================================================
# RealNVP POOL COMMANDS
# ========================================================================
# - Random pool of physics/realnvp: 190 trials of 200 epochs, in the Optuna
#   study `pool` over configs/hparams_search/physics/realnvp_optuna.yaml.
# - Wave 1 is drivers 1 to 4 (23, 23, 22 and 22 trials); wave 2, drivers 5 to 8
#   (25 trials each, fresh seeds), is launched after it with --only 5,6,7,8.
# - Each trial checkpoints every validation strategy at both rate tiers and is
#   evaluated at those checkpoints.
# - Timeout per trial: 142 min, measured on an idle V100 at the largest configuration
#   (half again 200 epochs of 26.3 s, plus 2.2 min of start-up and 7.3 of evaluation).
# - Per driver: 3 threads (OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS,
#   NUMEXPR_NUM_THREADS = 3) and about 31 GiB of RAM.
# - A driver that stops early is relaunched with a fresh seed, which
#   scripts/cluster/sweep_status.py prints.
# - The blocks name no launcher; scripts/cluster/submit_search.sh appends the
#   venue's to every driver, e.g. one process per trial on a GPU host:
#     bash scripts/cluster/submit_search.sh --only 1,2,3,4 \
#         scripts/physics/runrealnvp_search.sh hydra/launcher=submitit_local \
#         hydra.launcher.gpus_per_node=1 hydra.launcher.timeout_min=142


# ========================================================================
# WAVE 1 DRIVERS
# ========================================================================
# ------------------------------------------------------------------------
# Driver 1 of 8 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.n_trials=23 \
#     hydra.sweeper.sampler.seed=1051 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 2 of 8 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.n_trials=23 \
#     hydra.sweeper.sampler.seed=1052 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 3 of 8 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.n_trials=22 \
#     hydra.sweeper.sampler.seed=1053 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 4 of 8 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.n_trials=22 \
#     hydra.sweeper.sampler.seed=1054 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ========================================================================
# WAVE 2 DRIVERS
# ========================================================================
# ------------------------------------------------------------------------
# Driver 5 of 8 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.n_trials=25 \
#     hydra.sweeper.sampler.seed=1055 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 6 of 8 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.n_trials=25 \
#     hydra.sweeper.sampler.seed=1056 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 7 of 8 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.n_trials=25 \
#     hydra.sweeper.sampler.seed=1057 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 8 of 8 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.n_trials=25 \
#     hydra.sweeper.sampler.seed=1058 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]
