#!/usr/bin/env bash
# ========================================================================
# DeepSets AE POOL COMMANDS
# ========================================================================
# - Random pool of physics/dsae: 190 trials of 200 epochs, in the Optuna
#   study `pool` over configs/hparams_search/physics/dsae_optuna.yaml.
# - Wave 1 is drivers 1 to 4 (23, 23, 22 and 22 trials); wave 2, drivers 5 to 8
#   (25 trials each, fresh seeds), is launched after it with --only 5,6,7,8.
# - Each trial checkpoints every validation strategy at both rate tiers and is
#   evaluated at those checkpoints.
# - Timeout per trial: 148 min, measured on an idle V100 at the largest configuration
#   (half again 200 epochs of 27.6 s, plus 2.3 min of start-up and 7.7 of evaluation).
# - Per driver: 3 threads (OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS,
#   NUMEXPR_NUM_THREADS = 3) and about 31 GiB of RAM.
# - A driver that stops early is relaunched with a fresh seed, which
#   scripts/cluster/sweep_status.py prints.
# - The blocks name no launcher; scripts/cluster/submit_search.sh appends the
#   venue's to every driver, e.g. one process per trial on a GPU host:
#     bash scripts/cluster/submit_search.sh --only 1,2,3,4 \
#         scripts/physics/rundsae_search.sh hydra/launcher=submitit_local \
#         hydra.launcher.gpus_per_node=1 hydra.launcher.timeout_min=148


# ========================================================================
# WAVE 1 DRIVERS
# ========================================================================
# ------------------------------------------------------------------------
# Driver 1 of 8 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.n_trials=23 \
#     hydra.sweeper.sampler.seed=1031 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 2 of 8 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.n_trials=23 \
#     hydra.sweeper.sampler.seed=1032 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 3 of 8 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.n_trials=22 \
#     hydra.sweeper.sampler.seed=1033 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 4 of 8 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.n_trials=22 \
#     hydra.sweeper.sampler.seed=1034 \
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
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.n_trials=25 \
#     hydra.sweeper.sampler.seed=1035 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 6 of 8 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.n_trials=25 \
#     hydra.sweeper.sampler.seed=1036 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 7 of 8 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.n_trials=25 \
#     hydra.sweeper.sampler.seed=1037 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 8 of 8 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.n_trials=25 \
#     hydra.sweeper.sampler.seed=1038 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]
