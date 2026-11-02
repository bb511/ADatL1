#!/usr/bin/env bash
# ========================================================================
# SVDD POOL COMMANDS
# ========================================================================
# - Random pool of physics/svdd: 196 trials of 200 epochs, in the Optuna
#   study `pool` over configs/hparams_search/physics/svdd_optuna.yaml.
# - 48 drivers, both waves running together. Wave 1 is drivers 1 to 24:
#   4 trials each (96 in all).
#   Wave 2 is drivers 25 to 48, with fresh seeds:
#   5 trials for drivers 25 to 28, 4 for 29 to 48 (100 in all).
# - A driver holds at most 6 trials, so that it ends within 11 h at half the
#   V100 speed: 6 trials x 2 x 51.2 min (one trial on an idle V100 at the
#   largest configuration, 200 epochs of 12.9 s plus 2.4 min of start-up and
#   5.8 of evaluation) stay under 660 min.
# - One sqlite study file per driver, logs/optuna/physics/svdd_pool/<seed>.db, so
#   no file has two writers; the URL carries no busy timeout, since Hydra's
#   override grammar does not take an unquoted '=' and Python's default 5 s
#   covers a reader.
# - Each trial checkpoints every validation strategy at both rate tiers, dumps
#   its weights and probe scores at every epoch, and is evaluated at those
#   checkpoints.
# - Timeout per trial: 73 min, measured on an idle V100 at the largest configuration
#   (half again 200 epochs of 12.9 s, plus 2.4 min of start-up and 5.8 of evaluation).
# - Per driver: 3 threads on a GPU host such as olgpu, 11 on a clariden node of
#   24 drivers (OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS,
#   NUMEXPR_NUM_THREADS), and about 31 GiB of RAM.
# - A driver that stops early is relaunched with a fresh seed, which
#   scripts/cluster/sweep_status.py prints.
# - The blocks name no launcher; scripts/cluster/submit_search.sh appends the
#   venue's to every driver. On clariden, one whole-node job per 24 drivers,
#   with the record cache built beforehand and HF_HUB_OFFLINE=1:
#     sbatch scripts/cluster/clariden_node.sbatch --only $(seq -s, 1 24) \
#         scripts/physics/runsvdd_search.sh
#   then --only $(seq -s, 25 48).
#   On a GPU host, one process per trial (here drivers 1 to 4 on four GPUs):
#     bash scripts/cluster/submit_search.sh --only $(seq -s, 1 4) \
#         scripts/physics/runsvdd_search.sh hydra/launcher=submitit_local \
#         hydra.launcher.gpus_per_node=1 hydra.launcher.timeout_min=73


# ========================================================================
# WAVE 1 DRIVERS
# ========================================================================
# ------------------------------------------------------------------------
# Driver 1 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14001.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14001 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 2 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14002.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14002 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 3 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14003.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14003 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 4 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14004.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14004 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 5 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14005.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14005 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 6 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14006.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14006 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 7 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14007.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14007 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 8 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14008.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14008 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 9 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14009.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14009 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 10 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14010.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14010 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 11 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14011.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14011 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 12 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14012.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14012 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 13 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14013.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14013 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 14 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14014.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14014 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 15 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14015.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14015 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 16 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14016.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14016 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 17 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14017.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14017 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 18 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14018.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14018 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 19 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14019.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14019 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 20 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14020.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14020 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 21 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14021.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14021 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 22 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14022.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14022 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 23 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14023.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14023 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 24 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14024.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14024 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ========================================================================
# WAVE 2 DRIVERS
# ========================================================================
# ------------------------------------------------------------------------
# Driver 25 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14025.db' \
#     hydra.sweeper.n_trials=5 \
#     hydra.sweeper.sampler.seed=14025 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 26 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14026.db' \
#     hydra.sweeper.n_trials=5 \
#     hydra.sweeper.sampler.seed=14026 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 27 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14027.db' \
#     hydra.sweeper.n_trials=5 \
#     hydra.sweeper.sampler.seed=14027 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 28 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14028.db' \
#     hydra.sweeper.n_trials=5 \
#     hydra.sweeper.sampler.seed=14028 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 29 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14029.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14029 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 30 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14030.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14030 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 31 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14031.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14031 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 32 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14032.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14032 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 33 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14033.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14033 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 34 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14034.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14034 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 35 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14035.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14035 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 36 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14036.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14036 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 37 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14037.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14037 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 38 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14038.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14038 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 39 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14039.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14039 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 40 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14040.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14040 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 41 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14041.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14041 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 42 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14042.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14042 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 43 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14043.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14043 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 44 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14044.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14044 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 45 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14045.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14045 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 46 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14046.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14046 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 47 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14047.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14047 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 48 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/svdd \
#     logger=mlflow \
#     hparams_search=physics/svdd_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/svdd_pool/14048.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=14048 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]
