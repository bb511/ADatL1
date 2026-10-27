#!/usr/bin/env bash
# ========================================================================
# AE POOL COMMANDS
# ========================================================================
# - Random pool of physics/ae: 196 trials of 200 epochs, in the Optuna
#   study `pool` over configs/hparams_search/physics/ae_optuna.yaml.
# - 48 drivers, both waves running together. Wave 1 is drivers 1 to 24:
#   4 trials each (96 in all).
#   Wave 2 is drivers 25 to 48, with fresh seeds:
#   5 trials for drivers 25 to 28, 4 for 29 to 48 (100 in all).
# - A driver holds at most 6 trials, so that it ends within 11 h at half the
#   V100 speed: 6 trials x 2 x 51.9 min (one trial on an idle V100 at the
#   largest configuration, 200 epochs of 13.1 s plus 2.3 min of start-up and
#   5.9 of evaluation) stay under 660 min.
# - One sqlite study file per driver, logs/optuna/physics/ae_pool/<seed>.db, so
#   no file has two writers; the URL carries no busy timeout, since Hydra's
#   override grammar does not take an unquoted '=' and Python's default 5 s
#   covers a reader.
# - Each trial checkpoints every validation strategy at both rate tiers, dumps
#   its weights and probe scores at every epoch, and is evaluated at those
#   checkpoints.
# - Timeout per trial: 74 min, measured on an idle V100 at the largest configuration
#   (half again 200 epochs of 13.1 s, plus 2.3 min of start-up and 5.9 of evaluation).
# - Per driver: 3 threads on a GPU host such as olgpu, 11 on a clariden node of
#   24 drivers (OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS,
#   NUMEXPR_NUM_THREADS), and about 31 GiB of RAM.
# - A driver that stops early is relaunched with a fresh seed, which
#   scripts/cluster/sweep_status.py prints.
# - The blocks name no launcher; scripts/cluster/submit_search.sh appends the
#   venue's to every driver. On clariden, one whole-node job per 24 drivers,
#   with the record cache built beforehand and HF_HUB_OFFLINE=1:
#     sbatch scripts/cluster/clariden_node.sbatch --only $(seq -s, 1 24) \
#         scripts/physics/runae_search.sh
#   then --only $(seq -s, 25 48).
#   On a GPU host, one process per trial (here drivers 1 to 4 on four GPUs):
#     bash scripts/cluster/submit_search.sh --only $(seq -s, 1 4) \
#         scripts/physics/runae_search.sh hydra/launcher=submitit_local \
#         hydra.launcher.gpus_per_node=1 hydra.launcher.timeout_min=74


# ========================================================================
# WAVE 1 DRIVERS
# ========================================================================
# ------------------------------------------------------------------------
# Driver 1 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11001.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11001 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 2 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11002.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11002 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 3 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11003.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11003 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 4 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11004.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11004 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 5 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11005.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11005 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 6 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11006.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11006 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 7 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11007.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11007 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 8 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11008.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11008 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 9 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11009.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11009 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 10 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11010.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11010 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 11 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11011.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11011 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 12 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11012.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11012 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 13 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11013.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11013 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 14 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11014.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11014 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 15 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11015.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11015 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 16 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11016.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11016 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 17 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11017.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11017 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 18 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11018.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11018 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 19 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11019.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11019 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 20 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11020.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11020 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 21 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11021.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11021 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 22 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11022.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11022 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 23 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11023.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11023 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 24 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11024.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11024 \
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
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11025.db' \
#     hydra.sweeper.n_trials=5 \
#     hydra.sweeper.sampler.seed=11025 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 26 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11026.db' \
#     hydra.sweeper.n_trials=5 \
#     hydra.sweeper.sampler.seed=11026 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 27 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11027.db' \
#     hydra.sweeper.n_trials=5 \
#     hydra.sweeper.sampler.seed=11027 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 28 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11028.db' \
#     hydra.sweeper.n_trials=5 \
#     hydra.sweeper.sampler.seed=11028 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 29 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11029.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11029 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 30 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11030.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11030 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 31 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11031.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11031 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 32 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11032.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11032 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 33 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11033.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11033 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 34 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11034.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11034 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 35 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11035.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11035 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 36 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11036.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11036 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 37 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11037.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11037 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 38 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11038.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11038 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 39 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11039.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11039 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 40 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11040.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11040 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 41 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11041.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11041 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 42 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11042.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11042 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 43 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11043.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11043 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 44 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11044.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11044 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 45 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11045.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11045 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 46 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11046.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11046 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 47 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11047.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11047 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 48 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ae \
#     logger=mlflow \
#     hparams_search=physics/ae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ae_pool/11048.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=11048 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]
