#!/usr/bin/env bash
# ========================================================================
# DDL POOL COMMANDS
# ========================================================================
# - Random pool of physics/ddl: 200 trials of 200 epochs, in the Optuna
#   study `pool` over configs/hparams_search/physics/ddl_optuna.yaml; its 10
#   dimensions set wave 1 at 100 trials (10 per dimension).
# - 200 drivers of one trial each, both waves running together. Wave 1 is
#   drivers 1 to 100 and wave 2 drivers 101 to 200, with fresh seeds.
# - A driver holds one trial, since two would not end within 11 h at half the
#   V100 speed: one trial at the largest configuration (width 256, depth 3) with
#   20 score timesteps took 3.4 min to start, 47 to 66 s per epoch (56.5 s on
#   average) and 22 min of evaluation on a V100S that other jobs shared (1 Nov).
#   Scoring the validation pass takes 26.2 s of that epoch at 20 timesteps and
#   13.6 s at the fixed 10, so an epoch takes about 43.9 s, a trial about 172 min,
#   and 2 x 2 x 172 min exceed 660.
# - One sqlite study file per driver, logs/optuna/physics/ddl_pool/<seed>.db, so
#   no file has two writers; the URL carries no busy timeout, since Hydra's
#   override grammar does not take an unquoted '=' and Python's default 5 s
#   covers a reader.
# - Each trial checkpoints every validation strategy at both rate tiers, dumps
#   its weights and probe scores at every epoch, and is evaluated at those
#   checkpoints.
# - Timeout per trial: 245 min, from the times above (half again 200 epochs of
#   43.9 s, plus 3.4 min of start-up and 22 of evaluation).
# - Per driver: 3 threads on a GPU host such as olgpu, 11 on a clariden node of
#   24 drivers (OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS,
#   NUMEXPR_NUM_THREADS), and about 31 GiB of RAM.
# - A driver that stops early is relaunched with a fresh seed, which
#   scripts/cluster/sweep_status.py prints.
# - The blocks name no launcher; scripts/cluster/submit_search.sh appends the
#   venue's to every driver. On clariden, one whole-node job per 24 drivers,
#   with the record cache built beforehand and HF_HUB_OFFLINE=1:
#     sbatch scripts/cluster/clariden_node.sbatch --only $(seq -s, 1 24) \
#         scripts/physics/runddl_search.sh
#   and likewise for drivers 25 to 200, 24 at a time.
#   On a GPU host, one process per trial (here drivers 1 to 4 on four GPUs):
#     bash scripts/cluster/submit_search.sh --only $(seq -s, 1 4) \
#         scripts/physics/runddl_search.sh hydra/launcher=submitit_local \
#         hydra.launcher.gpus_per_node=1 hydra.launcher.timeout_min=245


# ========================================================================
# WAVE 1 DRIVERS
# ========================================================================
# ------------------------------------------------------------------------
# Driver 1 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18001.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18001 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 2 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18002.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18002 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 3 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18003.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18003 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 4 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18004.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18004 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 5 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18005.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18005 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 6 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18006.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18006 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 7 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18007.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18007 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 8 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18008.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18008 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 9 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18009.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18009 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 10 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18010.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18010 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 11 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18011.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18011 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 12 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18012.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18012 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 13 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18013.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18013 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 14 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18014.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18014 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 15 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18015.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18015 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 16 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18016.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18016 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 17 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18017.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18017 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 18 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18018.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18018 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 19 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18019.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18019 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 20 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18020.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18020 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 21 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18021.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18021 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 22 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18022.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18022 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 23 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18023.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18023 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 24 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18024.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18024 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 25 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18025.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18025 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 26 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18026.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18026 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 27 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18027.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18027 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 28 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18028.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18028 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 29 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18029.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18029 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 30 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18030.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18030 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 31 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18031.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18031 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 32 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18032.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18032 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 33 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18033.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18033 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 34 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18034.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18034 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 35 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18035.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18035 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 36 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18036.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18036 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 37 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18037.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18037 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 38 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18038.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18038 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 39 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18039.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18039 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 40 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18040.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18040 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 41 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18041.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18041 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 42 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18042.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18042 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 43 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18043.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18043 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 44 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18044.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18044 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 45 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18045.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18045 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 46 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18046.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18046 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 47 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18047.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18047 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 48 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18048.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18048 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 49 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18049.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18049 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 50 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18050.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18050 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 51 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18051.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18051 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 52 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18052.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18052 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 53 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18053.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18053 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 54 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18054.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18054 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 55 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18055.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18055 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 56 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18056.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18056 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 57 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18057.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18057 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 58 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18058.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18058 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 59 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18059.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18059 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 60 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18060.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18060 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 61 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18061.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18061 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 62 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18062.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18062 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 63 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18063.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18063 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 64 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18064.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18064 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 65 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18065.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18065 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 66 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18066.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18066 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 67 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18067.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18067 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 68 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18068.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18068 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 69 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18069.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18069 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 70 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18070.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18070 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 71 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18071.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18071 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 72 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18072.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18072 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 73 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18073.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18073 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 74 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18074.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18074 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 75 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18075.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18075 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 76 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18076.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18076 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 77 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18077.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18077 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 78 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18078.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18078 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 79 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18079.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18079 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 80 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18080.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18080 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 81 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18081.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18081 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 82 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18082.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18082 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 83 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18083.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18083 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 84 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18084.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18084 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 85 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18085.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18085 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 86 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18086.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18086 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 87 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18087.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18087 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 88 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18088.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18088 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 89 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18089.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18089 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 90 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18090.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18090 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 91 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18091.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18091 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 92 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18092.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18092 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 93 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18093.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18093 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 94 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18094.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18094 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 95 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18095.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18095 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 96 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18096.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18096 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 97 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18097.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18097 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 98 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18098.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18098 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 99 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18099.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18099 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 100 of 200 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18100.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18100 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ========================================================================
# WAVE 2 DRIVERS
# ========================================================================
# ------------------------------------------------------------------------
# Driver 101 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18101.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18101 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 102 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18102.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18102 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 103 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18103.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18103 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 104 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18104.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18104 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 105 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18105.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18105 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 106 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18106.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18106 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 107 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18107.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18107 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 108 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18108.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18108 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 109 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18109.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18109 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 110 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18110.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18110 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 111 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18111.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18111 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 112 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18112.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18112 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 113 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18113.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18113 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 114 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18114.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18114 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 115 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18115.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18115 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 116 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18116.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18116 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 117 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18117.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18117 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 118 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18118.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18118 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 119 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18119.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18119 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 120 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18120.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18120 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 121 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18121.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18121 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 122 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18122.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18122 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 123 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18123.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18123 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 124 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18124.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18124 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 125 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18125.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18125 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 126 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18126.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18126 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 127 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18127.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18127 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 128 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18128.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18128 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 129 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18129.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18129 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 130 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18130.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18130 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 131 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18131.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18131 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 132 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18132.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18132 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 133 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18133.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18133 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 134 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18134.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18134 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 135 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18135.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18135 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 136 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18136.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18136 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 137 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18137.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18137 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 138 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18138.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18138 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 139 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18139.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18139 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 140 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18140.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18140 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 141 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18141.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18141 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 142 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18142.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18142 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 143 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18143.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18143 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 144 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18144.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18144 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 145 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18145.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18145 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 146 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18146.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18146 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 147 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18147.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18147 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 148 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18148.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18148 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 149 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18149.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18149 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 150 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18150.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18150 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 151 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18151.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18151 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 152 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18152.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18152 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 153 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18153.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18153 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 154 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18154.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18154 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 155 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18155.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18155 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 156 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18156.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18156 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 157 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18157.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18157 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 158 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18158.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18158 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 159 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18159.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18159 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 160 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18160.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18160 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 161 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18161.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18161 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 162 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18162.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18162 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 163 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18163.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18163 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 164 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18164.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18164 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 165 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18165.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18165 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 166 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18166.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18166 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 167 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18167.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18167 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 168 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18168.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18168 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 169 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18169.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18169 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 170 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18170.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18170 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 171 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18171.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18171 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 172 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18172.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18172 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 173 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18173.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18173 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 174 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18174.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18174 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 175 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18175.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18175 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 176 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18176.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18176 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 177 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18177.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18177 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 178 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18178.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18178 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 179 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18179.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18179 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 180 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18180.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18180 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 181 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18181.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18181 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 182 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18182.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18182 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 183 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18183.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18183 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 184 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18184.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18184 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 185 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18185.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18185 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 186 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18186.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18186 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 187 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18187.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18187 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 188 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18188.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18188 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 189 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18189.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18189 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 190 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18190.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18190 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 191 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18191.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18191 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 192 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18192.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18192 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 193 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18193.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18193 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 194 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18194.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18194 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 195 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18195.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18195 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 196 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18196.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18196 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 197 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18197.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18197 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 198 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18198.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18198 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 199 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18199.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18199 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 200 of 200 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/ddl \
#     logger=mlflow \
#     hparams_search=physics/ddl_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/ddl_pool/18200.db' \
#     hydra.sweeper.n_trials=1 \
#     hydra.sweeper.sampler.seed=18200 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]
