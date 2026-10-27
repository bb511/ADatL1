#!/usr/bin/env bash
# ========================================================================
# DTE POOL COMMANDS
# ========================================================================
# - Random pool of physics/dte: 190 trials of 200 epochs, in the Optuna
#   study `pool` over configs/hparams_search/physics/dte_optuna.yaml.
# - 48 drivers, both waves running together. Wave 1 is drivers 1 to 24:
#   4 trials for drivers 1 to 18, 3 for 19 to 24 (90 in all).
#   Wave 2 is drivers 25 to 48, with fresh seeds:
#   5 trials for drivers 25 to 28, 4 for 29 to 48 (100 in all).
# - A driver holds at most 5 trials, so that it ends within 11 h at half the
#   V100 speed: 5 trials x 2 x 62.6 min (one trial on an idle V100 at the
#   largest configuration, 200 epochs of 16.0 s plus 2.7 min of start-up and
#   6.6 of evaluation) stay under 660 min.
# - One sqlite study file per driver, logs/optuna/physics/dte_pool/<seed>.db, so
#   no file has two writers; the URL carries no busy timeout, since Hydra's
#   override grammar does not take an unquoted '=' and Python's default 5 s
#   covers a reader.
# - Each trial checkpoints every validation strategy at both rate tiers, dumps
#   its weights and probe scores at every epoch, and is evaluated at those
#   checkpoints.
# - Timeout per trial: 90 min, measured on an idle V100 at the largest configuration
#   (half again 200 epochs of 16.0 s, plus 2.7 min of start-up and 6.6 of evaluation).
# - Per driver: 3 threads on a GPU host such as olgpu, 11 on a clariden node of
#   24 drivers (OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS,
#   NUMEXPR_NUM_THREADS), and about 31 GiB of RAM.
# - A driver that stops early is relaunched with a fresh seed, which
#   scripts/cluster/sweep_status.py prints.
# - The blocks name no launcher; scripts/cluster/submit_search.sh appends the
#   venue's to every driver. On clariden, one whole-node job per 24 drivers,
#   with the record cache built beforehand and HF_HUB_OFFLINE=1:
#     sbatch scripts/cluster/clariden_node.sbatch --only $(seq -s, 1 24) \
#         scripts/physics/rundte_search.sh
#   then --only $(seq -s, 25 48).
#   On a GPU host, one process per trial (here drivers 1 to 4 on four GPUs):
#     bash scripts/cluster/submit_search.sh --only $(seq -s, 1 4) \
#         scripts/physics/rundte_search.sh hydra/launcher=submitit_local \
#         hydra.launcher.gpus_per_node=1 hydra.launcher.timeout_min=90


# ========================================================================
# WAVE 1 DRIVERS
# ========================================================================
# ------------------------------------------------------------------------
# Driver 1 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16001.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=16001 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 2 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16002.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=16002 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 3 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16003.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=16003 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 4 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16004.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=16004 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 5 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16005.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=16005 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 6 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16006.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=16006 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 7 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16007.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=16007 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 8 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16008.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=16008 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 9 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16009.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=16009 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 10 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16010.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=16010 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 11 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16011.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=16011 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 12 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16012.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=16012 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 13 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16013.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=16013 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 14 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16014.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=16014 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 15 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16015.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=16015 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 16 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16016.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=16016 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 17 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16017.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=16017 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 18 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16018.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=16018 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 19 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16019.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=16019 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 20 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16020.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=16020 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 21 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16021.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=16021 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 22 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16022.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=16022 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 23 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16023.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=16023 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 24 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16024.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=16024 \
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
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16025.db' \
#     hydra.sweeper.n_trials=5 \
#     hydra.sweeper.sampler.seed=16025 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 26 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16026.db' \
#     hydra.sweeper.n_trials=5 \
#     hydra.sweeper.sampler.seed=16026 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 27 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16027.db' \
#     hydra.sweeper.n_trials=5 \
#     hydra.sweeper.sampler.seed=16027 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 28 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16028.db' \
#     hydra.sweeper.n_trials=5 \
#     hydra.sweeper.sampler.seed=16028 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 29 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16029.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=16029 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 30 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16030.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=16030 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 31 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16031.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=16031 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 32 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16032.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=16032 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 33 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16033.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=16033 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 34 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16034.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=16034 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 35 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16035.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=16035 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 36 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16036.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=16036 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 37 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16037.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=16037 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 38 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16038.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=16038 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 39 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16039.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=16039 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 40 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16040.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=16040 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 41 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16041.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=16041 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 42 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16042.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=16042 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 43 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16043.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=16043 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 44 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16044.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=16044 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 45 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16045.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=16045 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 46 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16046.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=16046 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 47 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16047.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=16047 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 48 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dte \
#     logger=mlflow \
#     hparams_search=physics/dte_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dte_pool/16048.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=16048 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]
