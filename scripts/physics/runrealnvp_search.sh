#!/usr/bin/env bash
# ========================================================================
# RealNVP POOL COMMANDS
# ========================================================================
# - Random pool of physics/realnvp: 190 trials of 200 epochs, in the Optuna
#   study `pool` over configs/hparams_search/physics/realnvp_optuna.yaml.
# - 72 drivers, both waves running together. Wave 1 is drivers 1 to 36:
#   3 trials for drivers 1 to 18, 2 for 19 to 36 (90 in all).
#   Wave 2 is drivers 37 to 72, with fresh seeds:
#   3 trials for drivers 37 to 64, 2 for 65 to 72 (100 in all).
# - A driver holds at most 3 trials, so that it ends within 11 h at half the
#   V100 speed: 3 trials x 2 x 97.2 min (one trial on an idle V100 at the
#   largest configuration, 200 epochs of 26.3 s plus 2.2 min of start-up and
#   7.3 of evaluation) stay under 660 min.
# - One sqlite study file per driver, logs/optuna/physics/realnvp_pool/<seed>.db, so
#   no file has two writers; the URL carries no busy timeout, since Hydra's
#   override grammar does not take an unquoted '=' and Python's default 5 s
#   covers a reader.
# - Each trial checkpoints every validation strategy at both rate tiers, dumps
#   its weights and probe scores at every epoch, and is evaluated at those
#   checkpoints.
# - Timeout per trial: 142 min, measured on an idle V100 at the largest configuration
#   (half again 200 epochs of 26.3 s, plus 2.2 min of start-up and 7.3 of evaluation).
# - Per driver: 3 threads on a GPU host such as olgpu, 11 on a clariden node of
#   24 drivers (OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS,
#   NUMEXPR_NUM_THREADS), and about 31 GiB of RAM.
# - A driver that stops early is relaunched with a fresh seed, which
#   scripts/cluster/sweep_status.py prints.
# - The blocks name no launcher; scripts/cluster/submit_search.sh appends the
#   venue's to every driver. On clariden, one whole-node job per 24 drivers,
#   with the record cache built beforehand and HF_HUB_OFFLINE=1:
#     sbatch scripts/cluster/clariden_node.sbatch --only $(seq -s, 1 24) \
#         scripts/physics/runrealnvp_search.sh
#   then --only $(seq -s, 25 48), and --only $(seq -s, 49 72).
#   On a GPU host, one process per trial (here drivers 1 to 4 on four GPUs):
#     bash scripts/cluster/submit_search.sh --only $(seq -s, 1 4) \
#         scripts/physics/runrealnvp_search.sh hydra/launcher=submitit_local \
#         hydra.launcher.gpus_per_node=1 hydra.launcher.timeout_min=142


# ========================================================================
# WAVE 1 DRIVERS
# ========================================================================
# ------------------------------------------------------------------------
# Driver 1 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15001.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15001 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 2 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15002.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15002 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 3 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15003.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15003 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 4 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15004.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15004 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 5 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15005.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15005 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 6 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15006.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15006 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 7 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15007.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15007 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 8 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15008.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15008 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 9 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15009.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15009 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 10 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15010.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15010 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 11 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15011.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15011 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 12 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15012.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15012 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 13 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15013.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15013 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 14 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15014.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15014 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 15 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15015.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15015 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 16 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15016.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15016 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 17 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15017.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15017 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 18 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15018.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15018 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 19 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15019.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=15019 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 20 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15020.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=15020 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 21 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15021.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=15021 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 22 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15022.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=15022 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 23 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15023.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=15023 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 24 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15024.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=15024 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 25 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15025.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=15025 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 26 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15026.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=15026 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 27 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15027.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=15027 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 28 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15028.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=15028 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 29 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15029.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=15029 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 30 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15030.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=15030 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 31 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15031.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=15031 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 32 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15032.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=15032 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 33 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15033.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=15033 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 34 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15034.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=15034 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 35 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15035.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=15035 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 36 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15036.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=15036 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ========================================================================
# WAVE 2 DRIVERS
# ========================================================================
# ------------------------------------------------------------------------
# Driver 37 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15037.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15037 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 38 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15038.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15038 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 39 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15039.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15039 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 40 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15040.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15040 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 41 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15041.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15041 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 42 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15042.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15042 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 43 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15043.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15043 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 44 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15044.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15044 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 45 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15045.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15045 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 46 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15046.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15046 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 47 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15047.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15047 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 48 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15048.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15048 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 49 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15049.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15049 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 50 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15050.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15050 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 51 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15051.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15051 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 52 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15052.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15052 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 53 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15053.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15053 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 54 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15054.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15054 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 55 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15055.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15055 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 56 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15056.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15056 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 57 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15057.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15057 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 58 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15058.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15058 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 59 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15059.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15059 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 60 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15060.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15060 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 61 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15061.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15061 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 62 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15062.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15062 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 63 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15063.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15063 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 64 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15064.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=15064 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 65 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15065.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=15065 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 66 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15066.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=15066 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 67 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15067.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=15067 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 68 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15068.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=15068 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 69 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15069.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=15069 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 70 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15070.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=15070 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 71 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15071.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=15071 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 72 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/realnvp \
#     logger=mlflow \
#     hparams_search=physics/realnvp_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/realnvp_pool/15072.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=15072 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]
