#!/usr/bin/env bash
# ========================================================================
# VAE POOL COMMANDS
# ========================================================================
# - Random pool of physics/vae: 196 trials of 200 epochs, in the Optuna
#   study `pool` over configs/hparams_search/physics/vae_optuna.yaml.
# - 48 drivers, both waves running together. Wave 1 is drivers 1 to 24:
#   4 trials each (96 in all).
#   Wave 2 is drivers 25 to 48, with fresh seeds:
#   5 trials for drivers 25 to 28, 4 for 29 to 48 (100 in all).
# - A driver holds at most 5 trials, so that it ends within 11 h at half the
#   V100 speed: 5 trials x 2 x 58.0 min (one trial on an idle V100 at the
#   largest configuration, 200 epochs of 14.6 s plus 2.2 min of start-up and
#   7.1 of evaluation) stay under 660 min.
# - One sqlite study file per driver, logs/optuna/physics/vae_pool/<seed>.db, so
#   no file has two writers; the URL carries no busy timeout, since Hydra's
#   override grammar does not take an unquoted '=' and Python's default 5 s
#   covers a reader.
# - Each trial checkpoints every validation strategy at both rate tiers, dumps
#   its weights and probe scores at every epoch, and is evaluated at those
#   checkpoints.
# - Timeout per trial: 83 min, measured on an idle V100 at the largest configuration
#   (half again 200 epochs of 14.6 s, plus 2.2 min of start-up and 7.1 of evaluation).
# - Per driver: 3 threads on a GPU host such as olgpu, 11 on a clariden node of
#   24 drivers (OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS,
#   NUMEXPR_NUM_THREADS), and about 31 GiB of RAM.
# - A driver that stops early is relaunched with a fresh seed, which
#   scripts/cluster/sweep_status.py prints.
# - The blocks name no launcher; scripts/cluster/submit_search.sh appends the
#   venue's to every driver. On clariden, one whole-node job per 24 drivers,
#   with the record cache built beforehand and HF_HUB_OFFLINE=1:
#     sbatch scripts/cluster/clariden_node.sbatch --only $(seq -s, 1 24) \
#         scripts/physics/runvae_search.sh
#   then --only $(seq -s, 25 48).
#   On a GPU host, one process per trial (here drivers 1 to 4 on four GPUs):
#     bash scripts/cluster/submit_search.sh --only $(seq -s, 1 4) \
#         scripts/physics/runvae_search.sh hydra/launcher=submitit_local \
#         hydra.launcher.gpus_per_node=1 hydra.launcher.timeout_min=83


# ========================================================================
# WAVE 1 DRIVERS
# ========================================================================
# ------------------------------------------------------------------------
# Driver 1 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12001.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12001 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 2 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12002.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12002 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 3 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12003.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12003 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 4 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12004.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12004 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 5 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12005.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12005 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 6 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12006.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12006 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 7 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12007.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12007 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 8 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12008.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12008 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 9 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12009.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12009 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 10 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12010.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12010 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 11 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12011.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12011 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 12 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12012.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12012 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 13 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12013.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12013 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 14 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12014.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12014 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 15 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12015.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12015 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 16 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12016.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12016 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 17 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12017.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12017 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 18 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12018.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12018 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 19 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12019.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12019 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 20 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12020.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12020 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 21 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12021.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12021 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 22 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12022.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12022 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 23 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12023.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12023 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 24 of 48 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12024.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12024 \
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
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12025.db' \
#     hydra.sweeper.n_trials=5 \
#     hydra.sweeper.sampler.seed=12025 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 26 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12026.db' \
#     hydra.sweeper.n_trials=5 \
#     hydra.sweeper.sampler.seed=12026 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 27 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12027.db' \
#     hydra.sweeper.n_trials=5 \
#     hydra.sweeper.sampler.seed=12027 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 28 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12028.db' \
#     hydra.sweeper.n_trials=5 \
#     hydra.sweeper.sampler.seed=12028 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 29 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12029.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12029 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 30 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12030.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12030 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 31 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12031.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12031 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 32 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12032.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12032 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 33 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12033.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12033 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 34 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12034.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12034 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 35 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12035.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12035 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 36 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12036.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12036 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 37 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12037.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12037 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 38 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12038.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12038 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 39 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12039.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12039 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 40 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12040.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12040 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 41 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12041.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12041 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 42 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12042.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12042 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 43 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12043.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12043 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 44 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12044.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12044 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 45 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12045.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12045 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 46 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12046.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12046 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 47 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12047.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12047 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 48 of 48 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/vae_pool/12048.db' \
#     hydra.sweeper.n_trials=4 \
#     hydra.sweeper.sampler.seed=12048 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]
