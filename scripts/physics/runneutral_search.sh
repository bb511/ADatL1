#!/usr/bin/env bash
# ========================================================================
# NEUTRAL AD POOL COMMANDS
# ========================================================================
# - Random pool of physics/neutral: 196 trials of 200 epochs, in the Optuna
#   study `pool` over configs/hparams_search/physics/neutral_optuna.yaml.
# - 72 drivers, both waves running together. Wave 1 is drivers 1 to 36:
#   3 trials for drivers 1 to 24, 2 for 25 to 36 (96 in all).
#   Wave 2 is drivers 37 to 72, with fresh seeds:
#   3 trials for drivers 37 to 64, 2 for 65 to 72 (100 in all).
# - A driver holds at most 3 trials, so that it ends within 11 h at half the
#   V100 speed: 3 trials x 2 x 87.1 min (one trial on an idle V100 at the
#   largest configuration, 200 epochs of 23.1 s plus 2.7 min of start-up and
#   7.4 of evaluation) stay under 660 min.
# - One sqlite study file per driver, logs/optuna/physics/neutral_pool/<seed>.db, so
#   no file has two writers; the URL carries no busy timeout, since Hydra's
#   override grammar does not take an unquoted '=' and Python's default 5 s
#   covers a reader.
# - Each trial checkpoints every validation strategy at both rate tiers, dumps
#   its weights and probe scores at every epoch, and is evaluated at those
#   checkpoints.
# - Timeout per trial: 126 min, measured on an idle V100 at the largest configuration
#   (half again 200 epochs of 23.1 s, plus 2.7 min of start-up and 7.4 of evaluation).
# - Per driver: 3 threads on a GPU host such as olgpu, 11 on a clariden node of
#   24 drivers (OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS,
#   NUMEXPR_NUM_THREADS), and about 31 GiB of RAM.
# - A driver that stops early is relaunched with a fresh seed, which
#   scripts/cluster/sweep_status.py prints.
# - NeuTraL AD waits for the label-free near-empty-event check before launch;
#   the user decides after it.
# - The blocks name no launcher; scripts/cluster/submit_search.sh appends the
#   venue's to every driver. On clariden, one whole-node job per 24 drivers,
#   with the record cache built beforehand and HF_HUB_OFFLINE=1:
#     sbatch scripts/cluster/clariden_node.sbatch --only $(seq -s, 1 24) \
#         scripts/physics/runneutral_search.sh
#   then --only $(seq -s, 25 48), and --only $(seq -s, 49 72).
#   On a GPU host, one process per trial (here drivers 1 to 4 on four GPUs):
#     bash scripts/cluster/submit_search.sh --only $(seq -s, 1 4) \
#         scripts/physics/runneutral_search.sh hydra/launcher=submitit_local \
#         hydra.launcher.gpus_per_node=1 hydra.launcher.timeout_min=126


# ========================================================================
# WAVE 1 DRIVERS
# ========================================================================
# ------------------------------------------------------------------------
# Driver 1 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17001.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17001 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 2 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17002.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17002 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 3 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17003.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17003 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 4 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17004.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17004 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 5 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17005.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17005 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 6 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17006.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17006 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 7 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17007.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17007 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 8 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17008.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17008 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 9 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17009.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17009 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 10 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17010.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17010 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 11 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17011.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17011 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 12 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17012.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17012 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 13 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17013.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17013 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 14 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17014.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17014 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 15 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17015.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17015 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 16 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17016.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17016 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 17 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17017.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17017 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 18 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17018.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17018 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 19 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17019.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17019 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 20 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17020.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17020 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 21 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17021.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17021 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 22 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17022.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17022 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 23 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17023.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17023 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 24 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17024.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17024 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 25 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17025.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=17025 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 26 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17026.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=17026 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 27 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17027.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=17027 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 28 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17028.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=17028 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 29 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17029.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=17029 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 30 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17030.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=17030 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 31 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17031.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=17031 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 32 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17032.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=17032 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 33 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17033.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=17033 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 34 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17034.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=17034 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 35 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17035.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=17035 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 36 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17036.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=17036 \
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
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17037.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17037 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 38 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17038.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17038 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 39 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17039.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17039 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 40 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17040.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17040 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 41 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17041.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17041 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 42 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17042.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17042 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 43 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17043.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17043 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 44 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17044.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17044 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 45 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17045.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17045 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 46 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17046.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17046 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 47 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17047.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17047 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 48 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17048.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17048 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 49 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17049.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17049 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 50 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17050.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17050 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 51 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17051.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17051 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 52 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17052.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17052 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 53 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17053.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17053 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 54 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17054.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17054 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 55 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17055.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17055 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 56 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17056.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17056 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 57 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17057.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17057 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 58 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17058.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17058 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 59 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17059.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17059 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 60 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17060.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17060 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 61 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17061.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17061 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 62 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17062.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17062 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 63 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17063.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17063 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 64 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17064.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=17064 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 65 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17065.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=17065 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 66 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17066.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=17066 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 67 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17067.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=17067 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 68 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17068.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=17068 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 69 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17069.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=17069 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 70 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17070.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=17070 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 71 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17071.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=17071 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 72 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/neutral \
#     logger=mlflow \
#     hparams_search=physics/neutral_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/neutral_pool/17072.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=17072 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]
