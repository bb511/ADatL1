#!/usr/bin/env bash
# ========================================================================
# DeepSets AE POOL COMMANDS
# ========================================================================
# - Random pool of physics/dsae: 190 trials of 200 epochs, in the Optuna
#   study `pool` over configs/hparams_search/physics/dsae_optuna.yaml.
# - 72 drivers, both waves running together. Wave 1 is drivers 1 to 36:
#   3 trials for drivers 1 to 18, 2 for 19 to 36 (90 in all).
#   Wave 2 is drivers 37 to 72, with fresh seeds:
#   3 trials for drivers 37 to 64, 2 for 65 to 72 (100 in all).
# - A driver holds at most 3 trials, so that it ends within 11 h at half the
#   V100 speed: 3 trials x 2 x 102.0 min (one trial on an idle V100 at the
#   largest configuration, 200 epochs of 27.6 s plus 2.3 min of start-up and
#   7.7 of evaluation) stay under 660 min.
# - One sqlite study file per driver, logs/optuna/physics/dsae_pool/<seed>.db, so
#   no file has two writers; the URL carries no busy timeout, since Hydra's
#   override grammar does not take an unquoted '=' and Python's default 5 s
#   covers a reader.
# - Each trial checkpoints every validation strategy at both rate tiers, dumps
#   its weights and probe scores at every epoch, and is evaluated at those
#   checkpoints.
# - Timeout per trial: 148 min, measured on an idle V100 at the largest configuration
#   (half again 200 epochs of 27.6 s, plus 2.3 min of start-up and 7.7 of evaluation).
# - Per driver: 3 threads on a GPU host such as olgpu, 11 on a clariden node of
#   24 drivers (OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS,
#   NUMEXPR_NUM_THREADS), and about 31 GiB of RAM.
# - A driver that stops early is relaunched with a fresh seed, which
#   scripts/cluster/sweep_status.py prints.
# - The blocks name no launcher; scripts/cluster/submit_search.sh appends the
#   venue's to every driver. On clariden, one whole-node job per 24 drivers,
#   with the record cache built beforehand and HF_HUB_OFFLINE=1:
#     sbatch scripts/cluster/clariden_node.sbatch --only $(seq -s, 1 24) \
#         scripts/physics/rundsae_search.sh
#   then --only $(seq -s, 25 48), and --only $(seq -s, 49 72).
#   On a GPU host, one process per trial (here drivers 1 to 4 on four GPUs):
#     bash scripts/cluster/submit_search.sh --only $(seq -s, 1 4) \
#         scripts/physics/rundsae_search.sh hydra/launcher=submitit_local \
#         hydra.launcher.gpus_per_node=1 hydra.launcher.timeout_min=148


# ========================================================================
# WAVE 1 DRIVERS
# ========================================================================
# ------------------------------------------------------------------------
# Driver 1 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13001.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13001 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 2 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13002.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13002 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 3 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13003.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13003 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 4 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13004.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13004 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 5 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13005.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13005 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 6 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13006.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13006 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 7 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13007.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13007 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 8 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13008.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13008 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 9 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13009.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13009 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 10 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13010.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13010 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 11 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13011.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13011 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 12 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13012.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13012 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 13 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13013.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13013 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 14 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13014.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13014 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 15 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13015.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13015 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 16 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13016.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13016 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 17 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13017.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13017 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 18 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13018.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13018 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 19 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13019.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=13019 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 20 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13020.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=13020 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 21 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13021.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=13021 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 22 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13022.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=13022 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 23 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13023.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=13023 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 24 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13024.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=13024 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 25 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13025.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=13025 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 26 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13026.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=13026 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 27 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13027.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=13027 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 28 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13028.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=13028 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 29 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13029.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=13029 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 30 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13030.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=13030 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 31 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13031.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=13031 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 32 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13032.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=13032 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 33 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13033.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=13033 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 34 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13034.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=13034 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 35 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13035.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=13035 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 36 of 72 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13036.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=13036 \
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
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13037.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13037 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 38 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13038.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13038 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 39 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13039.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13039 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 40 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13040.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13040 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 41 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13041.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13041 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 42 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13042.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13042 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 43 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13043.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13043 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 44 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13044.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13044 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 45 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13045.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13045 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 46 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13046.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13046 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 47 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13047.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13047 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 48 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13048.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13048 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 49 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13049.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13049 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 50 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13050.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13050 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 51 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13051.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13051 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 52 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13052.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13052 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 53 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13053.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13053 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 54 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13054.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13054 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 55 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13055.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13055 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 56 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13056.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13056 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 57 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13057.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13057 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 58 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13058.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13058 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 59 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13059.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13059 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 60 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13060.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13060 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 61 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13061.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13061 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 62 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13062.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13062 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 63 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13063.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13063 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 64 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13064.db' \
#     hydra.sweeper.n_trials=3 \
#     hydra.sweeper.sampler.seed=13064 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 65 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13065.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=13065 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 66 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13066.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=13066 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 67 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13067.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=13067 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 68 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13068.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=13068 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 69 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13069.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=13069 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 70 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13070.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=13070 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 71 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13071.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=13071 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 72 of 72 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/dsae \
#     logger=mlflow \
#     hparams_search=physics/dsae_optuna \
#     hydra.sweeper.storage='sqlite:///logs/optuna/physics/dsae_pool/13072.db' \
#     hydra.sweeper.n_trials=2 \
#     hydra.sweeper.sampler.seed=13072 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]
