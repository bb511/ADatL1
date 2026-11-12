#!/usr/bin/env bash
# ========================================================================
# SVDD LATTICE COMMANDS (BLOOD COUNTS)
# ========================================================================
# - Random lattice of sbc/svdd: 200 trials of 200 epochs, in the Optuna
#   study `pool` over configs/hparams_search/sbc/svdd_optuna.yaml.
# - Four drivers of 50 trials each.
# - Each trial dumps its scores and weights at every epoch and selects nothing;
#   src/eval.py evaluates the dumps offline.
# - Timeout per trial: 35 min, ceil(1.5 x 200 x 6.9 s + 0.4 min),
#   the epoch time of the space's largest configuration, timed alone on one
#   V100S (11 Nov 2026).
# - A driver that stops early is relaunched with a fresh seed, which
#   scripts/cluster/sweep_status.py prints.
# - The blocks name no launcher; scripts/cluster/submit_search.sh appends the
#   venue's to every driver, e.g. one process per trial on a GPU host:
#     bash scripts/cluster/submit_search.sh scripts/sbc/runsvdd_search.sh \
#         hydra/launcher=submitit_local hydra.launcher.gpus_per_node=1 \
#         hydra.launcher.timeout_min=35


# ========================================================================
# LATTICE DRIVERS
# ========================================================================
# ------------------------------------------------------------------------
# Driver 1 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=sbc/svdd \
#     logger=mlflow \
#     hparams_search=sbc/svdd_optuna \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=7041 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 2 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=sbc/svdd \
#     logger=mlflow \
#     hparams_search=sbc/svdd_optuna \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=7042 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 3 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=sbc/svdd \
#     logger=mlflow \
#     hparams_search=sbc/svdd_optuna \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=7043 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 4 of 4
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=sbc/svdd \
#     logger=mlflow \
#     hparams_search=sbc/svdd_optuna \
#     hydra.sweeper.n_trials=50 \
#     hydra.sweeper.sampler.seed=7044 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]
