#!/usr/bin/env bash
# ========================================================================
# VAE POOL COMMANDS
# ========================================================================
# The random pool of physics/vae: 196 trials in two waves, drawn by eight
# drivers that share the Optuna study `pool` and differ only in their
# sampler seed. Wave 1 (drivers 1 to 4, 96 trials) is the pool as sized by the
# rule of subplan 8; wave 2 (drivers 5 to 8, 100 trials, fresh seeds) is
# launched after wave 1 completes, and the selector tables are read only
# once the wave that closes the pool is harvested. Every trial is a full
# 200-epoch training, evaluated at the checkpoint of each validation
# strategy, at both background-rate tiers, and the strategies then select
# from the pool.
#
# The random sampler replays its draws from the seed, so a driver that stops
# early is relaunched with a fresh seed, not its own, which would repeat the
# configurations already trained. scripts/cluster/sweep_status.py prints
# such a relaunch. Between the waves it reports this study short by 100
# trials and offers a top-up of driver 1; the wave-2 blocks are launched
# instead, with --only 5,6,7,8.
#
# The blocks name no launcher: the venue appends one at submission, with a
# timeout of 70 minutes per trial for this model (200 epochs at 13 s each on an H100, plus about 3 minutes of
# evaluation, and half again as margin).
#
# On clariden, each driver keeps six trials queued as slurm jobs:
#   bash scripts/cluster/submit_search.sh --only 1,2,3,4 scripts/physics/runvae_search.sh \
#       hydra/launcher=submitit_slurm_clariden hydra.sweeper.n_jobs=6 \
#       hydra.launcher.timeout_min=70
#
# On a local GPU host or an NGT pod, Hydra's basic launcher runs a driver's
# trials in-process, one after another:
#   bash scripts/cluster/submit_search.sh --only 1,2,3,4 scripts/physics/runvae_search.sh
# or, with each trial in its own process and under the timeout:
#   bash scripts/cluster/submit_search.sh --only 1,2,3,4 scripts/physics/runvae_search.sh \
#       hydra/launcher=submitit_local hydra.launcher.gpus_per_node=1 \
#       hydra.launcher.timeout_min=70
# submitit_local shows a trial GPUs 0 to gpus_per_node-1 only, none when it
# is unset, and trainer.devices picks among them.


# ========================================================================
# WAVE 1 DRIVERS
# ========================================================================
# ------------------------------------------------------------------------
# Driver 1 of 8 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.n_trials=24 \
#     hydra.sweeper.sampler.seed=1021 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 2 of 8 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.n_trials=24 \
#     hydra.sweeper.sampler.seed=1022 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 3 of 8 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.n_trials=24 \
#     hydra.sweeper.sampler.seed=1023 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 4 of 8 (wave 1)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.n_trials=24 \
#     hydra.sweeper.sampler.seed=1024 \
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
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.n_trials=25 \
#     hydra.sweeper.sampler.seed=1025 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 6 of 8 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.n_trials=25 \
#     hydra.sweeper.sampler.seed=1026 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 7 of 8 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.n_trials=25 \
#     hydra.sweeper.sampler.seed=1027 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# Driver 8 of 8 (wave 2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     data=basis_hf \
#     experiment=physics/vae \
#     logger=mlflow \
#     hparams_search=physics/vae_optuna \
#     hydra.sweeper.n_trials=25 \
#     hydra.sweeper.sampler.seed=1028 \
#     trainer=gpu \
#     trainer.max_epochs=200 \
#     trainer.devices=[0]
