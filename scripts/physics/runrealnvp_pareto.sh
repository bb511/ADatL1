#!/usr/bin/env bash
# ========================================================================
# REALNVP PARETO-FRONT TRAINING COMMANDS
# ========================================================================
# These are the training commands for every point on the Pareto front of
# each validation strategy. Generated from notebooks/paretos/physics/ by
# scripts/optuna/make_pareto_scripts.py -- regenerate rather than edit by hand.
#
# Run from the repository root. All commands are commented out -- uncomment
# the points you want to run locally (taskset pinning, GPUs cycling 0-3).
# To run the WHOLE file on clariden instead, use the single submit command
# at the bottom: it sends every point above to slurm, one job each, via
# scripts/cluster/submit_pareto.sh (submitit launcher).

# ========================================================================
# CVAR25 TRAINING  (study: cvar25eff_vs_logp_b16k, 9 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 69: cvar25eff=0.73154, logp=35.444
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp \
#     experiment_name=physics_realnvp_pareto \
#     run_name=cvar25_t69 \
#     algorithm.flow.activation=gelu \
#     algorithm.flow.hidden_dim=64 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0001193984523130763 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 304: cvar25eff=1.2356, logp=40.551
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp \
#     experiment_name=physics_realnvp_pareto \
#     run_name=cvar25_t304 \
#     algorithm.flow.activation=gelu \
#     algorithm.flow.hidden_dim=48 \
#     algorithm.flow.n_flows=6 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=5.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.00035197125905446204 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 306: cvar25eff=1.1231, logp=39.591
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp \
#     experiment_name=physics_realnvp_pareto \
#     run_name=cvar25_t306 \
#     algorithm.flow.activation=gelu \
#     algorithm.flow.hidden_dim=48 \
#     algorithm.flow.n_flows=6 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=5.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.00034610478450565853 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 321: cvar25eff=0.7354, logp=39.285
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp \
#     experiment_name=physics_realnvp_pareto \
#     run_name=cvar25_t321 \
#     algorithm.flow.activation=relu \
#     algorithm.flow.hidden_dim=64 \
#     algorithm.flow.n_flows=6 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=5.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.00011650496359385738 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 431: cvar25eff=0.58823, logp=29.441
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp \
#     experiment_name=physics_realnvp_pareto \
#     run_name=cvar25_t431 \
#     algorithm.flow.activation=relu \
#     algorithm.flow.hidden_dim=64 \
#     algorithm.flow.n_flows=6 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=5.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0006060428898214723 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 477: cvar25eff=1.7415, logp=49.747  << KNEE
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp \
#     experiment_name=physics_realnvp_pareto \
#     run_name=cvar25_t477 \
#     algorithm.flow.activation=gelu \
#     algorithm.flow.hidden_dim=48 \
#     algorithm.flow.n_flows=6 \
#     algorithm.flow.n_hidden_layers=1 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0013780717614807188 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 523: cvar25eff=3.3399, logp=159.11  << ORIGINAL PICK | BEST cvar25eff
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp \
#     experiment_name=physics_realnvp_pareto \
#     run_name=cvar25_t523 \
#     algorithm.flow.activation=gelu \
#     algorithm.flow.hidden_dim=48 \
#     algorithm.flow.n_flows=6 \
#     algorithm.flow.n_hidden_layers=1 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0012971125344674336 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 563: cvar25eff=2.0521, logp=94.253
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp \
#     experiment_name=physics_realnvp_pareto \
#     run_name=cvar25_t563 \
#     algorithm.flow.activation=gelu \
#     algorithm.flow.hidden_dim=48 \
#     algorithm.flow.n_flows=6 \
#     algorithm.flow.n_hidden_layers=1 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0012827112149607438 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 568: cvar25eff=1.8349, logp=56.546
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp \
#     experiment_name=physics_realnvp_pareto \
#     run_name=cvar25_t568 \
#     algorithm.flow.activation=gelu \
#     algorithm.flow.hidden_dim=48 \
#     algorithm.flow.n_flows=6 \
#     algorithm.flow.n_hidden_layers=1 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0013315387752524156 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ========================================================================
# CVAR10 TRAINING  (study: cvar10eff_vs_logp, 4 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 389: cvar10eff=0.87481, logp=2492.3  << BEST cvar10eff
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp \
#     experiment_name=physics_realnvp_pareto \
#     run_name=cvar10_t389 \
#     evaluation.callbacks.anomaly_efficiency.cvar_summary=0.10 \
#     algorithm.flow.activation=gelu \
#     algorithm.flow.hidden_dim=64 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=1 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0005946732019597374 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 401: cvar10eff=0.58433, logp=14.839
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp \
#     experiment_name=physics_realnvp_pareto \
#     run_name=cvar10_t401 \
#     evaluation.callbacks.anomaly_efficiency.cvar_summary=0.10 \
#     algorithm.flow.activation=gelu \
#     algorithm.flow.hidden_dim=64 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=1 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0005774599898578561 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 413: cvar10eff=0.69711, logp=18.271
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp \
#     experiment_name=physics_realnvp_pareto \
#     run_name=cvar10_t413 \
#     evaluation.callbacks.anomaly_efficiency.cvar_summary=0.10 \
#     algorithm.flow.activation=gelu \
#     algorithm.flow.hidden_dim=64 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=1 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.000663455835069413 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 491: cvar10eff=0.83668, logp=38.809  << KNEE
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp \
#     experiment_name=physics_realnvp_pareto \
#     run_name=cvar10_t491 \
#     evaluation.callbacks.anomaly_efficiency.cvar_summary=0.10 \
#     algorithm.flow.activation=gelu \
#     algorithm.flow.hidden_dim=64 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=1 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.000595242173552503 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ========================================================================
# CAP TRAINING  (study: cap_vs_logp_b16k, 11 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 161: cap=-0.13451, logp=31.165
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp_agnostic \
#     experiment_name=physics_realnvp_pareto \
#     run_name=cap_t161 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.flow.activation=relu \
#     algorithm.flow.hidden_dim=48 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0010270018408482442 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 164: cap=-0.1358, logp=19.599
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp_agnostic \
#     experiment_name=physics_realnvp_pareto \
#     run_name=cap_t164 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.flow.activation=relu \
#     algorithm.flow.hidden_dim=48 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.001030377645274494 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 187: cap=-0.12794, logp=71.67
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp_agnostic \
#     experiment_name=physics_realnvp_pareto \
#     run_name=cap_t187 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.flow.activation=relu \
#     algorithm.flow.hidden_dim=48 \
#     algorithm.flow.n_flows=4 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0009708225900578386 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 285: cap=-0.13498, logp=28.757
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp_agnostic \
#     experiment_name=physics_realnvp_pareto \
#     run_name=cap_t285 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.flow.activation=relu \
#     algorithm.flow.hidden_dim=48 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0010490934267674517 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 308: cap=-0.1276, logp=72.755  << KNEE
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp_agnostic \
#     experiment_name=physics_realnvp_pareto \
#     run_name=cap_t308 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.flow.activation=relu \
#     algorithm.flow.hidden_dim=32 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0010858278705171665 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 309: cap=-0.13161, logp=62.813
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp_agnostic \
#     experiment_name=physics_realnvp_pareto \
#     run_name=cap_t309 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.flow.activation=relu \
#     algorithm.flow.hidden_dim=32 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.001100517864629836 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 337: cap=-0.12919, logp=70.205
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp_agnostic \
#     experiment_name=physics_realnvp_pareto \
#     run_name=cap_t337 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.flow.activation=relu \
#     algorithm.flow.hidden_dim=32 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0010691956021680951 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 338: cap=-0.13198, logp=54.394
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp_agnostic \
#     experiment_name=physics_realnvp_pareto \
#     run_name=cap_t338 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.flow.activation=relu \
#     algorithm.flow.hidden_dim=32 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0010697730233659101 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 376: cap=-0.14036, logp=15.283  << ORIGINAL PICK
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp_agnostic \
#     experiment_name=physics_realnvp_pareto \
#     run_name=cap_t376 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.flow.activation=relu \
#     algorithm.flow.hidden_dim=48 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0010217753274727394 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 390: cap=-0.13355, logp=43.997
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp_agnostic \
#     experiment_name=physics_realnvp_pareto \
#     run_name=cap_t390 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.flow.activation=relu \
#     algorithm.flow.hidden_dim=48 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0009363977078958039 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 441: cap=-0.10768, logp=374.52  << BEST cap
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp_agnostic \
#     experiment_name=physics_realnvp_pareto \
#     run_name=cap_t441 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.flow.activation=relu \
#     algorithm.flow.hidden_dim=64 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=1 \
#     algorithm.flow.noise_scale=0.0 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.001330207301401317 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ========================================================================
# CONSISTENCY TRAINING  (study: consistency_vs_logp_b16k, 17 Pareto points, trimmed to 11 around the knee (endpoints kept))
# ========================================================================
# ------------------------------------------------------------------------
# trial 29: consistency=-1.6659e-07, logp=1723.6
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp_agnostic \
#     experiment_name=physics_realnvp_pareto \
#     run_name=consistency_t29 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.flow.activation=relu \
#     algorithm.flow.hidden_dim=48 \
#     algorithm.flow.n_flows=6 \
#     algorithm.flow.n_hidden_layers=1 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=5.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0002479141551240466 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 162: consistency=-2.6012e-11, logp=8.2436e+05
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp_agnostic \
#     experiment_name=physics_realnvp_pareto \
#     run_name=consistency_t162 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.flow.activation=relu \
#     algorithm.flow.hidden_dim=64 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=1 \
#     algorithm.flow.noise_scale=0.001 \
#     algorithm.flow.scale_clamp=5.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0001857825049884745 \
#     algorithm.optimizer.weight_decay=0.0 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 171: consistency=-2.4286e-18, logp=6.0079e+11  << BEST consistency | KNEE
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp_agnostic \
#     experiment_name=physics_realnvp_pareto \
#     run_name=consistency_t171 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.flow.activation=relu \
#     algorithm.flow.hidden_dim=64 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=1 \
#     algorithm.flow.noise_scale=0.001 \
#     algorithm.flow.scale_clamp=5.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.00022547781285140234 \
#     algorithm.optimizer.weight_decay=0.0 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 172: consistency=-9.8879e-17, logp=4.289e+09
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp_agnostic \
#     experiment_name=physics_realnvp_pareto \
#     run_name=consistency_t172 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.flow.activation=relu \
#     algorithm.flow.hidden_dim=64 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=1 \
#     algorithm.flow.noise_scale=0.001 \
#     algorithm.flow.scale_clamp=5.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.00023932392187527832 \
#     algorithm.optimizer.weight_decay=0.0 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 175: consistency=-7.2858e-18, logp=1.0995e+11
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp_agnostic \
#     experiment_name=physics_realnvp_pareto \
#     run_name=consistency_t175 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.flow.activation=relu \
#     algorithm.flow.hidden_dim=64 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=1 \
#     algorithm.flow.noise_scale=0.001 \
#     algorithm.flow.scale_clamp=5.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0002387424154199537 \
#     algorithm.optimizer.weight_decay=0.0 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 204: consistency=-3.3585e-13, logp=7.4351e+08
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp_agnostic \
#     experiment_name=physics_realnvp_pareto \
#     run_name=consistency_t204 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.flow.activation=relu \
#     algorithm.flow.hidden_dim=64 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=1 \
#     algorithm.flow.noise_scale=0.001 \
#     algorithm.flow.scale_clamp=5.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0002516336342859234 \
#     algorithm.optimizer.weight_decay=0.0 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 210: consistency=-5.0522e-06, logp=63.58
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp_agnostic \
#     experiment_name=physics_realnvp_pareto \
#     run_name=consistency_t210 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.flow.activation=relu \
#     algorithm.flow.hidden_dim=48 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=1 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=5.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.00020361810755846084 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 227: consistency=-2.828e-12, logp=1.337e+07
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp_agnostic \
#     experiment_name=physics_realnvp_pareto \
#     run_name=consistency_t227 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.flow.activation=relu \
#     algorithm.flow.hidden_dim=48 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=1 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=5.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0002786297030981475 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 232: consistency=-1.0538e-06, logp=95.692
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp_agnostic \
#     experiment_name=physics_realnvp_pareto \
#     run_name=consistency_t232 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.flow.activation=relu \
#     algorithm.flow.hidden_dim=48 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=1 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=5.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0002771652360779097 \
#     algorithm.optimizer.weight_decay=0.0 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 480: consistency=-0.14358, logp=4.9814
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp_agnostic \
#     experiment_name=physics_realnvp_pareto \
#     run_name=consistency_t480 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.flow.activation=relu \
#     algorithm.flow.hidden_dim=64 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=5.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.000734601865014836 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 517: consistency=-2.5003e-10, logp=4015.5
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp_agnostic \
#     experiment_name=physics_realnvp_pareto \
#     run_name=consistency_t517 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.flow.activation=relu \
#     algorithm.flow.hidden_dim=48 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=5.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.00017590004814952519 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ========================================================================
# STABILITY TRAINING  (study: drift_vs_logp_b16k, 4 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 352: drift=0.039093, logp=15.739  << KNEE
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp_agnostic \
#     experiment_name=physics_realnvp_pareto \
#     run_name=stability_t352 \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.flow.activation=gelu \
#     algorithm.flow.hidden_dim=64 \
#     algorithm.flow.n_flows=6 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0010871488886121502 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 505: drift=0.013551, logp=18.532  << ORIGINAL PICK | BEST drift
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp_agnostic \
#     experiment_name=physics_realnvp_pareto \
#     run_name=stability_t505 \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.flow.activation=gelu \
#     algorithm.flow.hidden_dim=64 \
#     algorithm.flow.n_flows=6 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0010542663916098317 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 511: drift=0.12796, logp=12.687
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp_agnostic \
#     experiment_name=physics_realnvp_pareto \
#     run_name=stability_t511 \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.flow.activation=gelu \
#     algorithm.flow.hidden_dim=64 \
#     algorithm.flow.n_flows=6 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0010396427392687347 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 528: drift=0.089103, logp=15.256
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp_agnostic \
#     experiment_name=physics_realnvp_pareto \
#     run_name=stability_t528 \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.flow.activation=gelu \
#     algorithm.flow.hidden_dim=64 \
#     algorithm.flow.n_flows=6 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0010036883712480243 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ========================================================================
# WASSERSTEIN TRAINING  (study: wasserstein_vs_logp_b16k, 4 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 181: wasserstein=0.00038952, logp=33.744  << BEST wasserstein
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp_agnostic \
#     experiment_name=physics_realnvp_pareto \
#     run_name=wasserstein_t181 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.flow.activation=relu \
#     algorithm.flow.hidden_dim=64 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=5.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0016793846197251674 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 231: wasserstein=0.0017399, logp=0.39724
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp_agnostic \
#     experiment_name=physics_realnvp_pareto \
#     run_name=wasserstein_t231 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.flow.activation=relu \
#     algorithm.flow.hidden_dim=64 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=5.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.001522150061499171 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 305: wasserstein=0.00064793, logp=0.5243  << KNEE
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp_agnostic \
#     experiment_name=physics_realnvp_pareto \
#     run_name=wasserstein_t305 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.flow.activation=relu \
#     algorithm.flow.hidden_dim=64 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=5.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0018419982523953588 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 383: wasserstein=0.0022136, logp=-1.174  << ORIGINAL PICK
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/realnvp_agnostic \
#     experiment_name=physics_realnvp_pareto \
#     run_name=wasserstein_t383 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.flow.activation=relu \
#     algorithm.flow.hidden_dim=64 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=5.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0016390004227671991 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ========================================================================
# SUBMIT EVERYTHING ABOVE TO CLARIDEN  (one slurm job per command)
# ========================================================================
# Set paths.raw_data_dir to the data location on clariden; any extra
# hydra overrides appended here are added to every job.
# bash scripts/cluster/submit_pareto.sh scripts/physics/runrealnvp_pareto.sh \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files
