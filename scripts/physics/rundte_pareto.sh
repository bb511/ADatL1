#!/usr/bin/env bash
# ========================================================================
# DTE PARETO-FRONT TRAINING COMMANDS
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
# CVAR25 TRAINING  (study: cvar25eff_vs_ascore, 14 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 58: cvar25eff=1.6875, ascore=0.44247
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_pareto \
#     run_name=cvar25_t58 \
#     algorithm.beta_end=0.00749056138898044 \
#     algorithm.n_bins=5 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0010077391038400452 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.hidden_dims='[64,32,32]' \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 68: cvar25eff=1.0488, ascore=0.33797
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_pareto \
#     run_name=cvar25_t68 \
#     algorithm.beta_end=0.0732674606359458 \
#     algorithm.n_bins=5 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0013196881059904526 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.hidden_dims='[128,64,32]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 122: cvar25eff=1.2416, ascore=0.35097
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_pareto \
#     run_name=cvar25_t122 \
#     algorithm.beta_end=0.02762479516320321 \
#     algorithm.n_bins=5 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0027869411352875947 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.hidden_dims='[128,64]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 126: cvar25eff=1.3848, ascore=0.3799
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_pareto \
#     run_name=cvar25_t126 \
#     algorithm.beta_end=0.012163268878855568 \
#     algorithm.n_bins=20 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.00032358366892793427 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.5 \
#     algorithm.predictor.hidden_dims='[64,32,16]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 194: cvar25eff=1.9464, ascore=0.44358
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_pareto \
#     run_name=cvar25_t194 \
#     algorithm.beta_end=0.012193044715840703 \
#     algorithm.n_bins=5 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.001696439216751085 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.hidden_dims='[64,32,16]' \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 206: cvar25eff=1.5011, ascore=0.42305
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_pareto \
#     run_name=cvar25_t206 \
#     algorithm.beta_end=0.013581799840166471 \
#     algorithm.n_bins=5 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0021635439101803195 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.hidden_dims='[64,32,16]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 277: cvar25eff=1.3976, ascore=0.3983
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_pareto \
#     run_name=cvar25_t277 \
#     algorithm.beta_end=0.03941073550151119 \
#     algorithm.n_bins=15 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0016561978126353432 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.hidden_dims='[128,64,32]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 307: cvar25eff=3.9404, ascore=0.5007  << BEST cvar25eff
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_pareto \
#     run_name=cvar25_t307 \
#     algorithm.beta_end=0.0011035841737487488 \
#     algorithm.n_bins=15 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0029913777200930736 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.hidden_dims='[128,64,32]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 401: cvar25eff=3.4436, ascore=0.48947
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_pareto \
#     run_name=cvar25_t401 \
#     algorithm.beta_end=0.0011235850845387379 \
#     algorithm.n_bins=10 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.002543249757880344 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.hidden_dims='[128,64,32]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 465: cvar25eff=1.0856, ascore=0.34472
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_pareto \
#     run_name=cvar25_t465 \
#     algorithm.beta_end=0.06303726399596656 \
#     algorithm.n_bins=15 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0029984745267809523 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.hidden_dims='[128,64,32]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 501: cvar25eff=1.2428, ascore=0.36718
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_pareto \
#     run_name=cvar25_t501 \
#     algorithm.beta_end=0.03958423072133179 \
#     algorithm.n_bins=15 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0027785272236042482 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.hidden_dims='[128,64,32]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 527: cvar25eff=3.3356, ascore=0.47506
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_pareto \
#     run_name=cvar25_t527 \
#     algorithm.beta_end=0.0011244223403293941 \
#     algorithm.n_bins=15 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.002774694364229204 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.5 \
#     algorithm.predictor.hidden_dims='[128,64,32]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 545: cvar25eff=1.6461, ascore=0.44121  << KNEE
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_pareto \
#     run_name=cvar25_t545 \
#     algorithm.beta_end=0.011135716341697528 \
#     algorithm.n_bins=15 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.002996609873208587 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.5 \
#     algorithm.predictor.hidden_dims='[128,64,32]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 557: cvar25eff=3.2708, ascore=0.46542
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_pareto \
#     run_name=cvar25_t557 \
#     algorithm.beta_end=0.0011680375639881346 \
#     algorithm.n_bins=15 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.002762618455516249 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.5 \
#     algorithm.predictor.hidden_dims='[128,64,32]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ========================================================================
# CVAR10 TRAINING  (study: cvar10eff_vs_ascore, 6 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 219: cvar10eff=0.51877, ascore=0.32652  << KNEE
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_pareto \
#     run_name=cvar10_t219 \
#     evaluation.callbacks.anomaly_efficiency.cvar_summary=0.10 \
#     algorithm.beta_end=0.02205651590544173 \
#     algorithm.n_bins=30.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0016024416367267224 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.5 \
#     algorithm.predictor.hidden_dims='[64,32,32]' \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 282: cvar10eff=0.26091, ascore=0.32356
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_pareto \
#     run_name=cvar10_t282 \
#     evaluation.callbacks.anomaly_efficiency.cvar_summary=0.10 \
#     algorithm.beta_end=0.04379645671511953 \
#     algorithm.n_bins=30.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.002964467487821067 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.5 \
#     algorithm.predictor.hidden_dims='[64,32,16]' \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 291: cvar10eff=0.28455, ascore=0.32493
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_pareto \
#     run_name=cvar10_t291 \
#     evaluation.callbacks.anomaly_efficiency.cvar_summary=0.10 \
#     algorithm.beta_end=0.044772918823879716 \
#     algorithm.n_bins=30.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0029568725505825745 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.5 \
#     algorithm.predictor.hidden_dims='[64,32,16]' \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 292: cvar10eff=0.26014, ascore=0.29965
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_pareto \
#     run_name=cvar10_t292 \
#     evaluation.callbacks.anomaly_efficiency.cvar_summary=0.10 \
#     algorithm.beta_end=0.044356233941550666 \
#     algorithm.n_bins=30.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0029775202162820444 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.5 \
#     algorithm.predictor.hidden_dims='[64,32,16]' \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 336: cvar10eff=0.21896, ascore=0.29567
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_pareto \
#     run_name=cvar10_t336 \
#     evaluation.callbacks.anomaly_efficiency.cvar_summary=0.10 \
#     algorithm.beta_end=0.020148700397133838 \
#     algorithm.n_bins=30.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.001695950534027964 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.5 \
#     algorithm.predictor.hidden_dims='[64,32,16]' \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 545: cvar10eff=0.84986, ascore=0.52574  << BEST cvar10eff
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_pareto \
#     run_name=cvar10_t545 \
#     evaluation.callbacks.anomaly_efficiency.cvar_summary=0.10 \
#     algorithm.beta_end=0.0013586760316323956 \
#     algorithm.n_bins=30.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0016605079688725795 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.5 \
#     algorithm.predictor.hidden_dims='[64,32,32]' \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ========================================================================
# CAP TRAINING  (study: cap_vs_ascore, 16 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 1: cap=-0.099675, ascore=0.55197  << BEST cap
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_pareto \
#     run_name=cap_t1 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.0014930571071433533 \
#     algorithm.n_bins=10 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=3.683792327535734e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.hidden_dims='[64,32,16]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 41: cap=-0.3679, ascore=0.29315
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_pareto \
#     run_name=cap_t41 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.0817358594611716 \
#     algorithm.n_bins=30 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.002453394466565529 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.hidden_dims='[128,64,32]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 62: cap=-0.11182, ascore=0.53093
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_pareto \
#     run_name=cap_t62 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.001464350964286908 \
#     algorithm.n_bins=30 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=8.53245148064481e-05 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.hidden_dims='[64,32,32]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 119: cap=-0.11712, ascore=0.5116
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_pareto \
#     run_name=cap_t119 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.001664951455817677 \
#     algorithm.n_bins=15 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=4.09979552848601e-05 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.dropout=0.5 \
#     algorithm.predictor.hidden_dims='[64,32,16]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 147: cap=-0.15276, ascore=0.50831
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_pareto \
#     run_name=cap_t147 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.007184510639871497 \
#     algorithm.n_bins=15 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0013836899978350954 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.hidden_dims='[64,32]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 177: cap=-0.11282, ascore=0.5231
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_pareto \
#     run_name=cap_t177 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.0015221182935409364 \
#     algorithm.n_bins=15 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=5.508559562537512e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.hidden_dims='[64,32,16]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 186: cap=-0.16145, ascore=0.363
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_pareto \
#     run_name=cap_t186 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.019225500330675143 \
#     algorithm.n_bins=15 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0010259531483760207 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.hidden_dims='[64,32,32]' \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 189: cap=-0.16198, ascore=0.36156
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_pareto \
#     run_name=cap_t189 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.01940311129125142 \
#     algorithm.n_bins=15 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0010291121809870181 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.hidden_dims='[64,32,32]' \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 214: cap=-0.17248, ascore=0.32793  << KNEE
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_pareto \
#     run_name=cap_t214 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.031267327626381805 \
#     algorithm.n_bins=15 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0007373143439849586 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.hidden_dims='[64,32,32]' \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 219: cap=-0.15508, ascore=0.44925
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_pareto \
#     run_name=cap_t219 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.026426444918953265 \
#     algorithm.n_bins=30 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0007213042111591376 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.hidden_dims='[128,64,32]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 458: cap=-0.57911, ascore=0.27642
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_pareto \
#     run_name=cap_t458 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.05479819396192555 \
#     algorithm.n_bins=7 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.002849929559631248 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.hidden_dims='[128,64,32]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 521: cap=-0.57234, ascore=0.27986
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_pareto \
#     run_name=cap_t521 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.0485414166087852 \
#     algorithm.n_bins=7 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0026403176560665853 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.hidden_dims='[128,64,32]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 592: cap=-0.35972, ascore=0.29847
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_pareto \
#     run_name=cap_t592 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.07221529568171373 \
#     algorithm.n_bins=30 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.002985698873940812 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.hidden_dims='[128,64,32]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 593: cap=-0.35621, ascore=0.2992
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_pareto \
#     run_name=cap_t593 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.07193829971883449 \
#     algorithm.n_bins=30 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.002978962302232853 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.hidden_dims='[128,64,32]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 616: cap=-0.17245, ascore=0.34197
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_pareto \
#     run_name=cap_t616 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.031139437514802006 \
#     algorithm.n_bins=15 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0007033480268811592 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.hidden_dims='[64,32,32]' \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 621: cap=-0.10232, ascore=0.53896
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_pareto \
#     run_name=cap_t621 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.001588139859758886 \
#     algorithm.n_bins=7 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=4.53640142984742e-05 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.dropout=0.5 \
#     algorithm.predictor.hidden_dims='[64,32,16]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ========================================================================
# CONSISTENCY TRAINING  (study: consistency_vs_ascore, 12 Pareto points, trimmed to 11 around the knee (endpoints kept))
# ========================================================================
# ------------------------------------------------------------------------
# trial 204: consistency=-0.00021696, ascore=0.39761  << BEST consistency
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_pareto \
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
#     algorithm.beta_end=0.06426085202802459 \
#     algorithm.n_bins=5.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0018731746578299398 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.hidden_dims='[64,32,32]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 208: consistency=-0.00033096, ascore=0.35763
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_pareto \
#     run_name=consistency_t208 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.06430401154392738 \
#     algorithm.n_bins=5.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0018767418906131094 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.hidden_dims='[64,32,32]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 223: consistency=-0.0020605, ascore=0.33996
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_pareto \
#     run_name=consistency_t223 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.06510214187669368 \
#     algorithm.n_bins=20.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0014153778991185806 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.hidden_dims='[64,32,32]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 347: consistency=-0.00055363, ascore=0.3464
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_pareto \
#     run_name=consistency_t347 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.0575355028792373 \
#     algorithm.n_bins=5.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0012875364054476088 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.hidden_dims='[64,32,32]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 352: consistency=-0.00053924, ascore=0.34986
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_pareto \
#     run_name=consistency_t352 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.050882926609688406 \
#     algorithm.n_bins=5.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0011320276005818578 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.hidden_dims='[64,32,32]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 353: consistency=-0.00046865, ascore=0.35317
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_pareto \
#     run_name=consistency_t353 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.05100810390713008 \
#     algorithm.n_bins=5.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0011254171959482644 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.hidden_dims='[64,32,32]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 368: consistency=-0.0060716, ascore=0.30888  << KNEE
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_pareto \
#     run_name=consistency_t368 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.07165100884901118 \
#     algorithm.n_bins=20.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0015828660078376195 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.hidden_dims='[64,32,32]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 434: consistency=-0.0043265, ascore=0.33883
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_pareto \
#     run_name=consistency_t434 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.07346043237443294 \
#     algorithm.n_bins=20.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.001505541029590358 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.hidden_dims='[64,32,32]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 458: consistency=-0.011344, ascore=0.30364
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_pareto \
#     run_name=consistency_t458 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.06806556887479132 \
#     algorithm.n_bins=20.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0016384534540630944 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.5 \
#     algorithm.predictor.hidden_dims='[64,32,16]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 474: consistency=-0.019874, ascore=0.29245
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_pareto \
#     run_name=consistency_t474 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.06692488832995255 \
#     algorithm.n_bins=20.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.001599513854254573 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.5 \
#     algorithm.predictor.hidden_dims='[64,32,16]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 512: consistency=-0.026691, ascore=0.28917
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_pareto \
#     run_name=consistency_t512 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.06790707146354132 \
#     algorithm.n_bins=20.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0014635479726218155 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.5 \
#     algorithm.predictor.hidden_dims='[64,32,16]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ========================================================================
# STABILITY TRAINING  (study: drift_vs_ascore, 3 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 185: drift=0.039093, ascore=0.30744  << KNEE
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_pareto \
#     run_name=stability_t185 \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.043581359221426065 \
#     algorithm.n_bins=15.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0003209426980616469 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.5 \
#     algorithm.predictor.hidden_dims='[64,32,16]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 304: drift=0.19048, ascore=0.30617
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_pareto \
#     run_name=stability_t304 \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.040267911639664 \
#     algorithm.n_bins=15.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0003688655763509913 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.5 \
#     algorithm.predictor.hidden_dims='[64,32,16]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 457: drift=0.013551, ascore=0.31099  << BEST drift
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_pareto \
#     run_name=stability_t457 \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.04226123529003391 \
#     algorithm.n_bins=15.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.002586718097813541 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.5 \
#     algorithm.predictor.hidden_dims='[64,32,16]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ========================================================================
# WASSERSTEIN TRAINING  (study: wasserstein_vs_ascore, 8 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 68: wasserstein=5.9585e-05, ascore=0.53909  << BEST wasserstein
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_pareto \
#     run_name=wasserstein_t68 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.004213652097556574 \
#     algorithm.n_bins=15.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=4.1901881816609435e-05 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.hidden_dims='[64,32,16]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 335: wasserstein=6.6122e-05, ascore=0.32785
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_pareto \
#     run_name=wasserstein_t335 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.07913691783743441 \
#     algorithm.n_bins=5.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0027567029244424295 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.hidden_dims='[128,64]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 344: wasserstein=0.0001092, ascore=0.32625
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_pareto \
#     run_name=wasserstein_t344 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.09052027626426597 \
#     algorithm.n_bins=5.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.002995323245820316 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.hidden_dims='[128,64]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 349: wasserstein=8.8428e-05, ascore=0.32721
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_pareto \
#     run_name=wasserstein_t349 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.08047839205952698 \
#     algorithm.n_bins=5.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.002731148304539135 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.hidden_dims='[128,64]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 352: wasserstein=0.00013695, ascore=0.3234
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_pareto \
#     run_name=wasserstein_t352 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.08040100591176842 \
#     algorithm.n_bins=5.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0027253855188714986 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.hidden_dims='[128,64]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 367: wasserstein=6.4729e-05, ascore=0.33039  << KNEE
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_pareto \
#     run_name=wasserstein_t367 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.08921773150073678 \
#     algorithm.n_bins=5.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.002999210411507182 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.hidden_dims='[128,64]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 499: wasserstein=0.00015147, ascore=0.28277
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_pareto \
#     run_name=wasserstein_t499 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.07798729135573579 \
#     algorithm.n_bins=5.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0025046165140447632 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.hidden_dims='[128,64]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 502: wasserstein=0.00014733, ascore=0.31332
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_pareto \
#     run_name=wasserstein_t502 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.07813742058698986 \
#     algorithm.n_bins=5.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0025111640801980333 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.hidden_dims='[128,64]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ========================================================================
# SUBMIT EVERYTHING ABOVE TO CLARIDEN  (one slurm job per command)
# ========================================================================
# Set paths.raw_data_dir to the data location on clariden; any extra
# hydra overrides appended here are added to every job.
# bash scripts/cluster/submit_pareto.sh scripts/physics/rundte_pareto.sh \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files
