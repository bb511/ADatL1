#!/usr/bin/env bash
# ========================================================================
# VAE PARETO-FRONT TRAINING COMMANDS
# ========================================================================
# q99 background-rate study: training commands for every point on the Pareto front of
# each validation strategy. Generated from notebooks/paretos/physics/ by
# scripts/optuna/make_pareto_scripts.py -- regenerate rather than edit by hand.
#
# Run from the repository root. All commands are commented out -- uncomment
# the points you want to run locally (taskset pinning, GPUs cycling 0-3).
# To run the WHOLE file on clariden instead, use the single submit command
# at the bottom: it sends every point above to slurm, one job each, via
# scripts/cluster/submit_pareto.sh (submitit launcher).

# ========================================================================
# CVAR25 TRAINING  (study: cvar25eff_vs_klq99_b16k, 11 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 179: cvar25eff=0.75343, klq99=11.921  << BEST cvar25eff
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/vae \
#     experiment_name=physics_vae_q99_pareto \
#     run_name=cvar25_t179 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.clamp_zlogvar_range='[-20,10]' \
#     algorithm.encoder.nodes='[64,32,24]' \
#     algorithm.kl_warmup_frac=0.05 \
#     algorithm.kl_scale=0.002 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0003116215519177581 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 242: cvar25eff=0.65866, klq99=9.4542
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/vae \
#     experiment_name=physics_vae_q99_pareto \
#     run_name=cvar25_t242 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.clamp_zlogvar_range='[-20,10]' \
#     algorithm.encoder.nodes='[32,16,4]' \
#     algorithm.kl_warmup_frac=0.1 \
#     algorithm.kl_scale=0.002 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.00028723727790733807 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 243: cvar25eff=0.62499, klq99=9.4477
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/vae \
#     experiment_name=physics_vae_q99_pareto \
#     run_name=cvar25_t243 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.clamp_zlogvar_range='[-20,10]' \
#     algorithm.encoder.nodes='[32,16,4]' \
#     algorithm.kl_warmup_frac=0.1 \
#     algorithm.kl_scale=0.002 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0002872268792982993 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 275: cvar25eff=0.75341, klq99=11.771  << ORIGINAL PICK
#   caveat: old script had algorithm.encoder.nodes=[64,32,32]; db trial 275 has [64,32,24]
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/vae \
#     experiment_name=physics_vae_q99_pareto \
#     run_name=cvar25_t275 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.clamp_zlogvar_range='[-20,10]' \
#     algorithm.encoder.nodes='[64,32,24]' \
#     algorithm.kl_warmup_frac=0.05 \
#     algorithm.kl_scale=0.002 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0003005101736177465 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 279: cvar25eff=0.61146, klq99=9.4175
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/vae \
#     experiment_name=physics_vae_q99_pareto \
#     run_name=cvar25_t279 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.clamp_zlogvar_range='[-20,10]' \
#     algorithm.encoder.nodes='[32,16,4]' \
#     algorithm.kl_warmup_frac=0.1 \
#     algorithm.kl_scale=0.002 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.00026456183565594 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 337: cvar25eff=0.73493, klq99=10.401
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/vae \
#     experiment_name=physics_vae_q99_pareto \
#     run_name=cvar25_t337 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.clamp_zlogvar_range='[-20,10]' \
#     algorithm.encoder.nodes='[48,16,8]' \
#     algorithm.kl_warmup_frac=0.1 \
#     algorithm.kl_scale=0.002 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.00026079976477308127 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 423: cvar25eff=0.59987, klq99=8.3002
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/vae \
#     experiment_name=physics_vae_q99_pareto \
#     run_name=cvar25_t423 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.clamp_zlogvar_range='[-20,10]' \
#     algorithm.encoder.nodes='[32,16,4]' \
#     algorithm.kl_warmup_frac=0.1 \
#     algorithm.kl_scale=0.002 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=9.784041414313416e-05 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 428: cvar25eff=0.60321, klq99=9.2809  << KNEE
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/vae \
#     experiment_name=physics_vae_q99_pareto \
#     run_name=cvar25_t428 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.clamp_zlogvar_range='[-20,10]' \
#     algorithm.encoder.nodes='[32,16,4]' \
#     algorithm.kl_warmup_frac=0.1 \
#     algorithm.kl_scale=0.002 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.00010033582802501038 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 544: cvar25eff=0.58832, klq99=1.3697
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/vae \
#     experiment_name=physics_vae_q99_pareto \
#     run_name=cvar25_t544 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.clamp_zlogvar_range='[-10,6]' \
#     algorithm.encoder.nodes='[48,16,8]' \
#     algorithm.kl_warmup_frac=0.1 \
#     algorithm.kl_scale=0.002 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=5.236233832409967e-05 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 592: cvar25eff=0.7181, klq99=10.374
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/vae \
#     experiment_name=physics_vae_q99_pareto \
#     run_name=cvar25_t592 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.clamp_zlogvar_range='[-20,10]' \
#     algorithm.encoder.nodes='[48,16,8]' \
#     algorithm.kl_warmup_frac=0.1 \
#     algorithm.kl_scale=0.002 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0002598362463919353 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 593: cvar25eff=0.71377, klq99=10.355
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/vae \
#     experiment_name=physics_vae_q99_pareto \
#     run_name=cvar25_t593 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.clamp_zlogvar_range='[-20,10]' \
#     algorithm.encoder.nodes='[48,16,8]' \
#     algorithm.kl_warmup_frac=0.1 \
#     algorithm.kl_scale=0.002 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.00025954151805076907 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ========================================================================
# CAP TRAINING  (study: cap_vs_klq99_b16k, 9 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 199: cap=-0.16323, klq99=9.0245
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/vae_agnostic \
#     experiment_name=physics_vae_q99_pareto \
#     run_name=cap_t199 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.clamp_zlogvar_range='[-20,10]' \
#     algorithm.encoder.nodes='[24,4]' \
#     algorithm.kl_warmup_frac=0.05 \
#     algorithm.kl_scale=0.002 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0004741364222465262 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 337: cap=-0.13789, klq99=11.453  << BEST cap
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/vae_agnostic \
#     experiment_name=physics_vae_q99_pareto \
#     run_name=cap_t337 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.clamp_zlogvar_range='[-20,10]' \
#     algorithm.encoder.nodes='[48,24,8]' \
#     algorithm.kl_warmup_frac=0.05 \
#     algorithm.kl_scale=0.002 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0003072659350788133 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 447: cap=-0.15021, klq99=9.3433
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/vae_agnostic \
#     experiment_name=physics_vae_q99_pareto \
#     run_name=cap_t447 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.clamp_zlogvar_range='[-20,10]' \
#     algorithm.encoder.nodes='[32,16,4]' \
#     algorithm.kl_warmup_frac=0.05 \
#     algorithm.kl_scale=0.002 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.00031377607758620004 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 457: cap=-0.15094, klq99=9.2923
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/vae_agnostic \
#     experiment_name=physics_vae_q99_pareto \
#     run_name=cap_t457 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.clamp_zlogvar_range='[-20,10]' \
#     algorithm.encoder.nodes='[32,16,4]' \
#     algorithm.kl_warmup_frac=0.05 \
#     algorithm.kl_scale=0.002 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0003121904587141562 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 473: cap=-0.14577, klq99=9.3667
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/vae_agnostic \
#     experiment_name=physics_vae_q99_pareto \
#     run_name=cap_t473 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.clamp_zlogvar_range='[-20,10]' \
#     algorithm.encoder.nodes='[32,16,4]' \
#     algorithm.kl_warmup_frac=0.05 \
#     algorithm.kl_scale=0.002 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0003213689213330541 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 474: cap=-0.14169, klq99=9.441
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/vae_agnostic \
#     experiment_name=physics_vae_q99_pareto \
#     run_name=cap_t474 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.clamp_zlogvar_range='[-20,10]' \
#     algorithm.encoder.nodes='[32,16,4]' \
#     algorithm.kl_warmup_frac=0.05 \
#     algorithm.kl_scale=0.002 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.00032150045630309287 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 480: cap=-0.13957, klq99=9.4936
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/vae_agnostic \
#     experiment_name=physics_vae_q99_pareto \
#     run_name=cap_t480 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.clamp_zlogvar_range='[-20,10]' \
#     algorithm.encoder.nodes='[32,16,4]' \
#     algorithm.kl_warmup_frac=0.05 \
#     algorithm.kl_scale=0.002 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0003264027635667405 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 569: cap=-0.15959, klq99=9.2623
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/vae_agnostic \
#     experiment_name=physics_vae_q99_pareto \
#     run_name=cap_t569 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.clamp_zlogvar_range='[-20,10]' \
#     algorithm.encoder.nodes='[32,16,4]' \
#     algorithm.kl_warmup_frac=0.05 \
#     algorithm.kl_scale=0.002 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.00031529187167450085 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 577: cap=-0.13965, klq99=9.4663  << ORIGINAL PICK | KNEE
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/vae_agnostic \
#     experiment_name=physics_vae_q99_pareto \
#     run_name=cap_t577 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.clamp_zlogvar_range='[-20,10]' \
#     algorithm.encoder.nodes='[32,16,4]' \
#     algorithm.kl_warmup_frac=0.05 \
#     algorithm.kl_scale=0.002 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0003257255014602454 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ========================================================================
# STABILITY TRAINING  (study: drift_vs_klq99_b16k, 5 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 371: drift=1.2428e-05, klq99=0.25253  << BEST drift
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/vae_agnostic \
#     experiment_name=physics_vae_q99_pareto \
#     run_name=stability_t371 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.clamp_zlogvar_range='[-6,4]' \
#     algorithm.encoder.nodes='[24,8,4]' \
#     algorithm.kl_warmup_frac=0.3 \
#     algorithm.kl_scale=0.002 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=5.264689172190425e-05 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 433: drift=3.5462e-05, klq99=0.23479  << KNEE
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/vae_agnostic \
#     experiment_name=physics_vae_q99_pareto \
#     run_name=stability_t433 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.clamp_zlogvar_range='[-6,4]' \
#     algorithm.encoder.nodes='[24,8,4]' \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.kl_scale=0.002 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=5.000785389905901e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 437: drift=0.00044334, klq99=0.2307
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/vae_agnostic \
#     experiment_name=physics_vae_q99_pareto \
#     run_name=stability_t437 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.clamp_zlogvar_range='[-6,4]' \
#     algorithm.encoder.nodes='[24,8,4]' \
#     algorithm.kl_warmup_frac=0.3 \
#     algorithm.kl_scale=0.002 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=5.0014930692817375e-05 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 523: drift=0.00029972, klq99=0.2327
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/vae_agnostic \
#     experiment_name=physics_vae_q99_pareto \
#     run_name=stability_t523 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.clamp_zlogvar_range='[-6,4]' \
#     algorithm.encoder.nodes='[24,8,4]' \
#     algorithm.kl_warmup_frac=0.0 \
#     algorithm.kl_scale=0.001 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=5.0000379952775584e-05 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 567: drift=0.00039547, klq99=0.23072  << ORIGINAL PICK
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/vae_agnostic \
#     experiment_name=physics_vae_q99_pareto \
#     run_name=stability_t567 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.clamp_zlogvar_range='[-6,4]' \
#     algorithm.encoder.nodes='[24,8,4]' \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.kl_scale=0.001 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=5.0003969774639385e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ========================================================================
# WASSERSTEIN TRAINING  (study: wasserstein_vs_klq99_b16k, 345 Pareto points, trimmed to 12 around the knee (endpoints kept))
# ========================================================================
# ------------------------------------------------------------------------
# trial 160: wasserstein=0.0007095, klq99=0.33278
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/vae_agnostic \
#     experiment_name=physics_vae_q99_pareto \
#     run_name=wasserstein_t160 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.clamp_zlogvar_range='[-20,10]' \
#     algorithm.encoder.nodes='[24,8,4]' \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.kl_scale=0.0003 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=5.96279527351775e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 161: wasserstein=0.00070091, klq99=0.33371
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/vae_agnostic \
#     experiment_name=physics_vae_q99_pareto \
#     run_name=wasserstein_t161 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.clamp_zlogvar_range='[-20,10]' \
#     algorithm.encoder.nodes='[24,8,4]' \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.kl_scale=0.0003 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=5.972743945313945e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 162: wasserstein=0.00070786, klq99=0.33295
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/vae_agnostic \
#     experiment_name=physics_vae_q99_pareto \
#     run_name=wasserstein_t162 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.clamp_zlogvar_range='[-20,10]' \
#     algorithm.encoder.nodes='[24,8,4]' \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.kl_scale=0.0003 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=5.964625289983021e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 163: wasserstein=0.00071619, klq99=0.33217
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/vae_agnostic \
#     experiment_name=physics_vae_q99_pareto \
#     run_name=wasserstein_t163 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.clamp_zlogvar_range='[-20,10]' \
#     algorithm.encoder.nodes='[24,8,4]' \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.kl_scale=0.0003 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=5.9562224662998264e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 165: wasserstein=0.00072821, klq99=0.33115
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/vae_agnostic \
#     experiment_name=physics_vae_q99_pareto \
#     run_name=wasserstein_t165 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.clamp_zlogvar_range='[-20,10]' \
#     algorithm.encoder.nodes='[24,8,4]' \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.kl_scale=0.0003 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=5.945229642478961e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 173: wasserstein=0.00072851, klq99=0.33112
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/vae_agnostic \
#     experiment_name=physics_vae_q99_pareto \
#     run_name=wasserstein_t173 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.clamp_zlogvar_range='[-20,10]' \
#     algorithm.encoder.nodes='[24,8,4]' \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.kl_scale=0.0003 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=5.944906758840489e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 174: wasserstein=0.00071596, klq99=0.33219  << KNEE
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/vae_agnostic \
#     experiment_name=physics_vae_q99_pareto \
#     run_name=wasserstein_t174 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.clamp_zlogvar_range='[-20,10]' \
#     algorithm.encoder.nodes='[24,8,4]' \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.kl_scale=0.0003 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=5.9564289866078115e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 178: wasserstein=0.00072888, klq99=0.3311
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/vae_agnostic \
#     experiment_name=physics_vae_q99_pareto \
#     run_name=wasserstein_t178 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.clamp_zlogvar_range='[-20,10]' \
#     algorithm.encoder.nodes='[24,8,4]' \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.kl_scale=0.0003 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=5.944602771344287e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 179: wasserstein=0.00069856, klq99=0.33397
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/vae_agnostic \
#     experiment_name=physics_vae_q99_pareto \
#     run_name=wasserstein_t179 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.clamp_zlogvar_range='[-20,10]' \
#     algorithm.encoder.nodes='[24,8,4]' \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.kl_scale=0.0003 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=5.975482498254285e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 504: wasserstein=0.00066921, klq99=0.3402  << ORIGINAL PICK | BEST wasserstein
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/vae_agnostic \
#     experiment_name=physics_vae_q99_pareto \
#     run_name=wasserstein_t504 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.clamp_zlogvar_range='[-20,10]' \
#     algorithm.encoder.nodes='[24,8,4]' \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.kl_scale=0.0003 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=6.040999153331964e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 554: wasserstein=0.0015933, klq99=0.25817
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/vae_agnostic \
#     experiment_name=physics_vae_q99_pareto \
#     run_name=wasserstein_t554 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.clamp_zlogvar_range='[-20,10]' \
#     algorithm.encoder.nodes='[24,8,4]' \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.kl_scale=0.0003 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=5.0002308599703165e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 597: wasserstein=0.00072247, klq99=0.33163
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/vae_agnostic \
#     experiment_name=physics_vae_q99_pareto \
#     run_name=wasserstein_t597 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.clamp_zlogvar_range='[-20,10]' \
#     algorithm.encoder.nodes='[24,8,4]' \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.kl_scale=0.0003 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=5.950413522510673e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ========================================================================
# SUBMIT EVERYTHING ABOVE TO CLARIDEN  (one slurm job per command)
# ========================================================================
# Set paths.raw_data_dir to the data location on clariden; any extra
# hydra overrides appended here are added to every job.
# bash scripts/cluster/submit_pareto.sh scripts/physics/runvae_q99_pareto.sh \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files
