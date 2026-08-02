#!/usr/bin/env bash
# ========================================================================
# VAE PARETO-FRONT TRAINING COMMANDS
# ========================================================================
# These are the training commands for every point on the Pareto front of
# each validation strategy. Generated from notebooks/paretos/robustad/ by
# scripts/optuna/make_pareto_scripts.py -- regenerate rather than edit by hand.
#
# Run from the repository root. All commands are commented out -- uncomment
# the points you want to run locally (taskset pinning, GPUs cycling 0-3).
# To run the WHOLE file on clariden instead, use the single submit command
# at the bottom: it sends every point above to slurm, one job each, via
# scripts/cluster/submit_pareto.sh (submitit launcher).

# ========================================================================
# CVAR25 TRAINING  (study: cvar25eff_vs_kl, 12 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 59: cvar25eff=753.33, kl=0.51127
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/vae \
#     experiment_name=robustad_vae_pareto \
#     run_name=cvar25_t59 \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=3e-05 \
#     algorithm.kl_warmup_frac=0.3 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=3.313489686348507e-05 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 62: cvar25eff=780, kl=0.5534  << ORIGINAL PICK | KNEE
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/vae \
#     experiment_name=robustad_vae_pareto \
#     run_name=cvar25_t62 \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=0.0003 \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=3.99357685959721e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 75: cvar25eff=90, kl=0.40223
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/vae \
#     experiment_name=robustad_vae_pareto \
#     run_name=cvar25_t75 \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.clamp_zlogvar_range='[-6,4]' \
#     algorithm.encoder.nodes='[8,16,16]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=0.001 \
#     algorithm.kl_warmup_frac=0.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=4.5535727700964466e-05 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 118: cvar25eff=178.33, kl=0.40464
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/vae \
#     experiment_name=robustad_vae_pareto \
#     run_name=cvar25_t118 \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.nodes='[8,16,16]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=1e-05 \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=3.1617862138294124e-05 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 165: cvar25eff=733.33, kl=0.48738
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/vae \
#     experiment_name=robustad_vae_pareto \
#     run_name=cvar25_t165 \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=0.0001 \
#     algorithm.kl_warmup_frac=0.1 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=4.447713984338708e-05 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 247: cvar25eff=1000, kl=64.862  << BEST cvar25eff
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/vae \
#     experiment_name=robustad_vae_pareto \
#     run_name=cvar25_t247 \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=True \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=0.0001 \
#     algorithm.kl_warmup_frac=0.1 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009999771714553683 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 370: cvar25eff=916.67, kl=32.269
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/vae \
#     experiment_name=robustad_vae_pareto \
#     run_name=cvar25_t370 \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=True \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=0.0003 \
#     algorithm.kl_warmup_frac=0.1 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009993658230028897 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 409: cvar25eff=891.67, kl=32.178
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/vae \
#     experiment_name=robustad_vae_pareto \
#     run_name=cvar25_t409 \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=True \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=0.0003 \
#     algorithm.kl_warmup_frac=0.1 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009997771487464005 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 425: cvar25eff=925, kl=34.751
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/vae \
#     experiment_name=robustad_vae_pareto \
#     run_name=cvar25_t425 \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=True \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=0.0003 \
#     algorithm.kl_warmup_frac=0.1 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009426712968756927 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 431: cvar25eff=958.33, kl=38.796
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/vae \
#     experiment_name=robustad_vae_pareto \
#     run_name=cvar25_t431 \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=True \
#     algorithm.encoder.clamp_zlogvar_range='[-10,6]' \
#     algorithm.encoder.nodes='[32,64,64]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=0.0003 \
#     algorithm.kl_warmup_frac=0.1 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0008524401268621665 \
#     algorithm.optimizer.weight_decay=0.0 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 480: cvar25eff=933.33, kl=37.246
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/vae \
#     experiment_name=robustad_vae_pareto \
#     run_name=cvar25_t480 \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=True \
#     algorithm.encoder.clamp_zlogvar_range='[-10,6]' \
#     algorithm.encoder.nodes='[32,64,64]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=0.0003 \
#     algorithm.kl_warmup_frac=0.1 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0008147104225007444 \
#     algorithm.optimizer.weight_decay=0.0 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 557: cvar25eff=900, kl=32.231
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/vae \
#     experiment_name=robustad_vae_pareto \
#     run_name=cvar25_t557 \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=True \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=0.0003 \
#     algorithm.kl_warmup_frac=0.1 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009997894032741622 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ========================================================================
# CAP TRAINING  (study: cap_vs_kl, 42 Pareto points, trimmed to 12 around the knee (endpoints kept))
# ========================================================================
# ------------------------------------------------------------------------
# trial 189: cap=-0.4618, kl=0.43506
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/vae_agnostic \
#     experiment_name=robustad_vae_pareto \
#     run_name=cap_t189 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.clamp_zlogvar_range='[-6,4]' \
#     algorithm.encoder.nodes='[8,16,16]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=1e-05 \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=3.0200079174699998e-05 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 194: cap=-0.46144, kl=0.43526
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/vae_agnostic \
#     experiment_name=robustad_vae_pareto \
#     run_name=cap_t194 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.clamp_zlogvar_range='[-6,4]' \
#     algorithm.encoder.nodes='[8,16,16]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=1e-05 \
#     algorithm.kl_warmup_frac=0.1 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=3.0009894487836343e-05 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 198: cap=-0.46165, kl=0.43515
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/vae_agnostic \
#     experiment_name=robustad_vae_pareto \
#     run_name=cap_t198 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.clamp_zlogvar_range='[-6,4]' \
#     algorithm.encoder.nodes='[8,16,16]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=1e-05 \
#     algorithm.kl_warmup_frac=0.1 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=3.014620977131691e-05 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 201: cap=-0.46151, kl=0.43522
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/vae_agnostic \
#     experiment_name=robustad_vae_pareto \
#     run_name=cap_t201 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.clamp_zlogvar_range='[-6,4]' \
#     algorithm.encoder.nodes='[8,16,16]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=1e-05 \
#     algorithm.kl_warmup_frac=0.1 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=3.0065947839190017e-05 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 322: cap=-0.56809, kl=0.40435
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/vae_agnostic \
#     experiment_name=robustad_vae_pareto \
#     run_name=cap_t322 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.clamp_zlogvar_range='[-6,4]' \
#     algorithm.encoder.nodes='[8,16,16]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=1e-05 \
#     algorithm.kl_warmup_frac=0.3 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=3.000039288122072e-05 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 400: cap=-0.46143, kl=0.43527  << KNEE
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/vae_agnostic \
#     experiment_name=robustad_vae_pareto \
#     run_name=cap_t400 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.clamp_zlogvar_range='[-6,4]' \
#     algorithm.encoder.nodes='[8,16,16]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=1e-05 \
#     algorithm.kl_warmup_frac=0.1 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=3.0002226965736812e-05 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 402: cap=-0.4615, kl=0.43522  << ORIGINAL PICK
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/vae_agnostic \
#     experiment_name=robustad_vae_pareto \
#     run_name=cap_t402 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.clamp_zlogvar_range='[-6,4]' \
#     algorithm.encoder.nodes='[8,16,16]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=1e-05 \
#     algorithm.kl_warmup_frac=0.1 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=3.0060074034422283e-05 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 463: cap=-0.36983, kl=0.64979
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/vae_agnostic \
#     experiment_name=robustad_vae_pareto \
#     run_name=cap_t463 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.clamp_zlogvar_range='[-6,4]' \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=1e-05 \
#     algorithm.kl_warmup_frac=0.3 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=8.867374820391195e-05 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 464: cap=-0.36952, kl=0.65137
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/vae_agnostic \
#     experiment_name=robustad_vae_pareto \
#     run_name=cap_t464 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.clamp_zlogvar_range='[-6,4]' \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=1e-05 \
#     algorithm.kl_warmup_frac=0.3 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=8.89570837092675e-05 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 465: cap=-0.36984, kl=0.64971
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/vae_agnostic \
#     experiment_name=robustad_vae_pareto \
#     run_name=cap_t465 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.clamp_zlogvar_range='[-6,4]' \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=1e-05 \
#     algorithm.kl_warmup_frac=0.3 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=8.86603236028065e-05 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 492: cap=-0.37031, kl=0.64661
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/vae_agnostic \
#     experiment_name=robustad_vae_pareto \
#     run_name=cap_t492 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.clamp_zlogvar_range='[-6,4]' \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=1e-05 \
#     algorithm.kl_warmup_frac=0.3 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=8.804327043428239e-05 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 494: cap=-0.36808, kl=0.66064  << BEST cap
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/vae_agnostic \
#     experiment_name=robustad_vae_pareto \
#     run_name=cap_t494 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.clamp_zlogvar_range='[-6,4]' \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=1e-05 \
#     algorithm.kl_warmup_frac=0.3 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=9.052799653652295e-05 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ========================================================================
# CONSISTENCY TRAINING  (study: consistency_vs_kl, 5 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 59: consistency=-0.71038, kl=0.45457
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/vae_agnostic \
#     experiment_name=robustad_vae_pareto \
#     run_name=consistency_t59 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.nodes='[8,16,16]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=3e-05 \
#     algorithm.kl_warmup_frac=0.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=3.336296531501544e-05 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 83: consistency=-0.11778, kl=35.108  << BEST consistency
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/vae_agnostic \
#     experiment_name=robustad_vae_pareto \
#     run_name=consistency_t83 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=True \
#     algorithm.encoder.clamp_zlogvar_range='[-10,6]' \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=0.001 \
#     algorithm.kl_warmup_frac=0.1 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0006708807309551948 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 185: consistency=-0.14545, kl=0.84044  << KNEE
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/vae_agnostic \
#     experiment_name=robustad_vae_pareto \
#     run_name=consistency_t185 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.clamp_zlogvar_range='[-10,6]' \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=0.0001 \
#     algorithm.kl_warmup_frac=0.1 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.00010771616601426432 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 208: consistency=-0.30374, kl=0.49458
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/vae_agnostic \
#     experiment_name=robustad_vae_pareto \
#     run_name=consistency_t208 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.clamp_zlogvar_range='[-10,6]' \
#     algorithm.encoder.nodes='[8,16,16]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=0.0003 \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=3.780415740177197e-05 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 259: consistency=-0.14201, kl=22.626
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/vae_agnostic \
#     experiment_name=robustad_vae_pareto \
#     run_name=consistency_t259 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.batchnorm=True \
#     algorithm.encoder.clamp_zlogvar_range='[-10,6]' \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=0.001 \
#     algorithm.kl_warmup_frac=0.1 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0007337808421568128 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ========================================================================
# STABILITY TRAINING  (study: drift_vs_kl, 2 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 389: drift=0.068993, kl=0.43081  << ORIGINAL PICK | BEST drift
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/vae_agnostic \
#     experiment_name=robustad_vae_pareto \
#     run_name=stability_t389 \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.nodes='[8,16,16]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=0.0003 \
#     algorithm.kl_warmup_frac=0.3 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=4.048861860453195e-05 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 598: drift=0.57982, kl=0.40066
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/vae_agnostic \
#     experiment_name=robustad_vae_pareto \
#     run_name=stability_t598 \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.clamp_zlogvar_range='[-6,4]' \
#     algorithm.encoder.nodes='[8,16,16]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=0.0003 \
#     algorithm.kl_warmup_frac=0.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=3.3767513098149e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ========================================================================
# WASSERSTEIN TRAINING  (study: wasserstein_vs_kl, 86 Pareto points, trimmed to 13 around the knee (endpoints kept))
# ========================================================================
# ------------------------------------------------------------------------
# trial 183: wasserstein=0.022592, kl=0.40538
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/vae_agnostic \
#     experiment_name=robustad_vae_pareto \
#     run_name=wasserstein_t183 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.nodes='[8,16,16]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=0.001 \
#     algorithm.kl_warmup_frac=0.3 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=3.240037635152798e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 189: wasserstein=0.022592, kl=0.40537  << KNEE
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/vae_agnostic \
#     experiment_name=robustad_vae_pareto \
#     run_name=wasserstein_t189 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.nodes='[8,16,16]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=0.001 \
#     algorithm.kl_warmup_frac=0.3 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=3.239468830257253e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 215: wasserstein=0.022593, kl=0.40536
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/vae_agnostic \
#     experiment_name=robustad_vae_pareto \
#     run_name=wasserstein_t215 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.nodes='[8,16,16]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=0.001 \
#     algorithm.kl_warmup_frac=0.3 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=3.237802188187936e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 217: wasserstein=0.022593, kl=0.40536
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/vae_agnostic \
#     experiment_name=robustad_vae_pareto \
#     run_name=wasserstein_t217 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.nodes='[8,16,16]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=0.001 \
#     algorithm.kl_warmup_frac=0.3 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=3.237074776028815e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 218: wasserstein=0.022593, kl=0.40536
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/vae_agnostic \
#     experiment_name=robustad_vae_pareto \
#     run_name=wasserstein_t218 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.nodes='[8,16,16]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=0.001 \
#     algorithm.kl_warmup_frac=0.3 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=3.2369237295553746e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 219: wasserstein=0.022593, kl=0.40535
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/vae_agnostic \
#     experiment_name=robustad_vae_pareto \
#     run_name=wasserstein_t219 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.nodes='[8,16,16]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=0.001 \
#     algorithm.kl_warmup_frac=0.3 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=3.236578433207624e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 272: wasserstein=0.022591, kl=0.40597
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/vae_agnostic \
#     experiment_name=robustad_vae_pareto \
#     run_name=wasserstein_t272 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.nodes='[8,16,16]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=0.001 \
#     algorithm.kl_warmup_frac=0.3 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=3.3280069442351026e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 274: wasserstein=0.022592, kl=0.40596
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/vae_agnostic \
#     experiment_name=robustad_vae_pareto \
#     run_name=wasserstein_t274 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.nodes='[8,16,16]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=0.001 \
#     algorithm.kl_warmup_frac=0.3 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=3.327654665317525e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 280: wasserstein=0.022591, kl=0.40598
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/vae_agnostic \
#     experiment_name=robustad_vae_pareto \
#     run_name=wasserstein_t280 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.nodes='[8,16,16]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=0.001 \
#     algorithm.kl_warmup_frac=0.3 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=3.329610622220651e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 282: wasserstein=0.022594, kl=0.40534
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/vae_agnostic \
#     experiment_name=robustad_vae_pareto \
#     run_name=wasserstein_t282 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.nodes='[8,16,16]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=0.001 \
#     algorithm.kl_warmup_frac=0.3 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=3.2347242056794795e-05 \
#     algorithm.optimizer.weight_decay=0.0 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 320: wasserstein=0.022843, kl=0.40412
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/vae_agnostic \
#     experiment_name=robustad_vae_pareto \
#     run_name=wasserstein_t320 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.nodes='[8,16,16]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=0.001 \
#     algorithm.kl_warmup_frac=0.3 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=3.000002836943652e-05 \
#     algorithm.optimizer.weight_decay=0.0 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 585: wasserstein=0.021796, kl=0.43065  << BEST wasserstein
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/vae_agnostic \
#     experiment_name=robustad_vae_pareto \
#     run_name=wasserstein_t585 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.nodes='[8,16,16]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=0.001 \
#     algorithm.kl_warmup_frac=0.3 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=4.245279802627297e-05 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 587: wasserstein=0.021799, kl=0.43064  << ORIGINAL PICK
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/vae_agnostic \
#     experiment_name=robustad_vae_pareto \
#     run_name=wasserstein_t587 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.nodes='[8,16,16]' \
#     algorithm.encoder.strides='[2]' \
#     algorithm.kl_scale=0.001 \
#     algorithm.kl_warmup_frac=0.3 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=4.2327787012183054e-05 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ========================================================================
# SUBMIT EVERYTHING ABOVE TO CLARIDEN  (one slurm job per command)
# ========================================================================
# This domain's data is downloaded rather than read from a raw ntuple path,
# so the blocks above carry no paths.raw_data_dir and none may be passed:
# submit_pareto.sh aborts on the '/path/to/...' placeholder. Any other hydra
# overrides appended here are added to every job.
# bash scripts/cluster/submit_pareto.sh scripts/robustad/runvae_pareto.sh
