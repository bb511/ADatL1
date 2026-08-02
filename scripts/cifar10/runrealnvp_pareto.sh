#!/usr/bin/env bash
# ========================================================================
# REALNVP PARETO-FRONT TRAINING COMMANDS
# ========================================================================
# These are the training commands for every point on the Pareto front of
# each validation strategy. Generated from notebooks/paretos/cifar10/ by
# scripts/optuna/make_pareto_scripts.py -- regenerate rather than edit by hand.
#
# Run from the repository root. All commands are commented out -- uncomment
# the points you want to run locally (taskset pinning, GPUs cycling 0-3).
# To run the WHOLE file on clariden instead, use the single submit command
# at the bottom: it sends every point above to slurm, one job each, via
# scripts/cluster/submit_pareto.sh (submitit launcher).

# ========================================================================
# CVAR25 TRAINING  (study: cvar25eff_vs_logp, 7 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 202: cvar25eff=76.267, logp=6975  << BEST cvar25eff
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/realnvp \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=cvar25_t202 \
#     algorithm.flow.activation=silu \
#     algorithm.flow.hidden_dim=768 \
#     algorithm.flow.n_flows=6 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0009957207406273527 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 423: cvar25eff=68, logp=4054.2
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/realnvp \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=cvar25_t423 \
#     algorithm.flow.activation=silu \
#     algorithm.flow.hidden_dim=768 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0009831735427421311 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 466: cvar25eff=51.8, logp=3218
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/realnvp \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=cvar25_t466 \
#     algorithm.flow.activation=silu \
#     algorithm.flow.hidden_dim=768 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0010076269052228957 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 480: cvar25eff=68.733, logp=4335.1
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=cifar10/realnvp \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=cvar25_t480 \
#     algorithm.flow.activation=silu \
#     algorithm.flow.hidden_dim=768 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0009486217340063567 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 510: cvar25eff=74.667, logp=4525.1  << KNEE
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/realnvp \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=cvar25_t510 \
#     algorithm.flow.activation=silu \
#     algorithm.flow.hidden_dim=768 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0010169276495106704 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 535: cvar25eff=61.867, logp=3373  << ORIGINAL PICK
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/realnvp \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=cvar25_t535 \
#     algorithm.flow.activation=silu \
#     algorithm.flow.hidden_dim=768 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0010115191329924005 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 561: cvar25eff=67.8, logp=3992.7
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/realnvp \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=cvar25_t561 \
#     algorithm.flow.activation=silu \
#     algorithm.flow.hidden_dim=768 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.001 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0009755799141771687 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ========================================================================
# CAP TRAINING  (study: cap_vs_logp, 18 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 169: cap=-0.35086, logp=3848.6
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=cap_t169 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=gelu \
#     algorithm.flow.hidden_dim=512 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=1 \
#     algorithm.flow.noise_scale=0.001 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.00012781505578566052 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 211: cap=-0.3528, logp=3504.1  << ORIGINAL PICK
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=cap_t211 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=gelu \
#     algorithm.flow.hidden_dim=512 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=1 \
#     algorithm.flow.noise_scale=0.001 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.00014437689046359488 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 302: cap=-0.35427, logp=3450.6
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=cap_t302 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=gelu \
#     algorithm.flow.hidden_dim=512 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=1 \
#     algorithm.flow.noise_scale=0.0001 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.00014097338628927283 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 305: cap=-0.3509, logp=3720.7
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=cap_t305 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=gelu \
#     algorithm.flow.hidden_dim=512 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=1 \
#     algorithm.flow.noise_scale=0.0001 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.00014190563708563 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 358: cap=-0.35151, logp=3516.7
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=cap_t358 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=gelu \
#     algorithm.flow.hidden_dim=512 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=1 \
#     algorithm.flow.noise_scale=0.0001 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0001344076101061297 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 381: cap=-0.35421, logp=3473.5
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=cap_t381 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=gelu \
#     algorithm.flow.hidden_dim=512 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=1 \
#     algorithm.flow.noise_scale=0.0001 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.00014127017142942245 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 389: cap=-0.34534, logp=4494.5  << KNEE
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=cap_t389 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=silu \
#     algorithm.flow.hidden_dim=512 \
#     algorithm.flow.n_flows=4 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.0001 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.00012810125443066736 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 398: cap=-0.34709, logp=4469.6
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=cap_t398 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=silu \
#     algorithm.flow.hidden_dim=512 \
#     algorithm.flow.n_flows=4 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.0001 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.00012783016311333813 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 416: cap=-0.34698, logp=4474.6
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=cap_t416 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=silu \
#     algorithm.flow.hidden_dim=512 \
#     algorithm.flow.n_flows=4 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.0001 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0001401391476537164 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 437: cap=-0.34777, logp=4303.9
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=cap_t437 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=silu \
#     algorithm.flow.hidden_dim=512 \
#     algorithm.flow.n_flows=6 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.0001 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.00013328038853833632 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 438: cap=-0.34769, logp=4315.8
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=cap_t438 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=silu \
#     algorithm.flow.hidden_dim=512 \
#     algorithm.flow.n_flows=6 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.0001 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.00013432885429002497 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 439: cap=-0.34982, logp=4214.1
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=cap_t439 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=silu \
#     algorithm.flow.hidden_dim=512 \
#     algorithm.flow.n_flows=6 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.0001 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.00013415866406839616 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 444: cap=-0.34855, logp=4236.6
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=cap_t444 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=silu \
#     algorithm.flow.hidden_dim=512 \
#     algorithm.flow.n_flows=6 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.0001 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.00013750971502539196 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 463: cap=-0.34776, logp=4311.6
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=cap_t463 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=silu \
#     algorithm.flow.hidden_dim=512 \
#     algorithm.flow.n_flows=6 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.0001 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.00013300492392898043 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 519: cap=-0.3505, logp=4202.1
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=cap_t519 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=silu \
#     algorithm.flow.hidden_dim=512 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.0001 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.00014488715706485747 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 573: cap=-0.33858, logp=12230  << BEST cap
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=cap_t573 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=relu \
#     algorithm.flow.hidden_dim=384 \
#     algorithm.flow.n_flows=4 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.0001 \
#     algorithm.flow.scale_clamp=5.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0014905095433944712 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 577: cap=-0.34375, logp=9884.6
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=cap_t577 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=relu \
#     algorithm.flow.hidden_dim=384 \
#     algorithm.flow.n_flows=4 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=5.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.001317682507464128 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 584: cap=-0.34157, logp=12199
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=cap_t584 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=relu \
#     algorithm.flow.hidden_dim=384 \
#     algorithm.flow.n_flows=4 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=5.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0023188763520046642 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ========================================================================
# CONSISTENCY TRAINING  (study: consistency_vs_logp, 28 Pareto points, trimmed to 11 around the knee (endpoints kept))
# ========================================================================
# ------------------------------------------------------------------------
# trial 29: consistency=-0.00032771, logp=1.2701e+05
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=consistency_t29 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=silu \
#     algorithm.flow.hidden_dim=384 \
#     algorithm.flow.n_flows=4 \
#     algorithm.flow.n_hidden_layers=1 \
#     algorithm.flow.noise_scale=0.0 \
#     algorithm.flow.scale_clamp=5.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0010495442683117516 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 48: consistency=-0.0005224, logp=25900
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=consistency_t48 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=relu \
#     algorithm.flow.hidden_dim=256 \
#     algorithm.flow.n_flows=6 \
#     algorithm.flow.n_hidden_layers=1 \
#     algorithm.flow.noise_scale=0.001 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0024214203937496146 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 167: consistency=-1.4232e-05, logp=3.7192e+08
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=consistency_t167 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=silu \
#     algorithm.flow.hidden_dim=256 \
#     algorithm.flow.n_flows=6 \
#     algorithm.flow.n_hidden_layers=1 \
#     algorithm.flow.noise_scale=0.001 \
#     algorithm.flow.scale_clamp=5.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=3.802293794485023e-05 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 185: consistency=-1.5774e-05, logp=9.6855e+07
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=consistency_t185 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=silu \
#     algorithm.flow.hidden_dim=256 \
#     algorithm.flow.n_flows=6 \
#     algorithm.flow.n_hidden_layers=1 \
#     algorithm.flow.noise_scale=0.001 \
#     algorithm.flow.scale_clamp=5.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=4.454849449543037e-05 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 246: consistency=0, logp=3.0943e+18  << BEST consistency
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=consistency_t246 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=relu \
#     algorithm.flow.hidden_dim=512 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=1 \
#     algorithm.flow.noise_scale=0.0 \
#     algorithm.flow.scale_clamp=5.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.00014968516727624284 \
#     algorithm.optimizer.weight_decay=0.0 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 254: consistency=-5.1155e-05, logp=1.0905e+07
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=consistency_t254 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=relu \
#     algorithm.flow.hidden_dim=512 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=1 \
#     algorithm.flow.noise_scale=0.0 \
#     algorithm.flow.scale_clamp=5.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=9.761402715276775e-05 \
#     algorithm.optimizer.weight_decay=0.0 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 289: consistency=-2.8113e-05, logp=1.7349e+07
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=consistency_t289 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=relu \
#     algorithm.flow.hidden_dim=384 \
#     algorithm.flow.n_flows=6 \
#     algorithm.flow.n_hidden_layers=1 \
#     algorithm.flow.noise_scale=0.001 \
#     algorithm.flow.scale_clamp=5.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.00011050081317738501 \
#     algorithm.optimizer.weight_decay=0.0 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 425: consistency=-0.050949, logp=3462.5
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=consistency_t425 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=gelu \
#     algorithm.flow.hidden_dim=512 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=1 \
#     algorithm.flow.noise_scale=0.0 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.00015516279090188026 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 453: consistency=-8.8994e-05, logp=2.5418e+06
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=consistency_t453 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=silu \
#     algorithm.flow.hidden_dim=768 \
#     algorithm.flow.n_flows=6 \
#     algorithm.flow.n_hidden_layers=1 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=5.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.00014558646270757133 \
#     algorithm.optimizer.weight_decay=0.0 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 509: consistency=-0.00053939, logp=7116.4
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=consistency_t509 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=relu \
#     algorithm.flow.hidden_dim=768 \
#     algorithm.flow.n_flows=6 \
#     algorithm.flow.n_hidden_layers=1 \
#     algorithm.flow.noise_scale=0.0 \
#     algorithm.flow.scale_clamp=5.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=4.2895150183902615e-05 \
#     algorithm.optimizer.weight_decay=0.0 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 534: consistency=-3.0745e-06, logp=1.7589e+09  << KNEE
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=consistency_t534 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=silu \
#     algorithm.flow.hidden_dim=256 \
#     algorithm.flow.n_flows=6 \
#     algorithm.flow.n_hidden_layers=1 \
#     algorithm.flow.noise_scale=0.001 \
#     algorithm.flow.scale_clamp=5.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=3.9205317049335124e-05 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ========================================================================
# STABILITY TRAINING  (study: drift_vs_logp, 2 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 242: drift=0.15415, logp=3457.6  << BEST drift
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=stability_t242 \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=silu \
#     algorithm.flow.hidden_dim=512 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.001 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0008697333713023401 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 527: drift=0.18232, logp=3268.7
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=stability_t527 \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=gelu \
#     algorithm.flow.hidden_dim=384 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.001 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0010153354474437143 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 104: drift=0.15415, logp=3534.6  << ORIGINAL PICK (handpicked; no longer on the current front)
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=stability_t104 \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=silu \
#     algorithm.flow.hidden_dim=768 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.001 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.000359807400273336 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ========================================================================
# WASSERSTEIN TRAINING  (study: wasserstein_vs_logp, 31 Pareto points, trimmed to 12 around the knee (endpoints kept))
# ========================================================================
# ------------------------------------------------------------------------
# trial 101: wasserstein=0.0074722, logp=5887  << BEST wasserstein
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=wasserstein_t101 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=relu \
#     algorithm.flow.hidden_dim=512 \
#     algorithm.flow.n_flows=6 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.0 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0006230155504472641 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 187: wasserstein=0.10685, logp=3253.7
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=wasserstein_t187 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=silu \
#     algorithm.flow.hidden_dim=768 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.0 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.000846217241604385 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 209: wasserstein=0.22164, logp=3055.4
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=wasserstein_t209 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=silu \
#     algorithm.flow.hidden_dim=768 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.0 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0008277544011632656 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 210: wasserstein=0.063397, logp=3565.2
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=wasserstein_t210 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=silu \
#     algorithm.flow.hidden_dim=768 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.0 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0009753777451150901 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 239: wasserstein=0.093117, logp=3409.3
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=wasserstein_t239 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=silu \
#     algorithm.flow.hidden_dim=768 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.0 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0008561489639827241 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 294: wasserstein=0.049734, logp=3775.9
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=wasserstein_t294 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=silu \
#     algorithm.flow.hidden_dim=768 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.0 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0008612525601649323 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 340: wasserstein=0.057017, logp=3759.4
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=wasserstein_t340 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=silu \
#     algorithm.flow.hidden_dim=768 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.0 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0007863662145101404 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 422: wasserstein=0.079743, logp=3413.4
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=wasserstein_t422 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=silu \
#     algorithm.flow.hidden_dim=768 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.0 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0008366537557578659 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 475: wasserstein=0.073541, logp=3448.6  << ORIGINAL PICK
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=wasserstein_t475 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=silu \
#     algorithm.flow.hidden_dim=512 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.0 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0008371472615602258 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 509: wasserstein=0.13574, logp=3226.8
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=wasserstein_t509 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=silu \
#     algorithm.flow.hidden_dim=768 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.0 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.000854181527881533 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 510: wasserstein=0.064409, logp=3508.9
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=wasserstein_t510 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=gelu \
#     algorithm.flow.hidden_dim=768 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.01 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0008256984464465109 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 564: wasserstein=0.067036, logp=3472.8  << KNEE
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=cifar10/realnvp_agnostic \
#     experiment_name=cifar10_realnvp_pareto \
#     run_name=wasserstein_t564 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.flow.activation=silu \
#     algorithm.flow.hidden_dim=768 \
#     algorithm.flow.n_flows=8 \
#     algorithm.flow.n_hidden_layers=2 \
#     algorithm.flow.noise_scale=0.0 \
#     algorithm.flow.scale_clamp=3.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0008574028791463721 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ========================================================================
# SUBMIT EVERYTHING ABOVE TO CLARIDEN  (one slurm job per command)
# ========================================================================
# This domain's data is downloaded rather than read from a raw ntuple path,
# so the blocks above carry no paths.raw_data_dir and none may be passed:
# submit_pareto.sh aborts on the '/path/to/...' placeholder. Any other hydra
# overrides appended here are added to every job.
# bash scripts/cluster/submit_pareto.sh scripts/cifar10/runrealnvp_pareto.sh
