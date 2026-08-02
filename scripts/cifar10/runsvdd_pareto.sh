#!/usr/bin/env bash
# ========================================================================
# SVDD PARETO-FRONT TRAINING COMMANDS
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
# CVAR25 TRAINING  (study: cvar25eff_vs_dist, 7 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 232: cvar25eff=32.867, dist=0.16272  << KNEE
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/svdd \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cvar25_t232 \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[1,2]' \
#     algorithm.network_weight_decay=1e-08 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.002675476707656654 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 249: cvar25eff=996.93, dist=2.6447  << BEST cvar25eff
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/svdd \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cvar25_t249 \
#     algorithm.encoder.nodes='[32,64,256]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-08 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0028231896714431648 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 316: cvar25eff=29.6, dist=0.15482
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/svdd \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cvar25_t316 \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[1,2]' \
#     algorithm.network_weight_decay=1e-08 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0027328054769430133 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 329: cvar25eff=38, dist=0.18572
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=cifar10/svdd \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cvar25_t329 \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[1,2]' \
#     algorithm.network_weight_decay=1e-08 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0027410854546448624 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 374: cvar25eff=16.467, dist=0.15255
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/svdd \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cvar25_t374 \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[1,2]' \
#     algorithm.network_weight_decay=1e-08 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0027822404736751885 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 585: cvar25eff=6.6667, dist=0.12371
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/svdd \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cvar25_t585 \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-08 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0026780532299204925 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 587: cvar25eff=6.8, dist=0.12457
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/svdd \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cvar25_t587 \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-08 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.002678387661012316 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 284: cvar25eff=9.8667, dist=0.17166  << ORIGINAL PICK (handpicked; no longer on the current front)
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=cifar10/svdd \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cvar25_t284 \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[1,2]' \
#     algorithm.network_weight_decay=1e-08 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=7.34142449582953e-05 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ========================================================================
# CAP TRAINING  (study: cap_vs_dist, 23 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 214: cap=-0.4432, dist=0.14698
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cap_t214 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[2,1]' \
#     algorithm.network_weight_decay=1e-06 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.001782098087663901 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 371: cap=-0.40507, dist=0.15364
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cap_t371 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[2,1]' \
#     algorithm.network_weight_decay=1e-06 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0016000347917200747 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 387: cap=-0.39894, dist=0.16672
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cap_t387 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[2,1]' \
#     algorithm.network_weight_decay=1e-06 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0017551584304010403 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 392: cap=-0.4276, dist=0.14922
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cap_t392 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[2,1]' \
#     algorithm.network_weight_decay=1e-06 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0018017137225526292 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 481: cap=-0.34399, dist=0.70768
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cap_t481 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[1,2]' \
#     algorithm.network_weight_decay=1e-06 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0016098556850999547 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 482: cap=-0.34778, dist=0.69253
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cap_t482 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[1,2]' \
#     algorithm.network_weight_decay=1e-06 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0016042051635007622 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 484: cap=-0.34409, dist=0.70686
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cap_t484 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[1,2]' \
#     algorithm.network_weight_decay=1e-06 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0016093087176729936 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 485: cap=-0.34325, dist=0.71023  << BEST cap
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cap_t485 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[1,2]' \
#     algorithm.network_weight_decay=1e-06 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.001611596670160256 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 487: cap=-0.36226, dist=0.64734
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cap_t487 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[1,2]' \
#     algorithm.network_weight_decay=1e-06 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0015525485452620564 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 489: cap=-0.36408, dist=0.64586
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cap_t489 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[1,2]' \
#     algorithm.network_weight_decay=1e-06 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0015519788206548545 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 492: cap=-0.35768, dist=0.65303
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cap_t492 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[1,2]' \
#     algorithm.network_weight_decay=1e-06 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0015553308506942173 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 493: cap=-0.35774, dist=0.65291
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cap_t493 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[1,2]' \
#     algorithm.network_weight_decay=1e-06 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0015551547806602057 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 494: cap=-0.35738, dist=0.65352
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cap_t494 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[1,2]' \
#     algorithm.network_weight_decay=1e-06 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.001555711235795169 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 496: cap=-0.35897, dist=0.65063
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cap_t496 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[1,2]' \
#     algorithm.network_weight_decay=1e-06 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0015537335864611536 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 497: cap=-0.35682, dist=0.65539
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cap_t497 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[1,2]' \
#     algorithm.network_weight_decay=1e-06 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.001557184940476275 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 498: cap=-0.35065, dist=0.66834
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cap_t498 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[1,2]' \
#     algorithm.network_weight_decay=1e-06 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.001574258776641337 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 499: cap=-0.3512, dist=0.66714
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cap_t499 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[1,2]' \
#     algorithm.network_weight_decay=1e-06 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.001572058652988268 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 500: cap=-0.35236, dist=0.6663
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cap_t500 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[1,2]' \
#     algorithm.network_weight_decay=1e-06 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.001569971340729096 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 502: cap=-0.3513, dist=0.66689
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cap_t502 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[1,2]' \
#     algorithm.network_weight_decay=1e-06 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0015715373865179932 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 503: cap=-0.35241, dist=0.66628
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cap_t503 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[1,2]' \
#     algorithm.network_weight_decay=1e-06 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0015700199976040406 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 510: cap=-0.38236, dist=0.17199
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cap_t510 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[2,1]' \
#     algorithm.network_weight_decay=1e-06 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.00162711317759456 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 562: cap=-0.36408, dist=0.22053  << KNEE
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cap_t562 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[2,1]' \
#     algorithm.network_weight_decay=1e-06 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.001601803862753566 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 563: cap=-0.37805, dist=0.1923
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cap_t563 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[2,1]' \
#     algorithm.network_weight_decay=1e-06 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.001598232012422273 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ========================================================================
# CONSISTENCY TRAINING  (study: consistency_vs_dist, 18 Pareto points, trimmed to 12 around the knee (endpoints kept))
# ========================================================================
# ------------------------------------------------------------------------
# trial 194: consistency=-0.0015591, dist=0.14684
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=consistency_t194 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-08 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0019227140631441985 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 196: consistency=-0.00074179, dist=0.15397
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=consistency_t196 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-08 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0019390796911555157 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 204: consistency=0, dist=2.8225  << BEST consistency
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=consistency_t204 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[64,128,256]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-08 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.002013554347945066 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 205: consistency=0, dist=2.8225
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=consistency_t205 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[64,128,256]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-08 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0020179502126558935 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 206: consistency=0, dist=2.8225
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=consistency_t206 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[64,128,256]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-08 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0020038054084601377 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 271: consistency=0, dist=2.8225
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=consistency_t271 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[64,128,256]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-08 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0024848665949166204 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 305: consistency=-0.0018958, dist=0.14067
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=consistency_t305 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-08 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0021256942154330655 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 482: consistency=-0.00058761, dist=0.2564
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=consistency_t482 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[1,2]' \
#     algorithm.network_weight_decay=1e-08 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.00032241502700420887 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 511: consistency=-0.00080393, dist=0.15174
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=consistency_t511 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-08 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0019603938458025096 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 565: consistency=-0.0006307, dist=0.15418  << KNEE
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=consistency_t565 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-08 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0020519973770592027 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 592: consistency=-0.0039522, dist=0.13443
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=consistency_t592 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-08 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.002186656311155761 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 595: consistency=-0.0015535, dist=0.15003
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=consistency_t595 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-08 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.002076037888830398 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ========================================================================
# STABILITY TRAINING  (study: drift_vs_dist, 1 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 391: drift=0.15415, dist=0.14016  << BEST drift
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=stability_t391 \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[2,1]' \
#     algorithm.network_weight_decay=1e-06 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0012689430772179818 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 191: drift=0.18232, dist=0.14754  << ORIGINAL PICK (handpicked; no longer on the current front)
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=stability_t191 \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[2,1]' \
#     algorithm.network_weight_decay=1e-07 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0006902334605550531 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ========================================================================
# WASSERSTEIN TRAINING  (study: wasserstein_vs_dist, 13 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 18: wasserstein=0, dist=2.8818  << BEST wasserstein
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=wasserstein_t18 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[64,128,256]' \
#     algorithm.encoder.strides='[1,2]' \
#     algorithm.network_weight_decay=1e-06 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.002798878646032515 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 42: wasserstein=0.0025217, dist=0.13845
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=wasserstein_t42 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[1,2]' \
#     algorithm.network_weight_decay=1e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.001367124621187054 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 57: wasserstein=0.0016304, dist=0.16193
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=wasserstein_t57 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[2,1]' \
#     algorithm.network_weight_decay=1e-07 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0004148387405854391 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 84: wasserstein=0, dist=2.8818
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=wasserstein_t84 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[64,128,256]' \
#     algorithm.encoder.strides='[1,2]' \
#     algorithm.network_weight_decay=1e-07 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0029363895946246184 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 170: wasserstein=0.00093119, dist=0.21153  << KNEE
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=wasserstein_t170 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[1,2]' \
#     algorithm.network_weight_decay=0.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.00022271088180090566 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 181: wasserstein=0.0010085, dist=0.20805
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=wasserstein_t181 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[1,2]' \
#     algorithm.network_weight_decay=0.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.00021523451732438645 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 183: wasserstein=0.0012078, dist=0.20479
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=wasserstein_t183 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[1,2]' \
#     algorithm.network_weight_decay=0.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.00021710607335699445 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 197: wasserstein=0.0010352, dist=0.20771
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=wasserstein_t197 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[1,2]' \
#     algorithm.network_weight_decay=0.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.00021343681279667313 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 223: wasserstein=0.0016208, dist=0.2025
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=wasserstein_t223 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[1,2]' \
#     algorithm.network_weight_decay=0.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.00015822804535744716 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 481: wasserstein=0.0019988, dist=0.15546
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=wasserstein_t481 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[1,2]' \
#     algorithm.network_weight_decay=0.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0022536388523644564 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 483: wasserstein=0.0023562, dist=0.14946
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=wasserstein_t483 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[1,2]' \
#     algorithm.network_weight_decay=0.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0022459274707370093 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 485: wasserstein=0.001731, dist=0.15649
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=wasserstein_t485 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[1,2]' \
#     algorithm.network_weight_decay=0.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.002258566406096357 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 542: wasserstein=0.0011566, dist=0.20571
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=wasserstein_t542 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[1,2]' \
#     algorithm.network_weight_decay=1e-08 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0002141258537059696 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ========================================================================
# SUBMIT EVERYTHING ABOVE TO CLARIDEN  (one slurm job per command)
# ========================================================================
# This domain's data is downloaded rather than read from a raw ntuple path,
# so the blocks above carry no paths.raw_data_dir and none may be passed:
# submit_pareto.sh aborts on the '/path/to/...' placeholder. Any other hydra
# overrides appended here are added to every job.
# bash scripts/cluster/submit_pareto.sh scripts/cifar10/runsvdd_pareto.sh
