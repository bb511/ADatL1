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
# CVAR25 TRAINING  (study: cvar25eff_vs_dist, 1 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 293: cvar25eff=1000, dist=1.0817e-07  << BEST cvar25eff
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/svdd \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cvar25_t293 \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[64,128,256]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.1 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.002671223582515294 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=1e-05 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 284: cvar25eff=5.2, dist=0.0003922  << ORIGINAL PICK (handpicked; no longer on the current front)
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/svdd \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cvar25_t284 \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[64,128,256]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.1 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.002347325794742755 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=1e-05 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ========================================================================
# CAP TRAINING  (study: cap_vs_dist, 19 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 27: cap=-0.44576, dist=0.00073663
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cap_t27 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0006449992334186136 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.soft_boundary=False \
#     algorithm.weight_decay=1e-08 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 96: cap=-0.50825, dist=2.873e-05
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cap_t96 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     algorithm.center_init_method=mean \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[16,64,128]' \
#     algorithm.encoder.strides='[2,1]' \
#     algorithm.nu=0.01 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0020244421689937157 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=1e-05 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 101: cap=-0.39661, dist=0.0020406
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cap_t101 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[1,2]' \
#     algorithm.nu=0.01 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0027433555830038694 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=1e-08 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 181: cap=-0.40167, dist=0.0019224
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cap_t181 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     algorithm.center_init_method=mean \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[16,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.01 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0007370432369984453 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.soft_boundary=False \
#     algorithm.weight_decay=1e-08 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 421: cap=-0.34978, dist=0.0051017
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cap_t421 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.batchnorm=True \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.2 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.000732168786010285 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.soft_boundary=False \
#     algorithm.weight_decay=1e-08 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 422: cap=-0.34538, dist=0.0085633
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cap_t422 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.batchnorm=True \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.2 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.000730962287511622 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.soft_boundary=False \
#     algorithm.weight_decay=1e-08 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 459: cap=-0.33181, dist=0.014407  << BEST cap
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cap_t459 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=True \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.2 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0007142790088755337 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.soft_boundary=False \
#     algorithm.weight_decay=1e-06 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 471: cap=-0.4224, dist=0.0011057
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cap_t471 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[2,1]' \
#     algorithm.nu=0.2 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0019777896305084733 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=1e-06 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 486: cap=-0.35858, dist=0.0038768
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cap_t486 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[2,1]' \
#     algorithm.nu=0.2 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.001938086961586347 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=1e-06 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 506: cap=-0.36526, dist=0.0038005
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cap_t506 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[2,1]' \
#     algorithm.nu=0.2 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0018175389455529643 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=1e-06 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 541: cap=-0.43585, dist=0.0010777
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cap_t541 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[2,1]' \
#     algorithm.nu=0.2 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0007069455974998409 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=1e-06 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 545: cap=-0.37554, dist=0.0020522  << KNEE
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cap_t545 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[2,1]' \
#     algorithm.nu=0.2 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0007059785726001196 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=1e-06 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 556: cap=-0.41418, dist=0.0017672
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cap_t556 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[2,1]' \
#     algorithm.nu=0.2 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0007519509083548563 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=1e-06 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 563: cap=-0.41912, dist=0.0012824
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cap_t563 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[2,1]' \
#     algorithm.nu=0.2 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0006971456024943326 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=1e-06 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 569: cap=-0.37544, dist=0.0030589
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cap_t569 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[2,1]' \
#     algorithm.nu=0.2 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0007002117990658676 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=1e-06 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 594: cap=-0.48407, dist=0.00051238
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cap_t594 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.01 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0006995395903081966 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.soft_boundary=False \
#     algorithm.weight_decay=1e-06 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 595: cap=-0.45012, dist=0.00068705
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cap_t595 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.01 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0006994468436498895 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.soft_boundary=False \
#     algorithm.weight_decay=1e-06 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 597: cap=-0.45115, dist=0.00057418
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cap_t597 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.01 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0007015656798134579 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.soft_boundary=False \
#     algorithm.weight_decay=1e-06 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 599: cap=-0.49221, dist=0.00048244  << ORIGINAL PICK
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=cap_t599 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.01 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0006990798503902553 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.soft_boundary=False \
#     algorithm.weight_decay=1e-06 \
#     trainer.gradient_clip_val=1.0 \
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
# trial 396: drift=0.15415, dist=2.7628e-06  << BEST drift
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=stability_t396 \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=True \
#     algorithm.encoder.nodes='[32,64,256]' \
#     algorithm.encoder.strides='[1,2]' \
#     algorithm.nu=0.01 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.002644941232195868 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=1e-08 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 191: drift=0.15415, dist=1.3221e-05  << ORIGINAL PICK (handpicked; no longer on the current front)
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=stability_t191 \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=True \
#     algorithm.encoder.nodes='[16,64,128]' \
#     algorithm.encoder.strides='[2,1]' \
#     algorithm.nu=0.01 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0029638593347981854 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=1e-08 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ========================================================================
# WASSERSTEIN TRAINING  (study: wasserstein_vs_dist, 6 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 260: wasserstein=2.2399e-08, dist=1.1859e-06  << KNEE
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=wasserstein_t260 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=True \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,1]' \
#     algorithm.nu=0.05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0026846353550811215 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=0.0 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 266: wasserstein=1.6368e-08, dist=1.3061e-06  << ORIGINAL PICK
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=wasserstein_t266 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=True \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,1]' \
#     algorithm.nu=0.05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0026915777942867008 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=0.0 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 324: wasserstein=2.6292e-07, dist=1.0561e-06
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=wasserstein_t324 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=True \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,1]' \
#     algorithm.nu=0.05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.002998484172250363 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=0.0 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 336: wasserstein=9.2356e-09, dist=1.584e-06  << BEST wasserstein
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=wasserstein_t336 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=True \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,1]' \
#     algorithm.nu=0.05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.002999232871602943 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=0.0 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 354: wasserstein=1.1998e-07, dist=1.0999e-06
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=wasserstein_t354 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=True \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,1]' \
#     algorithm.nu=0.05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.002614933776981812 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=0.0 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 433: wasserstein=9.9809e-08, dist=1.1456e-06
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/svdd_agnostic \
#     experiment_name=cifar10_svdd_pareto \
#     run_name=wasserstein_t433 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=True \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,1]' \
#     algorithm.nu=0.05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.002998149016072265 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=0.0 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ========================================================================
# SUBMIT EVERYTHING ABOVE TO CLARIDEN  (one slurm job per command)
# ========================================================================
# This domain's data is downloaded rather than read from a raw ntuple path,
# so the blocks above carry no paths.raw_data_dir and none may be passed:
# submit_pareto.sh aborts on the '/path/to/...' placeholder. Any other hydra
# overrides appended here are added to every job.
# bash scripts/cluster/submit_pareto.sh scripts/cifar10/runsvdd_pareto.sh
