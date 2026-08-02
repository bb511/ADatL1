#!/usr/bin/env bash
# ========================================================================
# SVDD PARETO-FRONT TRAINING COMMANDS
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
# CVAR25 TRAINING  (study: cvar25eff_vs_dist, 1 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 518: cvar25eff=1000, dist=6.3251e-05  << ORIGINAL PICK | BEST cvar25eff
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/svdd \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cvar25_t518 \
#     algorithm.center_init_method=mean \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.2 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0008931169751749622 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=1e-05 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ========================================================================
# CAP TRAINING  (study: cap_vs_dist, 26 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 220: cap=-0.72115, dist=0.00018649
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t220 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.1 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0009995761229175649 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.soft_boundary=False \
#     algorithm.weight_decay=1e-08 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 222: cap=-0.72237, dist=0.00016724
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t222 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.1 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0009955551099150152 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.soft_boundary=False \
#     algorithm.weight_decay=1e-08 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 227: cap=-0.72221, dist=0.00017457
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t227 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.05 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.000999365956058271 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.soft_boundary=False \
#     algorithm.weight_decay=1e-08 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 265: cap=-0.71611, dist=0.00019045
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t265 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.05 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.000999679348708815 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=1e-07 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 344: cap=-0.74106, dist=0.00011993
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t344 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     algorithm.center_init_method=mean \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.05 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.000999960363091811 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=1e-08 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 358: cap=-0.74119, dist=0.0001001
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t358 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     algorithm.center_init_method=mean \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.05 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0009424949076079646 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=1e-08 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 366: cap=-0.74157, dist=9.9424e-05
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t366 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     algorithm.center_init_method=mean \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.05 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0008779182094569178 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=1e-08 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 391: cap=-0.69405, dist=0.0030467
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t391 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     algorithm.center_init_method=mean \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.05 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0009410966842111044 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=1e-07 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 465: cap=-0.74017, dist=0.00015288
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t465 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     algorithm.center_init_method=mean \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.01 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009457302452553448 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=1e-07 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 490: cap=-0.69895, dist=0.0025104
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t490 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     algorithm.center_init_method=mean \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0008908476000488082 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=0.0 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 496: cap=-0.69489, dist=0.0026483
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t496 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     algorithm.center_init_method=mean \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.000948604434020951 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=0.0 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 525: cap=-0.70849, dist=0.00049658
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t525 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     algorithm.center_init_method=mean \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009525505170223769 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=0.0 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 538: cap=-0.70291, dist=0.0014079
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t538 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     algorithm.center_init_method=mean \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009533480508782476 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=0.0 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 539: cap=-0.71049, dist=0.00025746
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t539 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     algorithm.center_init_method=mean \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009537078274266963 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=0.0 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 543: cap=-0.71005, dist=0.00038515
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t543 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     algorithm.center_init_method=mean \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0008952477832509553 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=0.0 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 546: cap=-0.69141, dist=0.0052143  << ORIGINAL PICK | BEST cap
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t546 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     algorithm.center_init_method=mean \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009585626245761539 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=0.0 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 549: cap=-0.70435, dist=0.0011712
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t549 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     algorithm.center_init_method=mean \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009505968851670328 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=0.0 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 550: cap=-0.69939, dist=0.0020866
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t550 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     algorithm.center_init_method=mean \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009516354859217911 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=0.0 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 553: cap=-0.71286, dist=0.00020632
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t553 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     algorithm.center_init_method=mean \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009560418303424424 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=0.0 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 570: cap=-0.70803, dist=0.00056783
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t570 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     algorithm.center_init_method=mean \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009533955342615703 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=0.0 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 581: cap=-0.69936, dist=0.0024725
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t581 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     algorithm.center_init_method=mean \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009521065704804709 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=0.0 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 583: cap=-0.70357, dist=0.0013887
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t583 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     algorithm.center_init_method=mean \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009547425800153348 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=0.0 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 586: cap=-0.7122, dist=0.00021684
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t586 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     algorithm.center_init_method=mean \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009522371396542728 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=0.0 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 587: cap=-0.70967, dist=0.00046991
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t587 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     algorithm.center_init_method=mean \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009537999961874654 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=0.0 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 593: cap=-0.70518, dist=0.0010393
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t593 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     algorithm.center_init_method=mean \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009560766949692073 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=0.0 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 599: cap=-0.7061, dist=0.00057367  << KNEE
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t599 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     algorithm.center_init_method=mean \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.000949954213179476 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=0.0 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ========================================================================
# CONSISTENCY TRAINING  (study: consistency_vs_dist, 16 Pareto points, trimmed to 11 around the knee (endpoints kept))
# ========================================================================
# ------------------------------------------------------------------------
# trial 158: consistency=-0.79932, dist=0.0043557
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=consistency_t158 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=0.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.000887897647166254 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 220: consistency=-0.57772, dist=0.024435  << BEST consistency
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=consistency_t220 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=0.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0007906143539600401 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 345: consistency=-0.87181, dist=0.00046897  << KNEE
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=consistency_t345 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009982640975866187 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 347: consistency=-0.93522, dist=0.00036525
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=consistency_t347 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009994011468410068 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 379: consistency=-2.414, dist=0.00010662
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=consistency_t379 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=0.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0008619043827441943 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 518: consistency=-0.72897, dist=0.0085922
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=consistency_t518 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=0.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009199890407331748 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 519: consistency=-0.99923, dist=0.00020569
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=consistency_t519 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=0.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009202995401976893 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 532: consistency=-1.7765, dist=0.00018553
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=consistency_t532 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-06 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0009590353724841615 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 536: consistency=-1.8097, dist=0.00017549
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=consistency_t536 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-07 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.000951985916901306 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 544: consistency=-0.98023, dist=0.00024121
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=consistency_t544 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=0.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009426817409967698 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 575: consistency=-0.94942, dist=0.00030118
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=consistency_t575 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=0.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009991636334355626 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ========================================================================
# STABILITY TRAINING  (study: drift_vs_dist, 1 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 525: drift=0.16551, dist=7.0869e-05  << ORIGINAL PICK | BEST drift
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=stability_t525 \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.1 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0006495587687856368 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=0.0 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ========================================================================
# WASSERSTEIN TRAINING  (study: wasserstein_vs_dist, 14 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 236: wasserstein=0.0065399, dist=9.1514e-05  << KNEE
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=wasserstein_t236 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.1 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0005267397203039592 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=1e-07 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 267: wasserstein=0.0064749, dist=0.00012171
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=wasserstein_t267 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.1 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0005838066411827784 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=1e-07 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 297: wasserstein=0.0069254, dist=6.6856e-05
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=wasserstein_t297 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.1 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0006217491601128081 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=1e-07 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 307: wasserstein=0.0060349, dist=0.00027871
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=wasserstein_t307 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.1 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0006519187478717954 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=1e-07 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 377: wasserstein=0.0061945, dist=0.00026915
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=wasserstein_t377 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.1 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0006508355225077969 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=1e-07 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 386: wasserstein=0.0055496, dist=0.00046035  << BEST wasserstein
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=wasserstein_t386 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.1 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0005915538705440909 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=1e-07 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 414: wasserstein=0.0068413, dist=7.8557e-05
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=wasserstein_t414 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.1 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0006262104268807702 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=1e-07 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 447: wasserstein=0.0067264, dist=8.8788e-05
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=wasserstein_t447 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.1 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0006109532365044968 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=1e-07 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 480: wasserstein=0.0062807, dist=0.00019582
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=wasserstein_t480 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.1 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0006199731990980755 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=1e-07 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 483: wasserstein=0.0068413, dist=7.8552e-05
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=wasserstein_t483 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.1 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0006262209591177468 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=1e-07 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 513: wasserstein=0.0064326, dist=0.00016021
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=wasserstein_t513 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.1 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0006074287827437135 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=1e-07 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 524: wasserstein=0.0057214, dist=0.0003634
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=wasserstein_t524 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.1 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0005915704500204442 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=1e-07 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 547: wasserstein=0.00674, dist=8.0716e-05
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=wasserstein_t547 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.1 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0005897519048319808 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=1e-06 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 581: wasserstein=0.006826, dist=7.9576e-05  << ORIGINAL PICK
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=wasserstein_t581 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.center_init_method=zeros \
#     algorithm.encoder.activation=silu \
#     algorithm.encoder.batchnorm=False \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.nu=0.1 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0006079271679609457 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.soft_boundary=True \
#     algorithm.weight_decay=1e-07 \
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
# bash scripts/cluster/submit_pareto.sh scripts/robustad/runsvdd_pareto.sh
