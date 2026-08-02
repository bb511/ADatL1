#!/usr/bin/env bash
# ========================================================================
# DTE PARETO-FRONT TRAINING COMMANDS
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
# CVAR25 TRAINING  (study: cvar25eff_vs_ascore, 4 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 528: cvar25eff=1000, ascore=1.2944e-05  << BEST cvar25eff
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/dte \
#     experiment_name=robustad_dte_pareto \
#     run_name=cvar25_t528 \
#     algorithm.beta_end=0.0716295919324617 \
#     algorithm.n_bins=5 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.000698232575782558 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=True \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.nodes='[32,64,128]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 555: cvar25eff=905.56, ascore=5.0557e-06
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/dte \
#     experiment_name=robustad_dte_pareto \
#     run_name=cvar25_t555 \
#     algorithm.beta_end=0.07160760681155981 \
#     algorithm.n_bins=5 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0006917269369927018 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 584: cvar25eff=900, ascore=4.743e-06  << KNEE
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/dte \
#     experiment_name=robustad_dte_pareto \
#     run_name=cvar25_t584 \
#     algorithm.beta_end=0.07315538209949841 \
#     algorithm.n_bins=5 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0006897128032781165 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 598: cvar25eff=711.11, ascore=2.4732e-07
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/dte \
#     experiment_name=robustad_dte_pareto \
#     run_name=cvar25_t598 \
#     algorithm.beta_end=0.0674493170256094 \
#     algorithm.n_bins=5 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0006939102557826487 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.nodes='[32,64,128]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ========================================================================
# CAP TRAINING  (study: cap_vs_ascore, 53 Pareto points, trimmed to 12 around the knee (endpoints kept))
# ========================================================================
# ------------------------------------------------------------------------
# trial 123: cap=-0.3851, ascore=0.50005  << BEST cap
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=cap_t123 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.002468318673295561 \
#     algorithm.n_bins=30.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=3.0140856552381492e-05 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 204: cap=-0.50737, ascore=0.48294
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=cap_t204 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.06453747950091904 \
#     algorithm.n_bins=20.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=3.7857508558285734e-05 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.batchnorm=True \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[8,16,32]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 205: cap=-0.50764, ascore=0.48287
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=cap_t205 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.06469221980500255 \
#     algorithm.n_bins=20.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=3.8003883974146284e-05 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.batchnorm=True \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[8,16,32]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 206: cap=-0.5078, ascore=0.48283  << KNEE
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=cap_t206 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.06492103398287938 \
#     algorithm.n_bins=20.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=3.811222869408093e-05 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.batchnorm=True \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[8,16,32]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 207: cap=-0.50743, ascore=0.4829
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=cap_t207 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.06534148429313379 \
#     algorithm.n_bins=20.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=3.800727153755136e-05 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.batchnorm=True \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[8,16,32]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 208: cap=-0.50769, ascore=0.48284
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=cap_t208 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.06541130521573252 \
#     algorithm.n_bins=20.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=3.8135783230417595e-05 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.batchnorm=True \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[8,16,32]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 303: cap=-0.69407, ascore=2.8053e-07
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=cap_t303 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.0594927528940123 \
#     algorithm.n_bins=5.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0007815142162428424 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.nodes='[32,64,128]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 588: cap=-0.54878, ascore=0.33851
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=cap_t588 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.0014779833871523616 \
#     algorithm.n_bins=7.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.00026820070090454156 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 590: cap=-0.54809, ascore=0.33987
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=cap_t590 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.0014728225477603678 \
#     algorithm.n_bins=7.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0002681789662133718 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 591: cap=-0.54846, ascore=0.3391
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=cap_t591 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.00147658908747824 \
#     algorithm.n_bins=7.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.00026793191782302375 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 592: cap=-0.54817, ascore=0.33971
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=cap_t592 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.0014735638394942044 \
#     algorithm.n_bins=7.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0002681467797645876 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 593: cap=-0.54854, ascore=0.33901
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=cap_t593 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.0014760486843562947 \
#     algorithm.n_bins=7.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0002682109460827669 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ========================================================================
# CONSISTENCY TRAINING  (study: consistency_vs_ascore, 18 Pareto points, trimmed to 12 around the knee (endpoints kept))
# ========================================================================
# ------------------------------------------------------------------------
# trial 162: consistency=-0.089244, ascore=0.50174
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=consistency_t162 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.09949023012315432 \
#     algorithm.n_bins=20 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=8.334000988152023e-05 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.5 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 163: consistency=-0.089085, ascore=0.50174
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=consistency_t163 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.09922897256999601 \
#     algorithm.n_bins=20 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=8.370945478699874e-05 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.5 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 164: consistency=-0.088998, ascore=0.50175  << BEST consistency
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=consistency_t164 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.09981962087719913 \
#     algorithm.n_bins=20 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=8.385534532650996e-05 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.5 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 245: consistency=-0.1747, ascore=0.49435
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=consistency_t245 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.006480542627700202 \
#     algorithm.n_bins=20 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.00012810236720399565 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.batchnorm=True \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.nodes='[8,16,32]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 405: consistency=-0.2048, ascore=0.49397
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=consistency_t405 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.014268324602013932 \
#     algorithm.n_bins=20 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.00012129387431806213 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.batchnorm=True \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.nodes='[8,16,32]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 412: consistency=-0.2051, ascore=0.49392  << KNEE
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=consistency_t412 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.01422522982580057 \
#     algorithm.n_bins=20 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.00011836501339877941 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.batchnorm=True \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.nodes='[8,16,32]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 486: consistency=-0.30357, ascore=7.2104e-05
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=consistency_t486 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.09239010877781535 \
#     algorithm.n_bins=7 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0005779681808344695 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.batchnorm=True \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.nodes='[32,64,128]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 546: consistency=-0.21932, ascore=0.039986
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=consistency_t546 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.09998391369561856 \
#     algorithm.n_bins=20 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.00012581223500577555 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.batchnorm=True \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.nodes='[32,64,128]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 548: consistency=-0.21925, ascore=0.040073
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=consistency_t548 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.09987859668604988 \
#     algorithm.n_bins=20 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.00012571556643624552 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.batchnorm=True \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.nodes='[32,64,128]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 550: consistency=-0.21933, ascore=0.03997
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=consistency_t550 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.09994669446648995 \
#     algorithm.n_bins=20 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.00012585821351116883 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.batchnorm=True \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.nodes='[32,64,128]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 553: consistency=-0.22637, ascore=0.03232
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=consistency_t553 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.09987186211642242 \
#     algorithm.n_bins=20 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.00012419719467801954 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.batchnorm=True \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.nodes='[32,64,128]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 557: consistency=-0.22627, ascore=0.032332
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=consistency_t557 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.09988679909252014 \
#     algorithm.n_bins=20 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.000124137030912946 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.batchnorm=True \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.nodes='[32,64,128]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ========================================================================
# STABILITY TRAINING  (study: drift_vs_ascore, 1 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 593: drift=0.16551, ascore=3.9942e-11  << BEST drift
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=stability_t593 \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.09998486321923544 \
#     algorithm.n_bins=5 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0009999997467907992 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.nodes='[32,64,128]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ========================================================================
# WASSERSTEIN TRAINING  (study: wasserstein_vs_ascore, 21 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 10: wasserstein=0.00039721, ascore=0.49591
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=wasserstein_t10 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.004570767903593388 \
#     algorithm.n_bins=20 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.00010220608905631865 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.5 \
#     algorithm.predictor.nodes='[8,16,32]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 46: wasserstein=0.00039821, ascore=0.49539
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=wasserstein_t46 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.006184551123643087 \
#     algorithm.n_bins=20 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.00031329203070244396 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.nodes='[8,16,32]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 66: wasserstein=0.0002567, ascore=0.50775  << BEST wasserstein
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=wasserstein_t66 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.0023155182076266792 \
#     algorithm.n_bins=15 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=7.137783531378338e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 129: wasserstein=0.00026377, ascore=0.49667
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=wasserstein_t129 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.016827869317717806 \
#     algorithm.n_bins=7 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=3.1636310684473394e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 148: wasserstein=0.0004264, ascore=0.49129
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=wasserstein_t148 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.003013438133176057 \
#     algorithm.n_bins=30 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=7.740919707821988e-05 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=True \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 192: wasserstein=0.00044556, ascore=0.49117
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=wasserstein_t192 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.0037688138678291425 \
#     algorithm.n_bins=30 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=7.487560709776894e-05 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=True \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 193: wasserstein=0.00044671, ascore=0.49115
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=wasserstein_t193 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.003787956273718015 \
#     algorithm.n_bins=30 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=7.496655678878942e-05 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=True \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 194: wasserstein=0.00044569, ascore=0.49116
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=wasserstein_t194 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.0037906290709059 \
#     algorithm.n_bins=30 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=7.483512198319007e-05 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=True \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 195: wasserstein=0.0004455, ascore=0.49117
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=wasserstein_t195 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.003770917866101493 \
#     algorithm.n_bins=30 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=7.486127494130083e-05 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=True \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 196: wasserstein=0.00044396, ascore=0.49118
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=wasserstein_t196 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.003744843559630408 \
#     algorithm.n_bins=30 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=7.479298435657104e-05 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=True \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 197: wasserstein=0.00044993, ascore=0.49111
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=wasserstein_t197 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.00379845397041304 \
#     algorithm.n_bins=30 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=7.531014011939789e-05 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=True \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 231: wasserstein=0.014131, ascore=2.3896e-06
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=wasserstein_t231 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.09181362272235634 \
#     algorithm.n_bins=5 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0004387223814997855 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=True \
#     algorithm.predictor.dropout=0.5 \
#     algorithm.predictor.nodes='[32,64,128]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 232: wasserstein=0.014178, ascore=2.3855e-06
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=wasserstein_t232 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.0915993583412508 \
#     algorithm.n_bins=5 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0004391848422695364 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=True \
#     algorithm.predictor.dropout=0.5 \
#     algorithm.predictor.nodes='[32,64,128]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 346: wasserstein=0.0018267, ascore=6.9149e-05
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=wasserstein_t346 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.08716041005772042 \
#     algorithm.n_bins=5 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0008510591681790675 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.batchnorm=True \
#     algorithm.predictor.dropout=0.5 \
#     algorithm.predictor.nodes='[32,64,128]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 376: wasserstein=0.018374, ascore=1.4019e-08
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=wasserstein_t376 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.07858205531888922 \
#     algorithm.n_bins=5 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.000837935690481215 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.5 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 377: wasserstein=0.018337, ascore=1.443e-08
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=wasserstein_t377 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.07849461945304964 \
#     algorithm.n_bins=5 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0008376910659577614 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.5 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 417: wasserstein=0.0017585, ascore=9.1876e-05
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=wasserstein_t417 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.09038278195986552 \
#     algorithm.n_bins=5 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0007050402033420488 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.batchnorm=True \
#     algorithm.predictor.dropout=0.5 \
#     algorithm.predictor.nodes='[32,64,128]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 442: wasserstein=0.00084571, ascore=0.00016007  << KNEE
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=wasserstein_t442 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.09994238915978787 \
#     algorithm.n_bins=5 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0007041307634339394 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.batchnorm=True \
#     algorithm.predictor.dropout=0.5 \
#     algorithm.predictor.nodes='[32,64,128]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 451: wasserstein=0.016549, ascore=1.6603e-06
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=wasserstein_t451 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.09999463838472107 \
#     algorithm.n_bins=7 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0006642945068715392 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 486: wasserstein=0.0033431, ascore=6.6248e-06
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=wasserstein_t486 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.09983919210777133 \
#     algorithm.n_bins=5 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0007064685285709029 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=True \
#     algorithm.predictor.dropout=0.5 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 541: wasserstein=0.000877, ascore=0.00013057
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/dte_agnostic \
#     experiment_name=robustad_dte_pareto \
#     run_name=wasserstein_t541 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.beta_end=0.09989684733622459 \
#     algorithm.n_bins=5 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0006699640849268251 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.batchnorm=True \
#     algorithm.predictor.dropout=0.5 \
#     algorithm.predictor.nodes='[32,64,128]' \
#     algorithm.predictor.strides='[2,2]' \
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
# bash scripts/cluster/submit_pareto.sh scripts/robustad/rundte_pareto.sh
