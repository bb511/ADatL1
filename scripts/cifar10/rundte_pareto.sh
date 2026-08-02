#!/usr/bin/env bash
# ========================================================================
# DTE PARETO-FRONT TRAINING COMMANDS
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
# CVAR25 TRAINING  (study: cvar25eff_vs_ascore, 16 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 121: cvar25eff=14.4, ascore=1.7664e-07
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/dte \
#     experiment_name=cifar10_dte_pareto \
#     run_name=cvar25_t121 \
#     algorithm.beta_end=0.05464750926403297 \
#     algorithm.n_bins=5 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0002284291248193068 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=True \
#     algorithm.predictor.dropout=0.5 \
#     algorithm.predictor.nodes='[32,128,256]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 196: cvar25eff=5.8667, ascore=2.0897e-11
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/dte \
#     experiment_name=cifar10_dte_pareto \
#     run_name=cvar25_t196 \
#     algorithm.beta_end=0.08612367883670644 \
#     algorithm.n_bins=5 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.00025147222246235884 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=True \
#     algorithm.predictor.dropout=0.5 \
#     algorithm.predictor.nodes='[32,128,256]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 197: cvar25eff=8.8667, ascore=6.8493e-11
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/dte \
#     experiment_name=cifar10_dte_pareto \
#     run_name=cvar25_t197 \
#     algorithm.beta_end=0.08693761972112682 \
#     algorithm.n_bins=5 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.00025275690870003924 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=True \
#     algorithm.predictor.dropout=0.5 \
#     algorithm.predictor.nodes='[32,128,256]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 217: cvar25eff=8.9333, ascore=7.6224e-08
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=cifar10/dte \
#     experiment_name=cifar10_dte_pareto \
#     run_name=cvar25_t217 \
#     algorithm.beta_end=0.0840539699262339 \
#     algorithm.n_bins=5 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0004358066276307034 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=True \
#     algorithm.predictor.dropout=0.5 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 218: cvar25eff=9.5333, ascore=1.0674e-07
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/dte \
#     experiment_name=cifar10_dte_pareto \
#     run_name=cvar25_t218 \
#     algorithm.beta_end=0.085256867711439 \
#     algorithm.n_bins=5 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0004345316550707373 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=True \
#     algorithm.predictor.dropout=0.5 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 221: cvar25eff=9.6, ascore=1.6704e-07
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/dte \
#     experiment_name=cifar10_dte_pareto \
#     run_name=cvar25_t221 \
#     algorithm.beta_end=0.08590521152833921 \
#     algorithm.n_bins=5 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.00043264156462704254 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=True \
#     algorithm.predictor.dropout=0.5 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 310: cvar25eff=17.467, ascore=0.0024092
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/dte \
#     experiment_name=cifar10_dte_pareto \
#     run_name=cvar25_t310 \
#     algorithm.beta_end=0.09071016803493101 \
#     algorithm.n_bins=10 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0021502434877957136 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[16,32,128]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 317: cvar25eff=14.667, ascore=0.00026561
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=cifar10/dte \
#     experiment_name=cifar10_dte_pareto \
#     run_name=cvar25_t317 \
#     algorithm.beta_end=0.08633550088776897 \
#     algorithm.n_bins=10 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.002162711974020381 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[16,32,128]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 354: cvar25eff=27.267, ascore=0.029699
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/dte \
#     experiment_name=cifar10_dte_pareto \
#     run_name=cvar25_t354 \
#     algorithm.beta_end=0.09391635101511164 \
#     algorithm.n_bins=30 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0029973433712915266 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 359: cvar25eff=22.533, ascore=0.018451
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/dte \
#     experiment_name=cifar10_dte_pareto \
#     run_name=cvar25_t359 \
#     algorithm.beta_end=0.09382950364677173 \
#     algorithm.n_bins=30 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0029985450696703923 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 430: cvar25eff=987.13, ascore=0.16183  << KNEE
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/dte \
#     experiment_name=cifar10_dte_pareto \
#     run_name=cvar25_t430 \
#     algorithm.beta_end=0.09978493376201857 \
#     algorithm.n_bins=30 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0029966769245690725 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[32,64,256]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 445: cvar25eff=30.133, ascore=0.064638
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=cifar10/dte \
#     experiment_name=cifar10_dte_pareto \
#     run_name=cvar25_t445 \
#     algorithm.beta_end=0.0026940028565636477 \
#     algorithm.n_bins=30 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.002998532232181348 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[32,64,256]' \
#     algorithm.predictor.strides='[2,2]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 501: cvar25eff=1000, ascore=0.49813  << BEST cvar25eff
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/dte \
#     experiment_name=cifar10_dte_pareto \
#     run_name=cvar25_t501 \
#     algorithm.beta_end=0.001000344001949679 \
#     algorithm.n_bins=30 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.002786339140261317 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[32,64,256]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 507: cvar25eff=43.333, ascore=0.10856
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/dte \
#     experiment_name=cifar10_dte_pareto \
#     run_name=cvar25_t507 \
#     algorithm.beta_end=0.0010000272821727857 \
#     algorithm.n_bins=30 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.002998877282467283 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[32,64,256]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 526: cvar25eff=999.93, ascore=0.4981
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/dte \
#     experiment_name=cifar10_dte_pareto \
#     run_name=cvar25_t526 \
#     algorithm.beta_end=0.09996237929198748 \
#     algorithm.n_bins=30 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0029962230588317543 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[32,64,256]' \
#     algorithm.predictor.strides='[2,1]' \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 576: cvar25eff=999.8, ascore=0.49772
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=cifar10/dte \
#     experiment_name=cifar10_dte_pareto \
#     run_name=cvar25_t576 \
#     algorithm.beta_end=0.09345846828272289 \
#     algorithm.n_bins=30 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.002813540597429183 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.nodes='[32,64,256]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ========================================================================
# CAP TRAINING  (study: cap_vs_ascore, 29 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 178: cap=-0.66543, ascore=0.0041147
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/dte_agnostic \
#     experiment_name=cifar10_dte_pareto \
#     run_name=cap_t178 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.beta_end=0.0915033440426881 \
#     algorithm.n_bins=30.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0027377541301844667 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 194: cap=-0.69313, ascore=9.5395e-07
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/dte_agnostic \
#     experiment_name=cifar10_dte_pareto \
#     run_name=cap_t194 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.beta_end=0.06465030313662667 \
#     algorithm.n_bins=7.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0021670053352565635 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 215: cap=-0.69314, ascore=2.216e-08
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/dte_agnostic \
#     experiment_name=cifar10_dte_pareto \
#     run_name=cap_t215 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.beta_end=0.07596910577384303 \
#     algorithm.n_bins=7.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0019405622189657796 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 226: cap=-0.64302, ascore=0.0046641
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=cifar10/dte_agnostic \
#     experiment_name=cifar10_dte_pareto \
#     run_name=cap_t226 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.beta_end=0.09994573780899237 \
#     algorithm.n_bins=30.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.002998757860669344 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 273: cap=-0.66884, ascore=0.0039652
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/dte_agnostic \
#     experiment_name=cifar10_dte_pareto \
#     run_name=cap_t273 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.beta_end=0.09020324564263046 \
#     algorithm.n_bins=20.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.002631144466976538 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 282: cap=-0.35975, ascore=0.026384
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/dte_agnostic \
#     experiment_name=cifar10_dte_pareto \
#     run_name=cap_t282 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.beta_end=0.09987408865179606 \
#     algorithm.n_bins=30.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0029955731301279844 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 297: cap=-0.41803, ascore=0.018229
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/dte_agnostic \
#     experiment_name=cifar10_dte_pareto \
#     run_name=cap_t297 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.beta_end=0.09980195702378568 \
#     algorithm.n_bins=30.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0029916587765944722 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 370: cap=-0.67449, ascore=2.1469e-06
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=cifar10/dte_agnostic \
#     experiment_name=cifar10_dte_pareto \
#     run_name=cap_t370 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.beta_end=0.0025300786990167176 \
#     algorithm.n_bins=5.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0029986078762238096 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=True \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[32,64,128]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 418: cap=-0.33506, ascore=0.050043
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/dte_agnostic \
#     experiment_name=cifar10_dte_pareto \
#     run_name=cap_t418 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.beta_end=0.09278467683826877 \
#     algorithm.n_bins=20.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0027724094053979266 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[32,64,256]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 432: cap=-0.31836, ascore=0.050159  << BEST cap
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/dte_agnostic \
#     experiment_name=cifar10_dte_pareto \
#     run_name=cap_t432 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.beta_end=0.09998867755318785 \
#     algorithm.n_bins=20.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0029991811880369953 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 476: cap=-0.47046, ascore=0.013656
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/dte_agnostic \
#     experiment_name=cifar10_dte_pareto \
#     run_name=cap_t476 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.beta_end=0.08732482959287864 \
#     algorithm.n_bins=30.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0028050041010552447 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 496: cap=-0.57196, ascore=0.0090653
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=cifar10/dte_agnostic \
#     experiment_name=cifar10_dte_pareto \
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
#     algorithm.beta_end=0.09293226573168839 \
#     algorithm.n_bins=30.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0029996290674869615 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 501: cap=-0.65069, ascore=0.0042954
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/dte_agnostic \
#     experiment_name=cifar10_dte_pareto \
#     run_name=cap_t501 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.beta_end=0.09980988420291814 \
#     algorithm.n_bins=30.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.002781670893508758 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 502: cap=-0.61504, ascore=0.0075389
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/dte_agnostic \
#     experiment_name=cifar10_dte_pareto \
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
#     algorithm.beta_end=0.09992550318264376 \
#     algorithm.n_bins=30.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.002784861116348649 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 520: cap=-0.54735, ascore=0.0093336
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/dte_agnostic \
#     experiment_name=cifar10_dte_pareto \
#     run_name=cap_t520 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.beta_end=0.09377318333888729 \
#     algorithm.n_bins=30.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0027983805800081392 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 524: cap=-0.69315, ascore=7.3433e-20
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=cifar10/dte_agnostic \
#     experiment_name=cifar10_dte_pareto \
#     run_name=cap_t524 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.beta_end=0.09995064450535769 \
#     algorithm.n_bins=5.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0029969418374878524 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[32,64,128]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 526: cap=-0.69315, ascore=6.3659e-21
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/dte_agnostic \
#     experiment_name=cifar10_dte_pareto \
#     run_name=cap_t526 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.beta_end=0.09995346257373827 \
#     algorithm.n_bins=5.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0029966375056162966 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[32,64,128]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 533: cap=-0.59127, ascore=0.0077004
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/dte_agnostic \
#     experiment_name=cifar10_dte_pareto \
#     run_name=cap_t533 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.beta_end=0.09353555365011323 \
#     algorithm.n_bins=30.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.002797265616022036 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 538: cap=-0.63117, ascore=0.0048136
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/dte_agnostic \
#     experiment_name=cifar10_dte_pareto \
#     run_name=cap_t538 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.beta_end=0.09992763941832641 \
#     algorithm.n_bins=30.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0029972244273250783 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[16,32,64]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 559: cap=-0.34116, ascore=0.041393
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=cifar10/dte_agnostic \
#     experiment_name=cifar10_dte_pareto \
#     run_name=cap_t559 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.beta_end=0.09991385165946448 \
#     algorithm.n_bins=30.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.002797613671968827 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[32,64,128]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 561: cap=-0.49042, ascore=0.01006
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/dte_agnostic \
#     experiment_name=cifar10_dte_pareto \
#     run_name=cap_t561 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.beta_end=0.09998702198971045 \
#     algorithm.n_bins=30.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0027976183368596677 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[32,64,128]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 562: cap=-0.57464, ascore=0.0083236
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/dte_agnostic \
#     experiment_name=cifar10_dte_pareto \
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
#     algorithm.beta_end=0.09983998038311813 \
#     algorithm.n_bins=30.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.002797763804436625 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[32,64,128]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 563: cap=-0.48645, ascore=0.012595
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/dte_agnostic \
#     experiment_name=cifar10_dte_pareto \
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
#     algorithm.beta_end=0.0999976533074254 \
#     algorithm.n_bins=30.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0027958225026170655 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[32,64,128]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 565: cap=-0.37232, ascore=0.025803
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=cifar10/dte_agnostic \
#     experiment_name=cifar10_dte_pareto \
#     run_name=cap_t565 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.beta_end=0.09999886414567803 \
#     algorithm.n_bins=30.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0027196906030763444 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[32,64,128]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 566: cap=-0.37476, ascore=0.019354  << KNEE
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/dte_agnostic \
#     experiment_name=cifar10_dte_pareto \
#     run_name=cap_t566 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.beta_end=0.09998784513511914 \
#     algorithm.n_bins=30.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.002712290195930974 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[32,64,128]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 569: cap=-0.46419, ascore=0.014012
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/dte_agnostic \
#     experiment_name=cifar10_dte_pareto \
#     run_name=cap_t569 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.beta_end=0.09951122936033542 \
#     algorithm.n_bins=30.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0027137767558657816 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[32,64,128]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 570: cap=-0.35171, ascore=0.030678
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/dte_agnostic \
#     experiment_name=cifar10_dte_pareto \
#     run_name=cap_t570 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.beta_end=0.09995120788673743 \
#     algorithm.n_bins=30.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0025349440794683624 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[32,64,128]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 573: cap=-0.62773, ascore=0.0048903
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=cifar10/dte_agnostic \
#     experiment_name=cifar10_dte_pareto \
#     run_name=cap_t573 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.beta_end=0.09993638177248766 \
#     algorithm.n_bins=30.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.002530384690045032 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[32,64,128]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 604: cap=-0.53378, ascore=0.0093488
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/dte_agnostic \
#     experiment_name=cifar10_dte_pareto \
#     run_name=cap_t604 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.beta_end=0.09999204541995768 \
#     algorithm.n_bins=30.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0025781709237656333 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=False \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[32,64,128]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ========================================================================
# CONSISTENCY TRAINING  (study: consistency_vs_ascore, 1 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 535: consistency=0, ascore=2.1055e-17  << BEST consistency
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/dte_agnostic \
#     experiment_name=cifar10_dte_pareto \
#     run_name=consistency_t535 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     algorithm.beta_end=0.08742641438294169 \
#     algorithm.n_bins=5 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0023516602268996434 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=True \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[16,64,128]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ========================================================================
# STABILITY TRAINING  (study: drift_vs_ascore, 1 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 373: drift=0.15415, ascore=5.2057e-28  << BEST drift
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/dte_agnostic \
#     experiment_name=cifar10_dte_pareto \
#     run_name=stability_t373 \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.beta_end=0.09991451372648712 \
#     algorithm.n_bins=5 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.00218914441420134 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=True \
#     algorithm.predictor.dropout=0.5 \
#     algorithm.predictor.nodes='[64,128,256]' \
#     algorithm.predictor.strides='[1,2]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ========================================================================
# WASSERSTEIN TRAINING  (study: wasserstein_vs_ascore, 3 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 482: wasserstein=2.0785e-10, ascore=1.2043e-15
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=cifar10/dte_agnostic \
#     experiment_name=cifar10_dte_pareto \
#     run_name=wasserstein_t482 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.beta_end=0.08303407929596801 \
#     algorithm.n_bins=5.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.00262379400005636 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=True \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[32,64,128]' \
#     algorithm.predictor.strides='[2,1]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 548: wasserstein=1.5612e-10, ascore=1.2495e-14  << KNEE
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=cifar10/dte_agnostic \
#     experiment_name=cifar10_dte_pareto \
#     run_name=wasserstein_t548 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.beta_end=0.0861592010726732 \
#     algorithm.n_bins=5.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0027850333668297924 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=True \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[32,64,128]' \
#     algorithm.predictor.strides='[2,1]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 580: wasserstein=2.674e-11, ascore=6.0108e-14  << BEST wasserstein
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=cifar10/dte_agnostic \
#     experiment_name=cifar10_dte_pareto \
#     run_name=wasserstein_t580 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_reference_normal \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_reference_normal \
#     algorithm.beta_end=0.07991274169579592 \
#     algorithm.n_bins=5.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.002562648422160739 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.batchnorm=True \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.nodes='[32,64,128]' \
#     algorithm.predictor.strides='[2,1]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ========================================================================
# SUBMIT EVERYTHING ABOVE TO CLARIDEN  (one slurm job per command)
# ========================================================================
# This domain's data is downloaded rather than read from a raw ntuple path,
# so the blocks above carry no paths.raw_data_dir and none may be passed:
# submit_pareto.sh aborts on the '/path/to/...' placeholder. Any other hydra
# overrides appended here are added to every job.
# bash scripts/cluster/submit_pareto.sh scripts/cifar10/rundte_pareto.sh
