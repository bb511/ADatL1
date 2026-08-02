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
# trial 504: cvar25eff=1000, dist=0.00010483  << BEST cvar25eff
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/svdd \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cvar25_t504 \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=0.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.000954875333296137 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ========================================================================
# CAP TRAINING  (study: cap_vs_dist, 25 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 143: cap=-0.7371, dist=0.00016883
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t143 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-05 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0008933147576047501 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 181: cap=-0.71104, dist=0.00022244
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t181 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0008715972234716673 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 184: cap=-0.69646, dist=0.0063097
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t184 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0008675305083617928 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 186: cap=-0.70858, dist=0.00030801
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t186 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009975353100871247 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 188: cap=-0.70768, dist=0.00039871
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t188 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009918688940934667 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 192: cap=-0.70914, dist=0.00027162
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t192 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.000999863043063352 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 193: cap=-0.70767, dist=0.00055497
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t193 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.000997766616209934 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 194: cap=-0.70735, dist=0.0005658
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t194 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009987872528446273 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 206: cap=-0.70581, dist=0.00090382
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
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
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009995480562602053 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 238: cap=-0.70808, dist=0.00031605  << KNEE
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t238 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009458645243551287 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 256: cap=-0.70896, dist=0.00027286
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t256 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009543605145395421 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 300: cap=-0.70892, dist=0.00030651
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t300 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009437854097068876 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 303: cap=-0.70636, dist=0.00078259
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
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
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009461630714232329 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 359: cap=-0.70428, dist=0.00109
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t359 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009988631564632367 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 370: cap=-0.74407, dist=0.00013249
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t370 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.000942266592491852 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 371: cap=-0.74509, dist=0.00012196
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t371 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009428349949769751 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 375: cap=-0.66659, dist=0.011476  << BEST cap
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t375 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009380663091333337 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 415: cap=-0.69985, dist=0.0040498
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t415 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009127557591569211 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 423: cap=-0.74493, dist=0.00012709
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t423 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009160869543847125 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 464: cap=-0.70161, dist=0.002521
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t464 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=0.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.000952282300250192 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 511: cap=-0.74599, dist=0.00012039
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t511 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0008975557666056532 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 540: cap=-0.69772, dist=0.0052545
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t540 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.000908865634517757 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 546: cap=-0.73681, dist=0.00018522  << ORIGINAL PICK
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t546 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-07 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0009571264792363442 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 550: cap=-0.7365, dist=0.000188
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t550 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[16,32,64]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-07 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0009556037613343593 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 583: cap=-0.71098, dist=0.00026758
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=cap_t583 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[32,64,128]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-06 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009381743013465871 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

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
# trial 305: drift=0.16551, dist=0.00010297  << BEST drift
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=stability_t305 \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-08 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009147688701524873 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ========================================================================
# WASSERSTEIN TRAINING  (study: wasserstein_vs_dist, 16 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 402: wasserstein=0.019827, dist=0.00010689
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=wasserstein_t402 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-06 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009984106857605028 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 408: wasserstein=0.016194, dist=0.0014728  << BEST wasserstein
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=wasserstein_t408 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-06 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009995868644635997 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 412: wasserstein=0.019393, dist=0.00010695
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=wasserstein_t412 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-06 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009999639157469203 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 429: wasserstein=0.021142, dist=0.00010172
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=wasserstein_t429 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-07 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009476473911894325 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 441: wasserstein=0.019366, dist=0.00011006
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=wasserstein_t441 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-07 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.000999373012155083 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 477: wasserstein=0.016891, dist=0.00016
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=wasserstein_t477 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009446902695083377 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 518: wasserstein=0.017884, dist=0.00015206
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=wasserstein_t518 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009278224302659119 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 523: wasserstein=0.016676, dist=0.000398
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=wasserstein_t523 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009297976040354228 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 524: wasserstein=0.018076, dist=0.00013872
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=wasserstein_t524 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009306814897377092 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 553: wasserstein=0.018392, dist=0.00012867
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=wasserstein_t553 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009123298771913037 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 554: wasserstein=0.018806, dist=0.00012831
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=wasserstein_t554 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009131113704215903 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 556: wasserstein=0.019259, dist=0.00011217
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=wasserstein_t556 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009117957513465599 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 578: wasserstein=0.017647, dist=0.00015926
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=wasserstein_t578 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009200244031977769 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 579: wasserstein=0.018357, dist=0.00013586
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=wasserstein_t579 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.000920683037470414 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 583: wasserstein=0.016829, dist=0.00017721  << KNEE
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=wasserstein_t583 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009204598289099297 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 587: wasserstein=0.016428, dist=0.0010447
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     experiment=robustad/svdd_agnostic \
#     experiment_name=robustad_svdd_pareto \
#     run_name=wasserstein_t587 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_shifted_normal_all \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_shifted_normal_all \
#     algorithm.encoder.nodes='[8,16,32]' \
#     algorithm.encoder.strides='[2,2]' \
#     algorithm.network_weight_decay=1e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0009203556386300428 \
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
# bash scripts/cluster/submit_pareto.sh scripts/robustad/runsvdd_pareto.sh
