#!/usr/bin/env bash
# ========================================================================
# DSAE PARETO-FRONT TRAINING COMMANDS
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
# CVAR25 TRAINING  (study: cvar25eff_vs_mse_b16k, 5 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 457: cvar25eff=1.795, mse=0.20323  << BEST cvar25eff
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae \
#     experiment_name=physics_dsae_pareto \
#     run_name=cvar25_t457 \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[16,8],jets:[16,8],muons:[8],taus:[16,8]}' \
#     algorithm.encoder.pooling=sum_max \
#     algorithm.encoder.rho_nodes='[48,16]' \
#     algorithm.input_noise_std=0.003 \
#     algorithm.delta=10.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.002081733354631208 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=5.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 458: cvar25eff=1.5452, mse=0.1775  << KNEE
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae \
#     experiment_name=physics_dsae_pareto \
#     run_name=cvar25_t458 \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[16,8],jets:[16,8],muons:[8],taus:[16,8]}' \
#     algorithm.encoder.pooling=sum_max \
#     algorithm.encoder.rho_nodes='[48,16]' \
#     algorithm.input_noise_std=0.003 \
#     algorithm.delta=10.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.002110844716319439 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=5.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 496: cvar25eff=1.7341, mse=0.19974
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae \
#     experiment_name=physics_dsae_pareto \
#     run_name=cvar25_t496 \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[16,8],jets:[16,8],muons:[8],taus:[16,8]}' \
#     algorithm.encoder.pooling=sum_max \
#     algorithm.encoder.rho_nodes='[48,16]' \
#     algorithm.input_noise_std=0.003 \
#     algorithm.delta=10.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.002055416302776611 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 591: cvar25eff=1.6991, mse=0.19708
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae \
#     experiment_name=physics_dsae_pareto \
#     run_name=cvar25_t591 \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[16,8],jets:[16,8],muons:[8],taus:[16,8]}' \
#     algorithm.encoder.pooling=sum_max \
#     algorithm.encoder.rho_nodes='[48,16]' \
#     algorithm.input_noise_std=0.003 \
#     algorithm.delta=10.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0020826722159159876 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 599: cvar25eff=1.3415, mse=0.17205  << ORIGINAL PICK
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae \
#     experiment_name=physics_dsae_pareto \
#     run_name=cvar25_t599 \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=sum_max \
#     algorithm.encoder.rho_nodes='[48,16]' \
#     algorithm.input_noise_std=0.001 \
#     algorithm.delta=10.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0021320360922839053 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ========================================================================
# CVAR10 TRAINING  (study: cvar10eff_vs_mse, 8 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 158: cvar10eff=0.13656, mse=0.21135
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae \
#     experiment_name=physics_dsae_pareto \
#     run_name=cvar10_t158 \
#     evaluation.callbacks.anomaly_efficiency.cvar_summary=0.10 \
#     algorithm.delta=7.0 \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,16],jets:[24,16],muons:[8,8],taus:[24,16]}' \
#     algorithm.encoder.pooling=sum_max \
#     algorithm.encoder.rho_nodes='[48,16]' \
#     algorithm.input_noise_std=0.0003 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0016465450003744335 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 287: cvar10eff=0.1274, mse=0.21024
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae \
#     experiment_name=physics_dsae_pareto \
#     run_name=cvar10_t287 \
#     evaluation.callbacks.anomaly_efficiency.cvar_summary=0.10 \
#     algorithm.delta=4.0 \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,16],jets:[24,16],muons:[8,8],taus:[24,16]}' \
#     algorithm.encoder.pooling=sum_max \
#     algorithm.encoder.rho_nodes='[48,16]' \
#     algorithm.input_noise_std=0.0003 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0012047479270146562 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 303: cvar10eff=0.1213, mse=0.20207
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae \
#     experiment_name=physics_dsae_pareto \
#     run_name=cvar10_t303 \
#     evaluation.callbacks.anomaly_efficiency.cvar_summary=0.10 \
#     algorithm.delta=4.0 \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,16],jets:[24,16],muons:[8,8],taus:[24,16]}' \
#     algorithm.encoder.pooling=sum_max \
#     algorithm.encoder.rho_nodes='[48,16]' \
#     algorithm.input_noise_std=0.0003 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0013213706557827839 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 317: cvar10eff=0.16402, mse=0.23487
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae \
#     experiment_name=physics_dsae_pareto \
#     run_name=cvar10_t317 \
#     evaluation.callbacks.anomaly_efficiency.cvar_summary=0.10 \
#     algorithm.delta=4.0 \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,16],jets:[24,16],muons:[8,8],taus:[24,16]}' \
#     algorithm.encoder.pooling=sum_max \
#     algorithm.encoder.rho_nodes='[48,16]' \
#     algorithm.input_noise_std=0.0003 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0012654273659494972 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 345: cvar10eff=0.13961, mse=0.21418
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae \
#     experiment_name=physics_dsae_pareto \
#     run_name=cvar10_t345 \
#     evaluation.callbacks.anomaly_efficiency.cvar_summary=0.10 \
#     algorithm.delta=4.0 \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,16],jets:[24,16],muons:[8,8],taus:[24,16]}' \
#     algorithm.encoder.pooling=sum_max \
#     algorithm.encoder.rho_nodes='[48,16]' \
#     algorithm.input_noise_std=0.0003 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0010980307553091475 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 422: cvar10eff=0.15182, mse=0.22511
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae \
#     experiment_name=physics_dsae_pareto \
#     run_name=cvar10_t422 \
#     evaluation.callbacks.anomaly_efficiency.cvar_summary=0.10 \
#     algorithm.delta=4.0 \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,16],jets:[24,16],muons:[8,8],taus:[24,16]}' \
#     algorithm.encoder.pooling=sum_max \
#     algorithm.encoder.rho_nodes='[48,16]' \
#     algorithm.input_noise_std=0.0003 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.001143506422913001 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 460: cvar10eff=0.20064, mse=0.26134  << BEST cvar10eff
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae \
#     experiment_name=physics_dsae_pareto \
#     run_name=cvar10_t460 \
#     evaluation.callbacks.anomaly_efficiency.cvar_summary=0.10 \
#     algorithm.delta=4.0 \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,16],jets:[24,16],muons:[8,8],taus:[24,16]}' \
#     algorithm.encoder.pooling=sum_max \
#     algorithm.encoder.rho_nodes='[48,16]' \
#     algorithm.input_noise_std=0.0003 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0012357959029436457 \
#     algorithm.optimizer.weight_decay=0.0 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 599: cvar10eff=0.14571, mse=0.21631  << KNEE
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae \
#     experiment_name=physics_dsae_pareto \
#     run_name=cvar10_t599 \
#     evaluation.callbacks.anomaly_efficiency.cvar_summary=0.10 \
#     algorithm.delta=10.0 \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,16],jets:[24,16],muons:[8,8],taus:[24,16]}' \
#     algorithm.encoder.pooling=sum_max \
#     algorithm.encoder.rho_nodes='[48,16]' \
#     algorithm.input_noise_std=0.0003 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0012669527227387835 \
#     algorithm.optimizer.weight_decay=0.0 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ========================================================================
# CAP TRAINING  (study: cap_vs_mse_b16k, 7 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 172: cap=-0.21178, mse=0.16963
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae_agnostic \
#     experiment_name=physics_dsae_pareto \
#     run_name=cap_t172 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[32,16],jets:[32,16],muons:[8,8],taus:[32,16]}' \
#     algorithm.encoder.pooling=sum_max \
#     algorithm.encoder.rho_nodes='[48,16]' \
#     algorithm.input_noise_std=0.001 \
#     algorithm.delta=7.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0029743936189379084 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 200: cap=-0.21704, mse=0.16881  << KNEE
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae_agnostic \
#     experiment_name=physics_dsae_pareto \
#     run_name=cap_t200 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[32,16],jets:[32,16],muons:[8,8],taus:[32,16]}' \
#     algorithm.encoder.pooling=sum_max \
#     algorithm.encoder.rho_nodes='[48,16]' \
#     algorithm.input_noise_std=0.001 \
#     algorithm.delta=7.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.002767913212195896 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 274: cap=-0.20924, mse=0.17029
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae_agnostic \
#     experiment_name=physics_dsae_pareto \
#     run_name=cap_t274 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[32,16],jets:[32,16],muons:[8,8],taus:[32,16]}' \
#     algorithm.encoder.pooling=sum_max \
#     algorithm.encoder.rho_nodes='[48,16]' \
#     algorithm.input_noise_std=0.001 \
#     algorithm.delta=7.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0024357043481996543 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 376: cap=-0.20366, mse=0.17037
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae_agnostic \
#     experiment_name=physics_dsae_pareto \
#     run_name=cap_t376 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[32,16],jets:[32,16],muons:[8,8],taus:[32,16]}' \
#     algorithm.encoder.pooling=sum_max \
#     algorithm.encoder.rho_nodes='[48,16]' \
#     algorithm.input_noise_std=0.001 \
#     algorithm.delta=7.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.002572766239895491 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 561: cap=-0.21858, mse=0.15827
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae_agnostic \
#     experiment_name=physics_dsae_pareto \
#     run_name=cap_t561 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[32,16],jets:[32,16],muons:[8,8],taus:[32,16]}' \
#     algorithm.encoder.pooling=sum_max \
#     algorithm.encoder.rho_nodes='[48,16]' \
#     algorithm.input_noise_std=0.01 \
#     algorithm.delta=7.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0027460341691942776 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 570: cap=-0.19384, mse=0.17737  << ORIGINAL PICK | BEST cap
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae_agnostic \
#     experiment_name=physics_dsae_pareto \
#     run_name=cap_t570 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[32,16],jets:[32,16],muons:[8,8],taus:[32,16]}' \
#     algorithm.encoder.pooling=sum_max \
#     algorithm.encoder.rho_nodes='[48,16]' \
#     algorithm.input_noise_std=0.01 \
#     algorithm.delta=7.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0028585937890006803 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 584: cap=-0.19812, mse=0.17095
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae_agnostic \
#     experiment_name=physics_dsae_pareto \
#     run_name=cap_t584 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[32,16],jets:[32,16],muons:[8,8],taus:[32,16]}' \
#     algorithm.encoder.pooling=sum_max \
#     algorithm.encoder.rho_nodes='[48,16]' \
#     algorithm.input_noise_std=0.01 \
#     algorithm.delta=7.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0029989721615935492 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ========================================================================
# CONSISTENCY TRAINING  (study: consistency_vs_mse_b16k, 36 Pareto points, trimmed to 12 around the knee (endpoints kept))
# ========================================================================
# ------------------------------------------------------------------------
# trial 208: consistency=-4.4793e-05, mse=0.79302  << BEST consistency
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae_agnostic \
#     experiment_name=physics_dsae_pareto \
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
#     algorithm.delta=3.0 \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.rho_nodes='[48,24]' \
#     algorithm.input_noise_std=0.003 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0029915331047474544 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 262: consistency=-0.00038692, mse=0.55123
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae_agnostic \
#     experiment_name=physics_dsae_pareto \
#     run_name=consistency_t262 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.delta=10.0 \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.rho_nodes='[48,24]' \
#     algorithm.input_noise_std=0.003 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0026944638334525394 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 268: consistency=-0.00039188, mse=0.51945
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae_agnostic \
#     experiment_name=physics_dsae_pareto \
#     run_name=consistency_t268 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.delta=10.0 \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.rho_nodes='[48,24]' \
#     algorithm.input_noise_std=0.003 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0027036756563788324 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 278: consistency=-0.010234, mse=0.42559
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae_agnostic \
#     experiment_name=physics_dsae_pareto \
#     run_name=consistency_t278 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.delta=3.0 \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=sum \
#     algorithm.encoder.rho_nodes='[48,24]' \
#     algorithm.input_noise_std=0.003 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0023604287917878297 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 283: consistency=-0.0080067, mse=0.43155
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae_agnostic \
#     experiment_name=physics_dsae_pareto \
#     run_name=consistency_t283 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.delta=3.0 \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=sum \
#     algorithm.encoder.rho_nodes='[48,24]' \
#     algorithm.input_noise_std=0.003 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0023587126457593156 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 291: consistency=-0.01166, mse=0.42155
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae_agnostic \
#     experiment_name=physics_dsae_pareto \
#     run_name=consistency_t291 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.delta=3.0 \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=sum \
#     algorithm.encoder.rho_nodes='[48,24]' \
#     algorithm.input_noise_std=0.0001 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.00273699465069138 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 307: consistency=-0.053388, mse=0.18607
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae_agnostic \
#     experiment_name=physics_dsae_pareto \
#     run_name=consistency_t307 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.delta=3.0 \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=sum_max \
#     algorithm.encoder.rho_nodes='[48,24]' \
#     algorithm.input_noise_std=0.003 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.001998520820442121 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 320: consistency=-0.002197, mse=0.48389
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae_agnostic \
#     experiment_name=physics_dsae_pareto \
#     run_name=consistency_t320 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.delta=3.0 \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=sum \
#     algorithm.encoder.rho_nodes='[48,24]' \
#     algorithm.input_noise_std=0.003 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.00205311778063389 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 336: consistency=-0.002736, mse=0.46303  << KNEE
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae_agnostic \
#     experiment_name=physics_dsae_pareto \
#     run_name=consistency_t336 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.delta=3.0 \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[32,16],jets:[32,16],muons:[8,8],taus:[32,16]}' \
#     algorithm.encoder.pooling=sum \
#     algorithm.encoder.rho_nodes='[48,24]' \
#     algorithm.input_noise_std=0.003 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.002091992679006277 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 610: consistency=-0.0045894, mse=0.44517
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae_agnostic \
#     experiment_name=physics_dsae_pareto \
#     run_name=consistency_t610 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.delta=3.0 \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[32,16],jets:[32,16],muons:[8,8],taus:[32,16]}' \
#     algorithm.encoder.pooling=sum \
#     algorithm.encoder.rho_nodes='[48,24]' \
#     algorithm.input_noise_std=0.0003 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0023024964998540234 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 613: consistency=-0.012285, mse=0.41512
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae_agnostic \
#     experiment_name=physics_dsae_pareto \
#     run_name=consistency_t613 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.delta=3.0 \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=sum \
#     algorithm.encoder.rho_nodes='[48,24]' \
#     algorithm.input_noise_std=0.0003 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0022990449724021415 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 614: consistency=-0.0013858, mse=0.51138
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae_agnostic \
#     experiment_name=physics_dsae_pareto \
#     run_name=consistency_t614 \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.delta=3.0 \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=sum \
#     algorithm.encoder.rho_nodes='[48,24]' \
#     algorithm.input_noise_std=0.003 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.00230434773026104 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ========================================================================
# STABILITY TRAINING  (study: drift_vs_mse_b16k, 1 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 565: drift=0.013551, mse=0.19888  << ORIGINAL PICK | BEST drift
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae_agnostic \
#     experiment_name=physics_dsae_pareto \
#     run_name=stability_t565 \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,16],jets:[24,16],muons:[8,8],taus:[24,16]}' \
#     algorithm.encoder.pooling=sum_max \
#     algorithm.encoder.rho_nodes='[48,16]' \
#     algorithm.input_noise_std=0.003 \
#     algorithm.delta=10.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.002156643622954745 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ========================================================================
# WASSERSTEIN TRAINING  (study: wasserstein_vs_mse_b16k, 13 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 167: wasserstein=0.002219, mse=0.35188
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae_agnostic \
#     experiment_name=physics_dsae_pareto \
#     run_name=wasserstein_t167 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=sum_max \
#     algorithm.encoder.rho_nodes='[48,24]' \
#     algorithm.input_noise_std=0.0 \
#     algorithm.delta=4.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0016382588173169752 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=5.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 243: wasserstein=0.002217, mse=0.58801
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae_agnostic \
#     experiment_name=physics_dsae_pareto \
#     run_name=wasserstein_t243 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[32,16],jets:[32,16],muons:[8,8],taus:[32,16]}' \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.rho_nodes='[48,24]' \
#     algorithm.input_noise_std=0.01 \
#     algorithm.delta=7.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0007379438697720951 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=5.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 315: wasserstein=0.0022446, mse=0.3444
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae_agnostic \
#     experiment_name=physics_dsae_pareto \
#     run_name=wasserstein_t315 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=sum_max \
#     algorithm.encoder.rho_nodes='[48,24]' \
#     algorithm.input_noise_std=0.0 \
#     algorithm.delta=4.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0017210747742029834 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=5.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 333: wasserstein=0.0030911, mse=0.17715
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae_agnostic \
#     experiment_name=physics_dsae_pareto \
#     run_name=wasserstein_t333 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=sum_max \
#     algorithm.encoder.rho_nodes='[48,16]' \
#     algorithm.input_noise_std=0.01 \
#     algorithm.delta=7.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.002006911116552061 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 368: wasserstein=0.0023346, mse=0.32727
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae_agnostic \
#     experiment_name=physics_dsae_pareto \
#     run_name=wasserstein_t368 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=sum_max \
#     algorithm.encoder.rho_nodes='[48,24]' \
#     algorithm.input_noise_std=0.01 \
#     algorithm.delta=4.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0021730373757210134 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=5.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 383: wasserstein=0.001808, mse=0.67774  << ORIGINAL PICK | BEST wasserstein
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae_agnostic \
#     experiment_name=physics_dsae_pareto \
#     run_name=wasserstein_t383 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.rho_nodes='[48,24]' \
#     algorithm.input_noise_std=0.01 \
#     algorithm.delta=4.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0012119364214017788 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=5.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 393: wasserstein=0.001917, mse=0.66425
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae_agnostic \
#     experiment_name=physics_dsae_pareto \
#     run_name=wasserstein_t393 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.rho_nodes='[48,24]' \
#     algorithm.input_noise_std=0.01 \
#     algorithm.delta=4.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.001227384513406042 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=5.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 402: wasserstein=0.0027386, mse=0.21921
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae_agnostic \
#     experiment_name=physics_dsae_pareto \
#     run_name=wasserstein_t402 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=sum_max \
#     algorithm.encoder.rho_nodes='[48,24]' \
#     algorithm.input_noise_std=0.01 \
#     algorithm.delta=4.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0013043008136437097 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=5.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 429: wasserstein=0.0026811, mse=0.22303
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae_agnostic \
#     experiment_name=physics_dsae_pareto \
#     run_name=wasserstein_t429 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=sum_max \
#     algorithm.encoder.rho_nodes='[48,24]' \
#     algorithm.input_noise_std=0.01 \
#     algorithm.delta=4.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0013328254387575083 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=5.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 433: wasserstein=0.002403, mse=0.23095  << KNEE
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae_agnostic \
#     experiment_name=physics_dsae_pareto \
#     run_name=wasserstein_t433 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=sum_max \
#     algorithm.encoder.rho_nodes='[48,24]' \
#     algorithm.input_noise_std=0.01 \
#     algorithm.delta=4.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0013409767254587224 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=5.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 474: wasserstein=0.0019952, mse=0.61819
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae_agnostic \
#     experiment_name=physics_dsae_pareto \
#     run_name=wasserstein_t474 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.rho_nodes='[48,16]' \
#     algorithm.input_noise_std=0.01 \
#     algorithm.delta=7.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.001557712795504778 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 513: wasserstein=0.0021853, mse=0.61686
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae_agnostic \
#     experiment_name=physics_dsae_pareto \
#     run_name=wasserstein_t513 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.rho_nodes='[48,24]' \
#     algorithm.input_noise_std=0.01 \
#     algorithm.delta=4.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0013846645740463919 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=5.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 526: wasserstein=0.0019361, mse=0.63704
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsae_agnostic \
#     experiment_name=physics_dsae_pareto \
#     run_name=wasserstein_t526 \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.rho_nodes='[48,24]' \
#     algorithm.input_noise_std=0.01 \
#     algorithm.delta=4.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0013629344919308167 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=5.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ========================================================================
# SUBMIT EVERYTHING ABOVE TO CLARIDEN  (one slurm job per command)
# ========================================================================
# Set paths.raw_data_dir to the data location on clariden; any extra
# hydra overrides appended here are added to every job.
# bash scripts/cluster/submit_pareto.sh scripts/physics/rundsae_pareto.sh \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files
