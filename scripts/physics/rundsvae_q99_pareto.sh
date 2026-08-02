#!/usr/bin/env bash
# ========================================================================
# DSVAE PARETO-FRONT TRAINING COMMANDS
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
# CVAR25 TRAINING  (study: cvar25eff_vs_klq99_b16k, 28 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 152: cvar25eff=0.71033, klq99=0.74599
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=cvar25_t152 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.rho_nodes='[24,8]' \
#     algorithm.kl_warmup_frac=0.1 \
#     algorithm.kl_scale=0.002 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=8.493641609920852e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 170: cvar25eff=0.96304, klq99=7.9726
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=cvar25_t170 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.clamp_zlogvar_range='[-10,6]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[16,16],jets:[16,16],muons:[8,8],taus:[16,16]}' \
#     algorithm.encoder.pooling=sum \
#     algorithm.encoder.rho_nodes='[16,8]' \
#     algorithm.kl_warmup_frac=0.1 \
#     algorithm.kl_scale=0.002 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=5.495078002594691e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 174: cvar25eff=0.96516, klq99=7.9861
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=cvar25_t174 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.clamp_zlogvar_range='[-10,6]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[16,16],jets:[16,16],muons:[8,8],taus:[16,16]}' \
#     algorithm.encoder.pooling=sum \
#     algorithm.encoder.rho_nodes='[16,8]' \
#     algorithm.kl_warmup_frac=0.1 \
#     algorithm.kl_scale=0.002 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=5.450863835997349e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 185: cvar25eff=1.0275, klq99=9.1584
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=cvar25_t185 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,16],jets:[24,16],muons:[8,8],taus:[24,16]}' \
#     algorithm.encoder.pooling=sum \
#     algorithm.encoder.rho_nodes='[32,24]' \
#     algorithm.kl_warmup_frac=0.1 \
#     algorithm.kl_scale=0.002 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=5.3076563856024214e-05 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 191: cvar25eff=1.2253, klq99=9.2066
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=cvar25_t191 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,16],jets:[24,16],muons:[8,8],taus:[24,16]}' \
#     algorithm.encoder.pooling=sum \
#     algorithm.encoder.rho_nodes='[32,24]' \
#     algorithm.kl_warmup_frac=0.1 \
#     algorithm.kl_scale=0.002 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=5.008376369781194e-05 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 192: cvar25eff=1.2253, klq99=9.2082  << ORIGINAL PICK | BEST cvar25eff
#   caveat: old script had algorithm.encoder.activation=relu; db trial 192 has gelu
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=cvar25_t192 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,16],jets:[24,16],muons:[8,8],taus:[24,16]}' \
#     algorithm.encoder.pooling=sum \
#     algorithm.encoder.rho_nodes='[32,24]' \
#     algorithm.kl_warmup_frac=0.1 \
#     algorithm.kl_scale=0.002 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=5.010175246154735e-05 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 233: cvar25eff=0.86924, klq99=0.90642
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=cvar25_t233 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.rho_nodes='[24,8]' \
#     algorithm.kl_warmup_frac=0.1 \
#     algorithm.kl_scale=0.002 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=9.107458105238291e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 288: cvar25eff=0.78421, klq99=0.7915
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=cvar25_t288 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.rho_nodes='[24,8]' \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.kl_scale=0.002 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=8.637204093065809e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 290: cvar25eff=0.77808, klq99=0.78803
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=cvar25_t290 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.rho_nodes='[24,8]' \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.kl_scale=0.002 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=8.623558258470181e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 291: cvar25eff=0.73383, klq99=0.76091
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=cvar25_t291 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.rho_nodes='[24,8]' \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.kl_scale=0.002 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=8.51464363477854e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 303: cvar25eff=0.87432, klq99=0.91654
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=cvar25_t303 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.clamp_zlogvar_range='[-20,10]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.rho_nodes='[24,8]' \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.kl_scale=0.001 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=9.057077009701671e-05 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 316: cvar25eff=0.8731, klq99=0.9164
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=cvar25_t316 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.rho_nodes='[24,8]' \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.kl_scale=0.001 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=9.05528356527281e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 317: cvar25eff=0.87188, klq99=0.91255
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=cvar25_t317 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.rho_nodes='[24,8]' \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.kl_scale=0.001 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=9.04270605225249e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 318: cvar25eff=0.75673, klq99=0.77557
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=cvar25_t318 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.rho_nodes='[24,8]' \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.kl_scale=0.001 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=8.553766984551118e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 319: cvar25eff=0.75551, klq99=0.7754
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=cvar25_t319 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.clamp_zlogvar_range='[-20,10]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.rho_nodes='[24,8]' \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.kl_scale=0.001 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=8.55309760245463e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 328: cvar25eff=0.75184, klq99=0.77277
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=cvar25_t328 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.rho_nodes='[24,8]' \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.kl_scale=0.001 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=8.54265809763531e-05 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 329: cvar25eff=0.7433, klq99=0.7662
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=cvar25_t329 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.rho_nodes='[24,8]' \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.kl_scale=0.001 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=8.516358401836919e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 343: cvar25eff=0.89903, klq99=0.97815
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=cvar25_t343 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.rho_nodes='[24,8]' \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.kl_scale=0.001 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=9.24699241892491e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 353: cvar25eff=0.84354, klq99=0.89075
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=cvar25_t353 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.clamp_zlogvar_range='[-6,4]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.rho_nodes='[24,8]' \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.kl_scale=0.001 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=8.971250525267925e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 358: cvar25eff=0.84627, klq99=0.89488
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=cvar25_t358 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.rho_nodes='[24,8]' \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.kl_scale=0.001 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=8.984944295375458e-05 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 366: cvar25eff=0.93685, klq99=1.0154  << KNEE
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=cvar25_t366 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.rho_nodes='[24,8]' \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.kl_scale=0.001 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=9.356250030770893e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 385: cvar25eff=0.84316, klq99=0.8366
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=cvar25_t385 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.clamp_zlogvar_range='[-6,4]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.rho_nodes='[24,8]' \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.kl_scale=0.001 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=8.784159286039542e-05 \
#     algorithm.optimizer.weight_decay=0.0 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 465: cvar25eff=0.92527, klq99=0.99623
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=cvar25_t465 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.rho_nodes='[24,8]' \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.kl_scale=0.001 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=9.300456711470965e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 497: cvar25eff=0.88043, klq99=0.92196
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=cvar25_t497 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.rho_nodes='[24,8]' \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.kl_scale=0.0001 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=9.049366237578435e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 498: cvar25eff=0.85664, klq99=0.90331
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=cvar25_t498 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.clamp_zlogvar_range='[-10,6]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.rho_nodes='[24,8]' \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.kl_scale=0.0001 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=8.99014650442847e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 499: cvar25eff=0.8603, klq99=0.90562
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=cvar25_t499 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.clamp_zlogvar_range='[-10,6]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.rho_nodes='[24,8]' \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.kl_scale=0.0001 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=8.997555178252755e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 564: cvar25eff=0.89109, klq99=0.93228
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=cvar25_t564 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.rho_nodes='[24,8]' \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.kl_scale=3e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=9.079730020943924e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 566: cvar25eff=0.88987, klq99=0.92895
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=cvar25_t566 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.clamp_zlogvar_range='[-10,6]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.rho_nodes='[24,8]' \
#     algorithm.kl_warmup_frac=0.2 \
#     algorithm.kl_scale=3e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=9.069388935542937e-05 \
#     algorithm.optimizer.weight_decay=0.0 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ========================================================================
# CAP TRAINING  (study: cap_vs_klq99_b16k, 8 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 195: cap=-0.13253, klq99=7.2172
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae_agnostic \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=cap_t195 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.clamp_zlogvar_range='[-10,6]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[16,16],jets:[16,16],muons:[8,8],taus:[16,16]}' \
#     algorithm.encoder.pooling=sum \
#     algorithm.encoder.rho_nodes='[32,16,8]' \
#     algorithm.kl_warmup_frac=0.05 \
#     algorithm.kl_scale=0.002 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0007956895340834021 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 343: cap=-0.12164, klq99=7.7103  << BEST cap
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae_agnostic \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=cap_t343 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.clamp_zlogvar_range='[-10,6]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[16,16],jets:[16,16],muons:[8,8],taus:[16,16]}' \
#     algorithm.encoder.pooling=sum \
#     algorithm.encoder.rho_nodes='[32,16,8]' \
#     algorithm.kl_warmup_frac=0.05 \
#     algorithm.kl_scale=0.002 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0007668379284674402 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 346: cap=-0.13168, klq99=7.4219
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae_agnostic \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=cap_t346 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.clamp_zlogvar_range='[-10,6]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[16,16],jets:[16,16],muons:[8,8],taus:[16,16]}' \
#     algorithm.encoder.pooling=sum \
#     algorithm.encoder.rho_nodes='[32,16,8]' \
#     algorithm.kl_warmup_frac=0.05 \
#     algorithm.kl_scale=0.002 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0007693309904368051 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 353: cap=-0.15389, klq99=7.1874  << KNEE
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae_agnostic \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=cap_t353 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.clamp_zlogvar_range='[-10,6]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[16,16],jets:[16,16],muons:[8,8],taus:[16,16]}' \
#     algorithm.encoder.pooling=sum \
#     algorithm.encoder.rho_nodes='[32,16,8]' \
#     algorithm.kl_warmup_frac=0.05 \
#     algorithm.kl_scale=0.002 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.000758060537432625 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 436: cap=-0.12662, klq99=7.4597
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae_agnostic \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=cap_t436 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.clamp_zlogvar_range='[-10,6]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[16,16],jets:[16,16],muons:[8,8],taus:[16,16]}' \
#     algorithm.encoder.pooling=sum \
#     algorithm.encoder.rho_nodes='[32,16,8]' \
#     algorithm.kl_warmup_frac=0.05 \
#     algorithm.kl_scale=0.002 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0007694162811937691 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 438: cap=-0.1249, klq99=7.6181
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae_agnostic \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=cap_t438 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.clamp_zlogvar_range='[-10,6]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[16,16],jets:[16,16],muons:[8,8],taus:[16,16]}' \
#     algorithm.encoder.pooling=sum \
#     algorithm.encoder.rho_nodes='[32,16,8]' \
#     algorithm.kl_warmup_frac=0.05 \
#     algorithm.kl_scale=0.002 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0007676075984078361 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 488: cap=-0.1708, klq99=4.4555
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae_agnostic \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=cap_t488 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.clamp_zlogvar_range='[-10,6]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[16,16],jets:[16,16],muons:[8,8],taus:[16,16]}' \
#     algorithm.encoder.pooling=sum \
#     algorithm.encoder.rho_nodes='[16,8]' \
#     algorithm.kl_warmup_frac=0.05 \
#     algorithm.kl_scale=0.002 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0008105778315060535 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 595: cap=-0.16851, klq99=4.6758  << ORIGINAL PICK
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae_agnostic \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=cap_t595 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.clamp_zlogvar_range='[-10,6]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[16,16],jets:[16,16],muons:[8,8],taus:[16,16]}' \
#     algorithm.encoder.pooling=sum \
#     algorithm.encoder.rho_nodes='[16,8]' \
#     algorithm.kl_warmup_frac=0.05 \
#     algorithm.kl_scale=0.002 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0008122419582794828 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ========================================================================
# STABILITY TRAINING  (study: drift_vs_klq99, 7 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 316: drift=8.1388e-06, klq99=2.4974  << BEST drift
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae_agnostic \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=stability_t316 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.clamp_zlogvar_range='[-20,10]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,16],jets:[24,16],muons:[8,8],taus:[24,16]}' \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.rho_nodes='[16,8]' \
#     algorithm.kl_scale=0.001 \
#     algorithm.kl_warmup_frac=0.05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=7.356143597833672e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 559: drift=0.00015178, klq99=0.60131
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae_agnostic \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=stability_t559 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.clamp_zlogvar_range='[-6,4]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[16,8],jets:[16,8],muons:[8],taus:[16,8]}' \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.rho_nodes='[16,8]' \
#     algorithm.kl_scale=3e-05 \
#     algorithm.kl_warmup_frac=0.05 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=7.154666838543678e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 562: drift=0.001014, klq99=0.55921
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae_agnostic \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=stability_t562 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.clamp_zlogvar_range='[-6,4]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[16,8],jets:[16,8],muons:[8],taus:[16,8]}' \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.rho_nodes='[16,8]' \
#     algorithm.kl_scale=3e-05 \
#     algorithm.kl_warmup_frac=0.05 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=6.997477947477446e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 565: drift=3.9736e-05, klq99=0.61185  << ORIGINAL PICK | KNEE
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae_agnostic \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=stability_t565 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.clamp_zlogvar_range='[-6,4]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[16,8],jets:[16,8],muons:[8],taus:[16,8]}' \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.rho_nodes='[16,8]' \
#     algorithm.kl_scale=3e-05 \
#     algorithm.kl_warmup_frac=0.05 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=7.190998896334822e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 566: drift=0.00058281, klq99=0.57308
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae_agnostic \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=stability_t566 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.clamp_zlogvar_range='[-6,4]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[16,8],jets:[16,8],muons:[8],taus:[16,8]}' \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.rho_nodes='[16,8]' \
#     algorithm.kl_scale=3e-05 \
#     algorithm.kl_warmup_frac=0.05 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=7.051575336681769e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 567: drift=0.00019966, klq99=0.57576
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae_agnostic \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=stability_t567 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.clamp_zlogvar_range='[-6,4]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[16,8],jets:[16,8],muons:[8],taus:[16,8]}' \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.rho_nodes='[16,8]' \
#     algorithm.kl_scale=3e-05 \
#     algorithm.kl_warmup_frac=0.05 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=7.061801456986643e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 569: drift=0.00043911, klq99=0.57443
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae_agnostic \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=stability_t569 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.clamp_zlogvar_range='[-6,4]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[16,8],jets:[16,8],muons:[8],taus:[16,8]}' \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.rho_nodes='[16,8]' \
#     algorithm.kl_scale=3e-05 \
#     algorithm.kl_warmup_frac=0.05 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=7.056715861795169e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ========================================================================
# WASSERSTEIN TRAINING  (study: wasserstein_vs_klq99, 6 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 195: wasserstein=0.0022583, klq99=0.38421
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae_agnostic \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=wasserstein_t195 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[32,16],jets:[32,16],muons:[8,8],taus:[32,16]}' \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.rho_nodes='[24,8]' \
#     algorithm.kl_scale=0.0003 \
#     algorithm.kl_warmup_frac=0.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=5.000046783837419e-05 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 357: wasserstein=0.0021373, klq99=0.66343  << ORIGINAL PICK
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae_agnostic \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=wasserstein_t357 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=relu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.clamp_zlogvar_range='[-8,6]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.rho_nodes='[16,8]' \
#     algorithm.kl_scale=0.0003 \
#     algorithm.kl_warmup_frac=0.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=5.000065644317902e-05 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 359: wasserstein=0.0022072, klq99=0.5983
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae_agnostic \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=wasserstein_t359 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=False \
#     algorithm.encoder.clamp_zlogvar_range='[-10,6]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=mean \
#     algorithm.encoder.rho_nodes='[48,24,8]' \
#     algorithm.kl_scale=0.0003 \
#     algorithm.kl_warmup_frac=0.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=5.00014015868093e-05 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 410: wasserstein=0.0018624, klq99=31.446  << KNEE
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae_agnostic \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=wasserstein_t410 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.clamp_zlogvar_range='[-10,6]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=sum_max \
#     algorithm.encoder.rho_nodes='[48,24,8]' \
#     algorithm.kl_scale=3e-05 \
#     algorithm.kl_warmup_frac=0.3 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0005976578178759831 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 510: wasserstein=0.0012249, klq99=31.64
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae_agnostic \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=wasserstein_t510 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.clamp_zlogvar_range='[-10,6]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=sum_max \
#     algorithm.encoder.rho_nodes='[48,24,8]' \
#     algorithm.kl_scale=3e-05 \
#     algorithm.kl_warmup_frac=0.3 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.000617510525367858 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 544: wasserstein=0.00077907, klq99=35.324  << BEST wasserstein
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dsvae_agnostic \
#     experiment_name=physics_dsvae_q99_pareto \
#     run_name=wasserstein_t544 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.encoder.activation=gelu \
#     algorithm.encoder.add_counts=True \
#     algorithm.encoder.clamp_zlogvar_range='[-10,6]' \
#     algorithm.encoder.object_phi_nodes='{FET:[8],egammas:[24,8],jets:[24,8],muons:[8],taus:[24,8]}' \
#     algorithm.encoder.pooling=sum_max \
#     algorithm.encoder.rho_nodes='[48,24,8]' \
#     algorithm.kl_scale=3e-05 \
#     algorithm.kl_warmup_frac=0.3 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0005930678592558549 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[1]

# ========================================================================
# SUBMIT EVERYTHING ABOVE TO CLARIDEN  (one slurm job per command)
# ========================================================================
# Set paths.raw_data_dir to the data location on clariden; any extra
# hydra overrides appended here are added to every job.
# bash scripts/cluster/submit_pareto.sh scripts/physics/rundsvae_q99_pareto.sh \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files
