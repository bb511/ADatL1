# ========================================================================
# DSAE PARETO-FRONT TRAINING COMMANDS
# ========================================================================
# These are the training commands for every point on the Pareto front of
# Retry file: the two physics_dsae_pareto runs that never produced an MLflow
# run in the 2026-07-27 campaign (drivers died before slurm submission).
# Submit with:
#   bash scripts/submit_pareto.sh scripts/physics/rundsae_pareto_retry.sh \
#       paths.raw_data_dir=/iopsstor/scratch/cscs/podagiu/adl1t_data/parquet_files

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

