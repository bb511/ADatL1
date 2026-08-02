#!/usr/bin/env bash
# ========================================================================
# DTE PARETO-FRONT TRAINING COMMANDS
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
# CVAR25 TRAINING  (study: cvar25eff_vs_ascoreq99, 25 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 17: cvar25eff=155.29, ascoreq99=0.32773
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=cvar25_t17 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.beta_end=0.02078962079330459 \
#     algorithm.n_bins=30.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0010116581432583122 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.dropout=0.5 \
#     algorithm.predictor.hidden_dims='[128,64,32]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 23: cvar25eff=148.63, ascoreq99=0.1462
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=cvar25_t23 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.beta_end=0.06169553177787322 \
#     algorithm.n_bins=5.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0007740988120237255 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.hidden_dims='[64,32,16]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 58: cvar25eff=135.44, ascoreq99=0.11721
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=cvar25_t58 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.beta_end=0.058505305971494864 \
#     algorithm.n_bins=7.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0012200168224538004 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.hidden_dims='[128,64]' \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 87: cvar25eff=161.84, ascoreq99=0.53883  << BEST cvar25eff
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=cvar25_t87 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.beta_end=0.0028154355895198494 \
#     algorithm.n_bins=20.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.000157298528838366 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.hidden_dims='[128,64]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 88: cvar25eff=153.66, ascoreq99=0.31153
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=cvar25_t88 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.beta_end=0.012031981750168142 \
#     algorithm.n_bins=5.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.00010477190839425544 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.hidden_dims='[128,64,32]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 91: cvar25eff=41.87, ascoreq99=0.055017
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=cvar25_t91 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.beta_end=0.032383825628901894 \
#     algorithm.n_bins=7.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.001659951985922931 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.hidden_dims='[128,64,32]' \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 120: cvar25eff=158.28, ascoreq99=0.53243
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=cvar25_t120 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.beta_end=0.0035503805342661653 \
#     algorithm.n_bins=15.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.00017061810157118412 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.hidden_dims='[64,32]' \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 121: cvar25eff=156.83, ascoreq99=0.34489
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=cvar25_t121 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.beta_end=0.009132991828827099 \
#     algorithm.n_bins=20.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.00038718341531395693 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.hidden_dims='[32,16]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 152: cvar25eff=141.01, ascoreq99=0.12547
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=cvar25_t152 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.beta_end=0.09945647952343563 \
#     algorithm.n_bins=5.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.001130625424986157 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.hidden_dims='[128,64]' \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 153: cvar25eff=139.92, ascoreq99=0.12459
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=cvar25_t153 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.beta_end=0.09879025665960872 \
#     algorithm.n_bins=5.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0011258131203286715 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.hidden_dims='[128,64]' \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 163: cvar25eff=67.512, ascoreq99=0.061744
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=cvar25_t163 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.beta_end=0.058987104092398795 \
#     algorithm.n_bins=20.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.001214352291507011 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.hidden_dims='[128,64]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 201: cvar25eff=119.92, ascoreq99=0.11114
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=cvar25_t201 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.beta_end=0.062151302037501986 \
#     algorithm.n_bins=5.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0020482583765641057 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.hidden_dims='[128,64]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 202: cvar25eff=113.52, ascoreq99=0.10558
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=cvar25_t202 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.beta_end=0.09874894477612282 \
#     algorithm.n_bins=5.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.002036786141034728 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.hidden_dims='[128,64]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 348: cvar25eff=121.55, ascoreq99=0.11233
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=cvar25_t348 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.beta_end=0.06022363269031471 \
#     algorithm.n_bins=5.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.001031069792496789 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.hidden_dims='[128,64,32]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 354: cvar25eff=156.14, ascoreq99=0.34274
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=cvar25_t354 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.beta_end=0.009013396547798442 \
#     algorithm.n_bins=20.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0005211809927962974 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.hidden_dims='[32,16]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 395: cvar25eff=137.79, ascoreq99=0.12279
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=cvar25_t395 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.beta_end=0.06421789420412916 \
#     algorithm.n_bins=5.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0007756796287318881 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.hidden_dims='[128,64]' \
#     trainer.gradient_clip_val=0.5 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 409: cvar25eff=151, ascoreq99=0.14779
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=cvar25_t409 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.beta_end=0.054075665882136525 \
#     algorithm.n_bins=5.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0008466147282093311 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.hidden_dims='[64,32,16]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 410: cvar25eff=153.04, ascoreq99=0.1494  << KNEE
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=cvar25_t410 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.beta_end=0.054157904535158584 \
#     algorithm.n_bins=5.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0008535551929693655 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.hidden_dims='[64,32,16]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 413: cvar25eff=152.92, ascoreq99=0.14927
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=cvar25_t413 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.beta_end=0.05437754142723145 \
#     algorithm.n_bins=5.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0008493312128309494 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.hidden_dims='[64,32,16]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 449: cvar25eff=35.355, ascoreq99=0.046734
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=cvar25_t449 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.beta_end=0.09982405891674605 \
#     algorithm.n_bins=20.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.001167211150099495 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.hidden_dims='[128,64]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 452: cvar25eff=49.729, ascoreq99=0.061485
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=cvar25_t452 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.beta_end=0.055774705522041376 \
#     algorithm.n_bins=20.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0012389802344133717 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.hidden_dims='[128,64,32]' \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 457: cvar25eff=33.602, ascoreq99=0.045442
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=cvar25_t457 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.beta_end=0.09977583753133519 \
#     algorithm.n_bins=20.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0012248857791561426 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.hidden_dims='[128,64]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 540: cvar25eff=160.28, ascoreq99=0.53611
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=cvar25_t540 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.beta_end=0.0030884745556762545 \
#     algorithm.n_bins=20.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.00014909682239741754 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.hidden_dims='[128,64]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 587: cvar25eff=144.56, ascoreq99=0.12547
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=cvar25_t587 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.beta_end=0.09941987344758356 \
#     algorithm.n_bins=5.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.00117694544531837 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.hidden_dims='[128,64]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 591: cvar25eff=144.87, ascoreq99=0.12553
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=cvar25_t591 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     algorithm.beta_end=0.09996917956881306 \
#     algorithm.n_bins=5.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0011671074344640079 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.1 \
#     algorithm.predictor.hidden_dims='[128,64]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ========================================================================
# CAP TRAINING  (study: cap_vs_ascoreq99, 50 Pareto points, trimmed to 12 around the knee (endpoints kept))
# ========================================================================
# ------------------------------------------------------------------------
# trial 39: cap=-0.099298, ascoreq99=0.50984  << BEST cap
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=cap_t39 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.002291892365390188 \
#     algorithm.n_bins=15.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=3.1159923079169685e-05 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.hidden_dims='[64,32,16]' \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 267: cap=-0.49004, ascoreq99=0.031121
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=cap_t267 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.09039141540798509 \
#     algorithm.n_bins=15.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.002039407010578193 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.hidden_dims='[128,64,32]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 339: cap=-0.17836, ascoreq99=0.11769
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=cap_t339 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.04749653061782472 \
#     algorithm.n_bins=20.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0015845730335422047 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.hidden_dims='[64,32,32]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 345: cap=-0.17876, ascoreq99=0.11758
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=cap_t345 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.04766886157791229 \
#     algorithm.n_bins=20.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.001582386383253037 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.hidden_dims='[64,32,32]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 347: cap=-0.179, ascoreq99=0.11746
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=cap_t347 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.04778874378193856 \
#     algorithm.n_bins=20.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0015853208998754225 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.hidden_dims='[64,32,32]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 394: cap=-0.26159, ascoreq99=0.088322
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=cap_t394 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.04173396338754444 \
#     algorithm.n_bins=30.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0025553679467790373 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.hidden_dims='[64,32,32]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 421: cap=-0.20177, ascoreq99=0.089173
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=cap_t421 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.05113841290787615 \
#     algorithm.n_bins=30.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0013900357592495128 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.hidden_dims='[64,32,32]' \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 427: cap=-0.20089, ascoreq99=0.093867
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=cap_t427 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.051722985148860624 \
#     algorithm.n_bins=30.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0014029470731003462 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.hidden_dims='[64,32,32]' \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 461: cap=-0.21357, ascoreq99=0.088694
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=cap_t461 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.05075672582536357 \
#     algorithm.n_bins=30.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0014638120155056914 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.hidden_dims='[64,32,32]' \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 465: cap=-0.20905, ascoreq99=0.0891
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=cap_t465 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.05070721293724392 \
#     algorithm.n_bins=30.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.001464007636036333 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.hidden_dims='[64,32,32]' \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 515: cap=-0.18059, ascoreq99=0.11088
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=cap_t515 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.04713276907831173 \
#     algorithm.n_bins=30.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0010651597807106949 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.hidden_dims='[64,32,32]' \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 528: cap=-0.18073, ascoreq99=0.10776  << KNEE
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=cap_t528 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.047127863374303476 \
#     algorithm.n_bins=30.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0010636054324071852 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.hidden_dims='[64,32,32]' \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ========================================================================
# CONSISTENCY TRAINING  (study: consistency_vs_ascoreq99, 10 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 5: consistency=-6.08e-05, ascoreq99=0.36966  << BEST consistency
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=consistency_t5 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.012250947975808205 \
#     algorithm.n_bins=20.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0002498629572791527 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.hidden_dims='[128,64]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 191: consistency=-0.00018181, ascoreq99=0.07059
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=consistency_t191 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.06585481390220657 \
#     algorithm.n_bins=5.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0005487725063117573 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.hidden_dims='[128,64]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 205: consistency=-0.00012767, ascoreq99=0.075014  << KNEE
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=consistency_t205 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.05306243794174806 \
#     algorithm.n_bins=5.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0006484061536001698 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.hidden_dims='[128,64,32]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 463: consistency=-9.3422e-05, ascoreq99=0.24944
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=consistency_t463 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.0493291038262456 \
#     algorithm.n_bins=20.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.00027868424048259853 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.hidden_dims='[128,64]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 464: consistency=-9.7675e-05, ascoreq99=0.24938
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=consistency_t464 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.049429676144959626 \
#     algorithm.n_bins=20.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.00027865585847051423 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.hidden_dims='[128,64]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 465: consistency=-8.7392e-05, ascoreq99=0.24952
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=consistency_t465 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.049174574568410366 \
#     algorithm.n_bins=20.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.000278708740614859 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.hidden_dims='[128,64]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 466: consistency=-0.00010585, ascoreq99=0.24935
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=consistency_t466 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.049221243893180375 \
#     algorithm.n_bins=20.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0002795301967491366 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.hidden_dims='[128,64]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 529: consistency=-0.00011513, ascoreq99=0.24342
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=consistency_t529 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.09128784935974237 \
#     algorithm.n_bins=20.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.00026501098870544226 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.hidden_dims='[128,64]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 559: consistency=-0.00023974, ascoreq99=0.016882
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=consistency_t559 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.05748437128047107 \
#     algorithm.n_bins=5.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.0015683898734669445 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.hidden_dims='[128,64,32]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 561: consistency=-0.00022804, ascoreq99=0.016983
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=consistency_t561 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.057338029173538536 \
#     algorithm.n_bins=5.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.lr=0.001567499856875433 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.hidden_dims='[128,64,32]' \
#     trainer.gradient_clip_val=1.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ========================================================================
# STABILITY TRAINING  (study: drift_vs_ascoreq99, 5 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 208: drift=5.6015e-05, ascoreq99=0.022241
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=stability_t208 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.08370977786290412 \
#     algorithm.n_bins=5.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0016537030995908877 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.hidden_dims='[128,64,32]' \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 474: drift=8.1388e-06, ascoreq99=0.030437  << BEST drift
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=stability_t474 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.0799026517208883 \
#     algorithm.n_bins=10.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.00261998631545004 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=silu \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.hidden_dims='[128,64,32]' \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 527: drift=0.00010389, ascoreq99=0.012386  << KNEE
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=stability_t527 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.09154738523174119 \
#     algorithm.n_bins=5.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.002353658607629892 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.hidden_dims='[128,64,32]' \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 576: drift=0.00037479, ascoreq99=0.009037
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=stability_t576 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.09111622327952709 \
#     algorithm.n_bins=5.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.002322131091618466 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.hidden_dims='[128,64,32]' \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 578: drift=0.00018334, ascoreq99=0.011298
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=stability_t578 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.09113478034211861 \
#     algorithm.n_bins=5.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0023230543913367687 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.predictor.activation=gelu \
#     algorithm.predictor.dropout=0.0 \
#     algorithm.predictor.hidden_dims='[128,64,32]' \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ========================================================================
# WASSERSTEIN TRAINING  (study: wasserstein_vs_ascoreq99, 6 Pareto points)
# ========================================================================
# ------------------------------------------------------------------------
# trial 175: wasserstein=7.2796e-05, ascoreq99=0.017403  << BEST wasserstein
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=wasserstein_t175 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.08666808768778901 \
#     algorithm.n_bins=5 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0007737525546878006 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.hidden_dims='[128,64,32]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 195: wasserstein=0.00012479, ascoreq99=0.012834
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=wasserstein_t195 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.08977027222716984 \
#     algorithm.n_bins=5 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0007027171512752789 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.hidden_dims='[128,64,32]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ------------------------------------------------------------------------
# trial 222: wasserstein=9.1631e-05, ascoreq99=0.015565
# ------------------------------------------------------------------------
# taskset -c 6-8 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=wasserstein_t222 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.0912098332810758 \
#     algorithm.n_bins=5 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0006471391262003666 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.hidden_dims='[128,64,32]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# ------------------------------------------------------------------------
# trial 226: wasserstein=0.00010962, ascoreq99=0.014887  << KNEE
# ------------------------------------------------------------------------
# taskset -c 9-11 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=wasserstein_t226 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.09107374002293242 \
#     algorithm.n_bins=5 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0006470661925113555 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.hidden_dims='[128,64,32]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[3]

# ------------------------------------------------------------------------
# trial 474: wasserstein=8.1056e-05, ascoreq99=0.01642
# ------------------------------------------------------------------------
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=wasserstein_t474 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.08572820953627151 \
#     algorithm.n_bins=5 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0008234102200488883 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.hidden_dims='[128,64,32]' \
#     trainer.gradient_clip_val=0.0 \
#     trainer=gpu \
#     trainer.devices=[0]

# ------------------------------------------------------------------------
# trial 596: wasserstein=0.00011452, ascoreq99=0.014123
# ------------------------------------------------------------------------
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/dte_agnostic \
#     experiment_name=physics_dte_q99_pareto \
#     run_name=wasserstein_t596 \
#     algorithm.target_rate=0.01 \
#     algorithm.base_rate=null \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.consistency_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.consistency_sn_zb_ema_ckpt=null \
#     ~evaluation.evaluator.ckpts.summary.operational_drift_ema \
#     ~evaluation.evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluation.evaluator.ckpts.summary.consistency_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.beta_end=0.0896221654060135 \
#     algorithm.n_bins=5 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.lr=0.0007547448062303283 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.predictor.activation=relu \
#     algorithm.predictor.dropout=0.25 \
#     algorithm.predictor.hidden_dims='[128,64,32]' \
#     trainer.gradient_clip_val=2.0 \
#     trainer=gpu \
#     trainer.devices=[1]

# ========================================================================
# SUBMIT EVERYTHING ABOVE TO CLARIDEN  (one slurm job per command)
# ========================================================================
# Set paths.raw_data_dir to the data location on clariden; any extra
# hydra overrides appended here are added to every job.
# bash scripts/cluster/submit_pareto.sh scripts/physics/rundte_q99_pareto.sh \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files
