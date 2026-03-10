# Run the AE models.


# MODEL TRAINING
# =======================

# Semi-supervised training.
# taskset -c 9-11 \
# python3 src/train.py \
#     -m \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=ae \
#     experiment_name=ae_best_models \
#     run_name=trial_352 \
#     algorithm.optimizer.lr=0.0008714400208882324 \
#     algorithm.loss.delta=3.0 \
#     trainer.gradient_clip_val=0.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.encoder.out_dim=96 \
#     algorithm.encoder.nodes='[64]' \
#     algorithm.input_noise_std=0.001 \
#     trainer=gpu \
#     trainer.devices=[2]

# Agnostic training.
# taskset -c 15-17 \
# python3 src/train.py \
#     -m \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=ae_agnostic \
#     experiment_name=ae_agnostic_best_models \
#     run_name=trial_238 \
#     algorithm.optimizer.lr=0.0023866403103855985 \
#     algorithm.loss.delta=3.0 \
#     trainer.gradient_clip_val=0.5 \
#     algorithm.optimizer.betas='[0.9, 0.99]' \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.encoder.out_dim=16 \
#     algorithm.encoder.nodes='[64, 64]' \
#     algorithm.input_noise_std=0.0 \
#     trainer=gpu \
#     trainer.devices=[2]

# Agnostic RMSE-threshold training.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=ae_agnostic \
#     experiment_name=ae_agnostic_rmse \
#     run_name=trial_260 \
#     evaluator_callbacks.cap_sn_zb=null \
#     algorithm.optimizer.lr=3.613509306601417e-05 \
#     algorithm.loss.delta=3.0 \
#     trainer.gradient_clip_val=5.0 \
#     algorithm.optimizer.betas='[0.9, 0.999]' \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.encoder.out_dim=96 \
#     algorithm.encoder.nodes='[64]' \
#     algorithm.input_noise_std=0.003 \
#     trainer=gpu \
#     trainer.devices=[0]


# HYPERPARAMETER SEARCHES.
# =======================

# AE hyperparameter search.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=ae \
#     experiment_name=ae_cvar_vs_rmse_search \
#     callbacks.max_rate_mse_ckpt=null \
#     callbacks.stable_mse_ckpt=null \
#     evaluator.ckpts.last=false \
#     evaluator.ckpts.single=null \
#     evaluator_callbacks.reco=null \
#     logger=none \
#     hparams_search=ae_optuna \
#     hydra.sweeper.study_name=cvar25eff_vs_mse_b16k \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=40 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]


# AE agnostic hyperparameter search - CAP vs MSE.
taskset -c 18-20 \
python3 src/train.py \
    -m \
    hydra/launcher=submitit_local \
    hydra.launcher.cpus_per_task=1 \
    hydra.launcher.gpus_per_node=4 \
    paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
    experiment=ae_agnostic \
    experiment_name=ae_agnostic_capSigmoidExponential_vs_mse_search \
    callbacks.max_rate_mse_ckpt=null \
    callbacks.cvar25_ema_ckpt=null \
    evaluator.ckpts.summary=null \
    evaluator_callbacks.reco=null \
    logger=none \
    hparams_search=ae_agnostic_optuna \
    hydra.sweeper.study_name=capSigmoidExponential_vs_mse_b16k \
    hydra.sweeper.n_trials=100 \
    hydra.sweeper.sampler.n_startup_trials=40 \
    trainer=gpu \
    trainer.max_epochs=50 \
    trainer.devices=[0]


# AE agnostic MSE and threshold stability.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=ae_agnostic \
#     experiment_name=ae_agnostic_mse_vs_thres_search \
#     callbacks.max_rate_mse_ckpt=null \
#     callbacks.cvar25_ema_ckpt=null \
#     evaluator.ckpts.last=false \
#     evaluator.ckpts.summary=null \
#     evaluator_callbacks.cap_sn_zb=null \
#     evaluator_callbacks.reco=null \
#     logger=none \
#     hparams_search=ae_agnostic_optuna \
#     optimized_metric_config.main_metric.callback.name=mse_mean_top_vals \
#     +optimized_metric_config.main_metric.callback.params.ckpt_name=main_val \
#     +optimized_metric_config.main_metric.callback.params.test_ds=main_test \
#     optimized_metric_config.main_metric.direction=minimize \
#     optimized_metric_config.sec_metric.callback.name=thres_transfer \
#     optimized_metric_config.sec_metric.callback.params.ckpt_name=null \
#     +optimized_metric_config.sec_metric.callback.params.target_rate=0.25 \
#     ~optimized_metric_config.sec_metric.callback.params.test_ds \
#     optimized_metric_config.sec_metric.direction=minimize \
#     hydra.sweeper.study_name=mse_vs_drift_b16k \
#     hydra.sweeper.n_trials=100 \
#     hydra.sweeper.sampler.n_startup_trials=40 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[0]
