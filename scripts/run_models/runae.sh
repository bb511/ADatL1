# Run the AE models.


# MODEL TRAINING
# =======================

# Semi-supervised training.
# taskset -c 6-8 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=ae \
#     experiment_name=ae_best_models \
#     run_name=best_model_6 \
#     algorithm.optimizer.lr=0.00037017129833297804 \
#     algorithm.loss.delta=3.0 \
#     trainer.gradient_clip_val=0.5 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.encoder.out_dim=64 \
#     algorithm.encoder.nodes='[64, 64]' \
#     algorithm.input_noise_std=0.0001 \
#     trainer=gpu \
#     trainer.devices=[1]

# Agnostic training.
# taskset -c 15-17 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     hydra.launcher.timeout_min=10000 \
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


# HYPERPARAMETER SEARCHES.
# =======================

# AE batch size study.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=ae \
#     experiment_name=ae_bs_study \
#     evaluator_callbacks.reco=null \
#     hparams_search=ae_optuna \
#     hydra.sweeper.study_name=bs_study \
#     hydra.sweeper.n_trials=8 \
#     hydra.sweeper.sampler.n_startup_trials=15 \
#     trainer=gpu \
#     +trainer.max_steps=30000 \
#     trainer.max_epochs=-1 \
#     trainer.devices=[0]

# AE hyperparameter search.
taskset -c 0-2 \
python3 src/train.py \
    -m \
    hydra/launcher=submitit_local \
    hydra.launcher.cpus_per_task=1 \
    hydra.launcher.gpus_per_node=4 \
    paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
    experiment=ae \
    experiment_name=cvar_vs_rmse_search \
    callbacks.max_rate_mse_ckpt=null \
    evaluator_callbacks.reco=null \
    logger=none \
    hparams_search=ae_optuna \
    hydra.sweeper.study_name=cvar_vs_rmse_b16k \
    hydra.sweeper.n_trials=60 \
    hydra.sweeper.sampler.n_startup_trials=40 \
    trainer=gpu \
    trainer.max_epochs=50 \
    trainer.devices=[0]


# AE agnostic hyperparameter search.
# taskset -c 15-17 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=ae_agnostic \
#     experiment_name=ae_agnostic_search \
#     evaluator_callbacks.reco=null \
#     callbacks.max_rate_mse_ckpt=null \
#     evaluator.ckpts.loss_mse_full_rate0.25kHz=null \
#     logger=none \
#     hparams_search=ae_agnostic_optuna \
#     hydra.sweeper.study_name=cap_vs_loss_bs16k_v3 \
#     hydra.sweeper.n_trials=60 \
#     hydra.sweeper.sampler.n_startup_trials=40 \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[2]

# AE agnostic only rmse.
# taskset -c 15-17 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=ae_agnostic \
#     experiment_name=ae_agnostic_rmse_search \
#     callbacks.max_rate_mse_ckpt=null \
#     callbacks.cvar25_ema_ckpt=null \
#     evaluator.ckpts.summary=null \
#     evaluator_callbacks.cap_sn_zb=null \
#     evaluator_callbacks.reco=null \
#     logger=none \
#     hparams_search=ae_agnostic_optuna \
#     optimized_metric_config.sec_metric=null \
#     optimized_metric_config.callback.name=rmse_q99 \
#     +optimized_metric_config.callback.params.ckpt_ds=main_val \
#     +optimized_metric_config.callback.params.test_ds=main_test \
#     optimized_metric_config.direction=minimize \
#     hydra/sweeper/sampler=tpe \
#     hydra.sweeper.study_name=rmse_b16k \
#     hydra.sweeper.n_trials=60 \
#     hydra.sweeper.sampler.n_startup_trials=40 \
#     hydra.sweeper.sampler.n_ei_candidates=48 \
#     hydra.sweeper.sampler.consider_prior=true \
#     hydra.sweeper.sampler.prior_weight=1.0 \
#     +hydra.sweeper.sampler.gamma=0.2 \
#     hydra.sweeper.direction=minimize \
#     trainer=gpu \
#     trainer.max_epochs=50 \
#     trainer.devices=[2]

