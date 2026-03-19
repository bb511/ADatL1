# Short debug script.
taskset -c 0-2 \
python3 src/train.py \
    -m \
    hydra/launcher=submitit_local \
    hydra.launcher.cpus_per_task=1 \
    hydra.launcher.gpus_per_node=4 \
    paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
    experiment=ae_agnostic \
    experiment_name=ae_agnostic_wasserstein_vs_mse_search \
    callbacks.max_rate_mse_ckpt=null \
    callbacks.cvar25_ema_ckpt=null \
    evaluator.ckpts.last=false \
    evaluator.ckpts.summary=null \
    evaluator_callbacks.cap_sn_zb=null \
    evaluator_callbacks.reco=null \
    logger=none \
    hparams_search=ae_agnostic_optuna \
    optimized_metric_config.main_metric.callback.name=wasserstein \
    optimized_metric_config.main_metric.direction=minimize \
    hydra.sweeper.study_name=wasserstein_vs_mse_b16k \
    hydra.sweeper.direction='[minimize, minimize]' \
    hydra.sweeper.n_trials=100 \
    hydra.sweeper.sampler.n_startup_trials=40 \
    trainer=gpu \
    trainer.max_epochs=1 \
    trainer.devices=[0]
