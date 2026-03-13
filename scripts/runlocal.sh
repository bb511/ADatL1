# Short debug script.
taskset -c 18-20 \
python3 src/train.py \
    -m \
    paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
    experiment=debug \
    experiment_name=debug \
    callbacks.max_rate_mse_ckpt=null \
    callbacks.cvar25_ema_ckpt=null \
    evaluator.ckpts.summary=null \
    evaluator_callbacks.reco=null \
    hparams_search=ae_agnostic_optuna \
    hydra.sweeper.study_name=debug \
    hydra.sweeper.n_trials=100 \
    hydra.sweeper.sampler.n_startup_trials=40 \
    trainer=gpu \
    trainer.max_epochs=1 \
    trainer.devices=[0]
