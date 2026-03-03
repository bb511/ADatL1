# Short debug script.
taskset -c 16-18 \
python3 src/train.py \
    -m \
    paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
    experiment=debug \
    experiment_name=debug \
    callbacks.max_rate_mse_ckpt=null \
    callbacks.cvar25_ema_ckpt=null \
    evaluator.ckpts.summary=null \
    evaluator.ckpts.last=false \
    evaluator_callbacks.reco=null \
    trainer=gpu \
    trainer.max_epochs=50 \
    trainer.devices=[0]

    # hparams_search=ae_agnostic_optuna \
    # hydra.sweeper.study_name=test_v4 \
    # logger=none
    # hydra/launcher=submitit_local \
    # hydra.launcher.cpus_per_task=1 \
    # hydra.launcher.gpus_per_node=4 \
