# Short debug script.
taskset -c 0-2 \
python3 src/train.py \
    -m \
    experiment=debug \
    experiment_name=debug \
    evaluator_callbacks.reco=null \
    paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
    trainer=gpu \
    trainer.max_epochs=5 \
    trainer.devices=[0]

    # +trainer.limit_train_batches=10 \
    # hparams_search=vae_optuna \
    # hydra.sweeper.study_name=test_v2 \
    # logger=none
    # hydra/launcher=submitit_local \
    # hydra.launcher.cpus_per_task=1 \
    # hydra.launcher.gpus_per_node=4 \
