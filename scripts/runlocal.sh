# Short debug script.
taskset -c 0-2 \
python3 src/train.py \
    -m \
    paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
    experiment=vae \
    experiment_name=debug \
    callbacks.max_rate_kl_ckpt=null \
    callbacks.stable_kl_ckpt=null \
    evaluator.ckpts.last=false \
    evaluator.ckpts.single=null \
    evaluator_callbacks.reco=null \
    hparams_search=vae_optuna \
    hydra.sweeper.study_name=cvar25eff_vs_kl_b16k \
    hydra.sweeper.n_trials=100 \
    hydra.sweeper.sampler.n_startup_trials=150 \
    trainer=gpu \
    trainer.max_epochs=1 \
    trainer.devices=[0]
