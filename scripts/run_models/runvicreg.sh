# VICreg running commands.
# ========================================================================

# MODEL TRAINING.
# ========================
# vicreg training.
# taskset -c 0-2 \
# python3 src/train.py \
#     -m \
#     hydra/launcher=submitit_local \
#     hydra.launcher.cpus_per_task=1 \
#     hydra.launcher.gpus_per_node=4 \
#     hydra.launcher.timeout_min=10000 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=vicreg \
#     experiment_name=vicreg_best_models \
#     run_name=trial_239 \
#     trainer.gradient_clip_val=2.0 \
#     algorithm.optimizer.lr=4.666306890293863e-05 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.scheduler.scheduler.warmup_ratio=0.03 \
#     algorithm.scheduler.scheduler.min_lr_ratio=0.05 \
#     algorithm.feature_blur.prob=0.2105162659851203 \
#     algorithm.feature_blur.magnitude=0.5635527237006906 \
#     algorithm.object_mask.prob=0.3808553122955952 \
#     algorithm.lorentz_rotation.prob=0.3 \
#     algorithm.loss.inv_coef=10.83423684987546 \
#     algorithm.loss.rvar_coef=1.1580456631609404 \
#     algorithm.loss.rcov_coef=0.3203341416949269 \
#     algorithm.model.nodes='[32]' \
#     algorithm.model.out_dim=64 \
#     algorithm.diagnosis_metrics=true \
#     trainer=gpu \
#     trainer.devices=[2]


# HYPERPARAMETER SEARCHES.
# =======================
# vicreg hyperparameter search.
taskset -c 15-17 \
python3 src/train.py \
    -m \
    hydra/launcher=submitit_local \
    hydra.launcher.cpus_per_task=1 \
    hydra.launcher.gpus_per_node=4 \
    paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
    experiment=vicreg \
    experiment_name=vicreg_search \
    evaluator.ckpts.single=null \
    logger=none \
    hparams_search=vicreg_optuna \
    hydra.sweeper.study_name=knneff_vs_loss_b16k \
    hydra.sweeper.n_trials=60 \
    hydra.sweeper.sampler.n_startup_trials=50 \
    trainer=gpu \
    trainer.max_epochs=50 \
    trainer.devices=[2]
