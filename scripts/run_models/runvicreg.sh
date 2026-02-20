# VICreg running commands.
# ========================================================================

# MODEL TRAINING.
# ========================
# # vicreg training.
# taskset -c 3-5 \
# python3 src/train.py \
#     -m \
#     experiment=vicreg \
#     run_name=best_model_5 \
#     trainer.gradient_clip_val=5.0 \
#     algorithm.optimizer.lr=0.0002683160957865875 \
#     algorithm.optimizer.weight_decay=5.707010619612e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.scheduler.scheduler.warmup_ratio=0.0 \
#     algorithm.scheduler.scheduler.min_lr_ratio=0.0 \
#     algorithm.feature_blur.prob=0.39921015566406076 \
#     algorithm.feature_blur.magnitude=0.48150039987848725 \
#     algorithm.feature_blur.strength=0.33584271293635204 \
#     algorithm.object_mask.prob=0.029803702188932333 \
#     algorithm.lorentz_rotation.prob=0.3 \
#     algorithm.loss.inv_coef=20.557268907601713 \
#     algorithm.loss.rvar_coef=2.358576838014784 \
#     algorithm.loss.rcov_coef=1.3355427472086183 \
#     algorithm.projector.nodes='[128, 128]' \
#     algorithm.projector.out_dim=32 \
#     algorithm.diagnosis_metrics=true \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     trainer=gpu \
#     trainer.devices=[0]

# vicreg train quantised model starting from floating point model.
# taskset -c 6-8 \
# python3 src/train.py \
#     -m \
#     experiment=qvicreg \
#     run_name=best_model_5 \
#     trainer.gradient_clip_val=5.0 \
#     algorithm.optimizer.lr=0.000134158047893 \
#     algorithm.optimizer.weight_decay=5.707010619612e-05 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.scheduler.scheduler.warmup_ratio=0.0 \
#     algorithm.scheduler.scheduler.min_lr_ratio=0.0 \
#     algorithm.feature_blur.prob=0.39921015566406076 \
#     algorithm.feature_blur.magnitude=0.48150039987848725 \
#     algorithm.feature_blur.strength=0.33584271293635204 \
#     algorithm.object_mask.prob=0.029803702188932333 \
#     algorithm.lorentz_rotation.prob=0.3 \
#     algorithm.loss.inv_coef=20.557268907601713 \
#     algorithm.loss.rvar_coef=2.358576838014784 \
#     algorithm.loss.rcov_coef=1.3355427472086183 \
#     algorithm.projector.nodes='[128, 128]' \
#     algorithm.projector.out_dim=32 \
#     algorithm.diagnosis_metrics=true \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     "algorithm.ckpt='/data/deodagiu/adl1t/checkpoints/vicreg_best_models/best_model_5/single/knn_auprc/max/ds=haa-4b-ma15__metric=knn_auprc__value=0.459602__epoch=48.ckpt'" \
#     trainer=gpu \
#     trainer.devices=[0]

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
