# Short debug script.

taskset -c 0-2 \
python3 src/train.py \
    paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
    experiment=physics/ae \
    experiment_name=debug \
    algorithm.optimizer.lr=0.001457369500608365 \
    algorithm.loss.delta=7.0 \
    trainer.gradient_clip_val=0.5 \
    algorithm.optimizer.betas='[0.9,0.999]' \
    algorithm.optimizer.weight_decay=0.0 \
    algorithm.encoder.nodes='[64,32,32]' \
    algorithm.input_noise_std=0.001 \
    trainer.max_epochs=1 \
    trainer=gpu \
    trainer.devices=[0]


# CAP training.
# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=ae_agnostic \
#     experiment_name=debug \
#     callbacks.wasserstein_dist=null \
#     callbacks.thres_drift=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.thres_drift_q99_ema_ckpt=null \
#     callbacks.stable_mse_q99_ckpt=null \
#     ~evaluator.ckpts.single.loss_mse_q99 \
#     ~evaluator.ckpts.summary.trate286_0kHz_drift_ema \
#     ~evaluator.ckpts.summary.trate0_25kHz_drift_ema \
#     ~evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.optimizer.lr=0.0027927024120831816 \
#     algorithm.loss.delta=10.0 \
#     trainer.gradient_clip_val=2.0 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.encoder.nodes='[64,32,16]' \
#     algorithm.input_noise_std=0.01 \
#     trainer.max_epochs=2 \
#     trainer=gpu \
#     trainer.devices=[1]

#     callbacks.stable_mse_mean_top_ckpt=null \
#     ~evaluator.ckpts.single.loss_mse_mean_top_vals \


# Agnostic stability training.
# taskset -c 3-5 \
# python3 src/train.py \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=ae_agnostic \
#     experiment_name=debug \
#     callbacks.wasserstein_dist=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.wasserstein_dist_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.thres_drift_q99_ema_ckpt=null \
    # callbacks.stable_mse_q99_ckpt=null \
    # ~evaluator.ckpts.single.loss_mse_q99 \
#     ~evaluator.ckpts.summary.trate286_0kHz_drift_ema \
#     ~evaluator.ckpts.summary.w1dist_ema_normal_vs_SingleNeutrino_E-10-gun \
#     ~evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.optimizer.lr=0.000814981343573229 \
#     algorithm.loss.delta=10.0 \
#     trainer.gradient_clip_val=1.0 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.encoder.nodes='[64,32,32]' \
#     algorithm.input_noise_std=0.0003 \
#     trainer=gpu \
#     trainer.devices=[2]

#     callbacks.stable_mse_mean_top_ckpt=null \
#     ~evaluator.ckpts.single.loss_mse_mean_top_vals \


# Agnostic KL-wasserstein training.
# taskset -c 12-14 \
# python3 src/train.py \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     experiment=ae_agnostic \
#     experiment_name=debug \
#     callbacks.thres_drift=null \
#     callbacks.cap_sn_zb=null \
#     callbacks.thres_drift_ema_ckpt=null \
#     callbacks.cap_sn_zb_ema_ckpt=null \
#     callbacks.thres_drift_q99_ema_ckpt=null \
#     ~evaluator.ckpts.single.loss_mse_q99 \
#     callbacks.stable_mse_q99_ckpt=null \
#     ~evaluator.ckpts.summary.trate286_0kHz_drift_ema \
#     ~evaluator.ckpts.summary.trate0_25kHz_drift_ema \
#     ~evaluator.ckpts.summary.cap_ema_normal_vs_SingleNeutrino_E-10-gun \
#     algorithm.optimizer.lr=0.00047124714609726086 \
#     algorithm.loss.delta=5.0 \
#     trainer.gradient_clip_val=0.5 \
#     algorithm.optimizer.betas='[0.9,0.99]' \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.encoder.nodes='[64,32,32]' \
#     algorithm.input_noise_std=0.001 \
#     trainer=gpu \
#     trainer.devices=[3]

#     callbacks.stable_mse_mean_top_ckpt=null \
#     ~evaluator.ckpts.single.loss_mse_mean_top_vals \
