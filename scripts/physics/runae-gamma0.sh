python3 src/train.py \
    paths.raw_data_dir=../../03_Data/adl1t_data/parquet_files \
    experiment=physics/ae \
    run_name="Bernoulli_MI_AE_Ep_10_Gamma_0.0_Run_1" \
    logger=mlflow \
    algorithm.optimizer.lr=0.0019859329798336714 \
    algorithm.delta=1.0 \
    algorithm.mi_gamma=0.0 \
    algorithm.mi_temperature=6.0 \
    trainer.gradient_clip_val=5.0 \
    algorithm.optimizer.betas='[0.9,0.999]' \
    algorithm.optimizer.weight_decay=1e-06 \
    algorithm.encoder.nodes='[64,32,8]' \
    algorithm.input_noise_std=0.0 \
    trainer.max_epochs=10 \
    trainer=gpu \
    trainer.devices='[0]'