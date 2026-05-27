python3 src/train.py \
    paths.raw_data_dir=../../03_Data/adl1t_data/parquet_files \
    experiment=physics/mi_aware_ae \
    run_name="MI_Aware_AE_Run_2_Gamma_5e-06_sum" \
    algorithm.optimizer.lr=0.0019859329798336714 \
    algorithm.delta=10.0 \
    trainer.gradient_clip_val=5.0 \
    algorithm.optimizer.betas='[0.9,0.999]' \
    algorithm.optimizer.weight_decay=1e-06 \
    algorithm.encoder.nodes='[64,32,8]' \
    algorithm.input_noise_std=0.0 \
    algorithm.use_bernoulli_bottleneck=true \
    algorithm.gamma=5e-06 \
    algorithm.bottleneck_temperature=1.0 \
    algorithm.deterministic_eval=true \
    trainer.max_epochs=100 \
    trainer=gpu \
    trainer.devices='[0]'

