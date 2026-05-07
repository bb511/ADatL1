# Short debug script.

# taskset -c 0-2 \
# python3 src/train.py \
#     paths.raw_data_dir=/path/to/adl1t_data/parquet_files \
#     experiment=physics/ae \
#     experiment_name=debug \
#     algorithm.optimizer.lr=0.001457369500608365 \
#     algorithm.loss.delta=7.0 \
#     trainer.gradient_clip_val=0.5 \
#     algorithm.optimizer.betas='[0.9,0.999]' \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.encoder.nodes='[64,32,32]' \
#     algorithm.input_noise_std=0.001 \
#     trainer.max_epochs=1 \
#     trainer=gpu \
#     trainer.devices=[0]

