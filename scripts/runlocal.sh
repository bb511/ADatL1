python3 src/train.py \
    -m \
    experiment=debug \
    paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \

# python3 src/train.py \
#     -m \
#     experiment=vae \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     trainer.max_epochs=1 \
#     trainer=cpu \
#     data.batch_size=16276,32552,65104 \
#     algorithm.optimizer._target_=torch.optim.SGD,torch.optim.AdamW \
#     algorithm.optimizer.lr=1e-5,1e-4,1e-3 \
#     algorithm.optimizer.weight_decay=1e-5,1e-4,1e-3 \
#     algorithm.loss.scale=0.0 \
#     +trainer.limit_train_batches=10
    # algorithm.loss.scale=0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
