python3 src/train.py \
    -m \
    experiment=debug \
    paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
    trainer=gpu0 \

# python3 src/train.py \
#     -m \
#     experiment=axov4 \
#     paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
#     trainer.max_epochs=3 \
#     trainer=gpu0 \
