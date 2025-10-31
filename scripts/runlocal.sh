# python3 src/train.py \
#     -m \
#     experiment=debug \
#     trainer.max_epochs=100 \
#     model.optimizer.lr=0.001 \
#     data.val_batches=10 \
    # data.classifier_signals=[haa-4b-ma15] \
    # trainer=gpu0 \
    # model.loss.alpha=0.6 \
    # callbacks.cap.capmetric.device=cuda:1
    # ,0.00075,0.0005,0.00025,0.0001 \


# Debug local run.
python3 src/train.py \
    -m \
    experiment=debug \
    paths.raw_data_dir=/data/deodagiu/adl1t_data/parquet_files \
    trainer=gpu0 \
