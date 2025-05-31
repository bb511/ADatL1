python3 src/train.py \
    experiment=qvae \
    trainer=gpu \
    model.loss.alpha=0.5736625791372814 \
    model.optimizer.lr=0.001 \
    seed=123 \
    trainer.max_epochs=480 \
    data.batch_size=16384 \
    data.num_workers=8 \
    data.pin_memory=True \
    logger.mlflow.run_name=test
