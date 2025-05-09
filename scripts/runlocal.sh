python3 src/train.py \
    --multirun \
    experiment=qvae \
    trainer=gpu \
    trainer.max_epochs=1 \
    trainer.limit_train_batches=1.23e-5 \
    trainer.limit_val_batches=9.8261e-5 \
    trainer.limit_test_batches=1.46e-4 \
    data.batch_size=16384 \
    data.num_workers=64 \
    data.pin_memory=True \
