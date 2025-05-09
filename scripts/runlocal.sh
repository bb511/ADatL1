python3 src/train.py \
    --multirun \
    experiment=qvae \
    trainer=gpu \
    model.loss.alpha=0.1, 0.2, 0.3 \ # from 0.1 to 1
    model.optimizer.lr=0.01, 0.001, 0.0001, 0.0001 \
    seed=511, 6023 \
    trainer.max_epochs=100 \
    data.batch_size=16384 \
    data.num_workers=32 \
    data.pin_memory=True \
