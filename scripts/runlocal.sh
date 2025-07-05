python3 src/train.py \
    -m \
    experiment=axov4 \
    trainer.max_epochs=3 \
    model.optimizer.lr=0.001 \
    model.loss.alpha=0.6 \
    trainer=gpu0 \
    data.val_batches=1 \
