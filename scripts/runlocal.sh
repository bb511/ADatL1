python3 src/train.py \
    -m \
    experiment=axov4 \
    trainer.max_epochs=100 \
    model.optimizer.lr=0.0001 \
    model.loss.alpha=0.5 \
    trainer=gpu3 \
    data.val_batches=1 \
