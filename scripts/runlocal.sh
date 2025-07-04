python3 src/train.py \
    -m \
    experiment=axov4 \
    trainer.max_epochs=100 \
    model.optimizer.lr=0.00025 \
    model.loss.alpha=0.7,0.8,0.9 \
    trainer=gpu3 \
    data.val_batches=1 \
