python3 src/train.py \
    -m \
    experiment=axov4 \
    trainer.max_epochs=100 \
    model.optimizer.lr=0.00025 \
    model.loss.alpha=0.9 \
    trainer=gpu0 \
