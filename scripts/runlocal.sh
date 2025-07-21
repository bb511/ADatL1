python3 src/train.py \
    -m \
    experiment=axov4 \
    trainer.max_epochs=100 \
    model.optimizer.lr=0.001 \
    model.loss.alpha=0.7,0.9 \
    trainer=gpu0 \
    data.val_batches=1 \
    # callbacks.cap.capmetric.device=cuda:3
