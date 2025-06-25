python3 src/train.py \
    experiment=axov4 \
    model.optimizer.lr=0.0001 \
    trainer.max_epochs=100 \
    data.use_entire_val_dataset=True \
    model.loss.alpha=0.1 \
    trainer=gpu0 \
