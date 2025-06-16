python3 src/train.py \
    experiment=axov4 \
    model.optimizer.lr=0.001 \
    model.loss.alpha=0.5 \
    trainer.max_epochs=5 \
    data.specialized_loader=True \
    data.device=cpu \
    trainer=gpu0 \
    data.use_entire_val_dataset=True \
    # logger.mlflow.run_name= \
