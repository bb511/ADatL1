python3 src/train.py \
    experiment=axov4 \
    model.optimizer.lr=0.001 \
    trainer.max_epochs=100 \
    data.batch_size=16384 \
    data.num_workers=1 \
    data.pin_memory=True \
    data.specialized_loader=True \
    data.device=cpu \
    # model.loss.alpha=1 \
    # logger.mlflow.run_name= \
