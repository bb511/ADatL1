python3 src/train.py \
    experiment=axov4 \
    model.optimizer.lr=0.001 \
    model.loss.alpha=0.5 \
    trainer.max_epochs=5 \
    trainer=gpu0 \
    data.use_entire_val_dataset=True \
    data.device=cpu \
    # logger.mlflow.run_name= \
    --multirun \
    experiment=vicreg \
    trainer=cpu \
    trainer.max_epochs=1 \
    data.batch_size=16384 \
    data.num_workers=50 \
    # trainer.limit_train_batches=1.23e-5 \
    # trainer.limit_val_batches=9.8261e-5 \
    # trainer.limit_test_batches=1.46e-4 \
