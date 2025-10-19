python src/eval.py `
    --multirun `
    experiment=axov4 `
    trainer=cpu `
    seed=0 `
    data.batch_size=100 `
    +trainer.limit_train_batches=1 `
    +trainer.limit_val_batches=1 `
    +trainer.limit_test_batches=1 `
    trainer.num_sanity_val_steps=0 `
    scan.alpha=0.1 `
    scan.lr=0.0005 `
    logger=none
