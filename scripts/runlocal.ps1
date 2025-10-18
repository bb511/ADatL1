python src/train.py `
    --multirun `
    experiment=axov4 `
    callbacks=default `
    trainer=cpu `
    trainer.max_epochs=1 `
    seed=0 `
    data.batch_size=100 `
    +trainer.limit_train_batches=1 `
    +trainer.limit_val_batches=1 `
    +trainer.limit_test_batches=1 `
    trainer.num_sanity_val_steps=0 `
    logger=none
    # callbacks.approximation_capacity.capmetric.n_epochs=5 `
    # callbacks.approximation_capacity.capmetric.lr=0.001

    # data.batch_size=16384 `
    #     trainer.limit_train_batches=1.23e-5 `
    # trainer.limit_val_batches=9.8261e-5 `
    # trainer.limit_test_batches=1.46e-4 `

    

