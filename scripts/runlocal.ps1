python src/train.py `
    --multirun `
    experiment=qvae `
    trainer=cpu `
    trainer.max_epochs=1 `
    model.loss.alpha=0.1 `
    model.optimizer.lr=0.01 `
    seed=511 `
    trainer.limit_train_batches=1.23e-4 `
    trainer.limit_val_batches=9.8261e-4 `
    trainer.limit_test_batches=1.46e-3 `
    +trainer.num_sanity_val_steps=0 `
    data.num_workers=0 `
    callbacks.approximation_capacity.n_epochs=5 `
    callbacks.approximation_capacity.lr=0.001

    # data.batch_size=16384 `
    #     trainer.limit_train_batches=1.23e-5 `
    # trainer.limit_val_batches=9.8261e-5 `
    # trainer.limit_test_batches=1.46e-4 `

    

