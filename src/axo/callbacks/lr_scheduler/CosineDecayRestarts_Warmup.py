import tensorflow as tf

class CosineDecayRestarts_Warmup(tf.keras.optimizers.schedules.CosineDecayRestarts):
    def __init__(self, initial_learning_rate, first_decay_steps, t_mul=2.0, m_mul=1.0, alpha=0.0, name=None, warmup_epochs=10):
        super().__init__(initial_learning_rate, first_decay_steps, t_mul, m_mul, alpha, name)
        self.warmup_epochs = warmup_epochs

    def __call__(self, epoch, logs=None):
        assert logs is not None
        ratio = min((epoch + 1) / self.warmup_epochs, 1)
        lr = super().__call__(epoch)
        return lr * ratio


