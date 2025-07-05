from pytorch_lightning.callbacks import Callback

class DebugCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        print(f"[DEBUG] Epoch: {trainer.current_epoch}")
        print(f"[DEBUG] Global Step: {trainer.global_step}")
        print(f"[DEBUG] Callback Metrics: {trainer.callback_metrics}")
