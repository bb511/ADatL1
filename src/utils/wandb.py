from pytorch_lightning.loggers.wandb import WandbLogger

class SafeWandbLogger(WandbLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._wandb_init.update({
            "settings": {
                "_service_wait": 1200,
                "init_timeout": 1200
            }
        })
        