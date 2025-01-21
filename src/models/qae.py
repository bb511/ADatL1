import torch
import torch.nn as nn
import pytorch_lightning
from pytorch_lightning.core.optimizer import LightningOptimizer
from omegaconf import DictConfig, OmegaConf

class QAE(pytorch_lightning.LightningModule):
    def __init__(
        self,
        autoencoder: nn.Module,
        loss: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: DictConfig,
        # lr: float = 1e-3,
    ):
        super().__init__()
        self.model = autoencoder
        self.loss = loss
        self.save_hyperparameters(ignore=["autoencoder", "loss"])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def model_step(self, batch: torch.Tensor) -> torch.Tensor:
        import ipdb; ipdb.set_trace()
        return {
            "loss": self.loss(self(batch), batch)
        }
        
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        output = self.model_step(batch)

        loss = output.get("loss")
        self.log('train/loss', loss)
        return loss
        
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        output = self.model_step(batch)

        loss = output.get("loss")
        self.log('val/loss', loss)
        return loss
        
    def configure_optimizers(self):
        optimizer = LightningOptimizer(self.hparams.optimizer(params=self.parameters()))

        if self.hparams.scheduler:
            scheduler = self.hparams.scheduler.scheduler(optimizer=optimizer)

            scheduler_dict = OmegaConf.to_container(self.hparams.scheduler, resolve=True) # convert to normal dict
            scheduler_dict.update({
                    "scheduler": scheduler,
            })

            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler_dict
            }
        
        return {"optimizer": optimizer}