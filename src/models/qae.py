import torch
from src.models import L1ADLightningModule

class QAE(L1ADLightningModule):
    
    def model_step(self, batch: torch.Tensor) -> torch.Tensor:
        import ipdb; ipdb.set_trace()
        return {
            "loss": self.loss(self(batch), batch)
        }