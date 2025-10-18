from typing import Optional, Tuple, Dict

import torch
from torch import nn

from src.algorithms import L1ADLightningModule


class Classifier(L1ADLightningModule):
    def __init__(
        self,
        features: Optional[nn.Module] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["model", "features", "loss"])
        self.features = features if features is not None else nn.Identity()
        self.features.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def model_step(self, batch: Tuple[torch.Tensor]) -> Dict[str, torch.Tensor]:
        x, s = batch
        logits = self.forward(x)
        loss = self.loss(
            input=logits.flatten(),
            target=s.to(logits.dtype),
        )
        return {
            "loss": loss.mean(),
            "score": torch.sigmoid(logits.detach()).flatten(),
        }
    
    def outlog(self, outdict: dict) -> dict:
        """Override with the values you want to log."""
        return {
            "loss/total": outdict.get("loss"),
        }
