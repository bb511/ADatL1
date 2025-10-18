from typing import Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryCrossEntropy(nn.Module):
    """Reconstruction loss for VAE without reduction."""

    def __init__(self, weight: float = 1.0, reduction: str = "none"):
        super().__init__()
        self.reduction = reduction
        self.weight = torch.tensor(weight, dtype=torch.float)
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=self.weight, reduction=reduction
        )
        self.name = "bce"

    def forward(self, score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        bce_loss = self.criterion(score, label)

        return bce_loss
