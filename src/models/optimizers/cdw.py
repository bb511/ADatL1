from typing import Optional
import torch
import math


class CosineWithWarmup(torch.optim.lr_scheduler.LambdaLR):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            warmup_epochs: int,
            decay_epochs: int,
            max_lr: Optional[float] = 1.0
        ):
        self.max_lr = max_lr
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        super().__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, epoch):
        if epoch < self.warmup_epochs:
            return epoch / self.warmup_epochs  # Linear warmup (0 → 1)
        else:
            cosine_decay = 0.5 * (1 + math.cos(math.pi * (epoch - self.warmup_epochs) / self.decay_epochs))
            return cosine_decay  # Cosine decay (1 → 0)