import math
from typing import Optional

import torch


class CosineWithWarmup(torch.optim.lr_scheduler.LambdaLR):
    """
    Step-based linear warmup -> cosine decay scheduler.

    Uses warmup_ratio in [0, 1) to derive warmup_steps from total_steps.
    Designed to be stepped every optimizer step (Lightning: interval="step").
    Returns a multiplicative factor applied to each param group's base LR.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_ratio: float,
        min_lr_ratio: float = 0.0,
        warmup_start_ratio: float = 1e-3,
        last_epoch: int = -1,
    ):
        assert total_steps > 0, "total_steps must be > 0"
        assert 0.0 <= warmup_ratio < 1.0, "warmup_ratio must be in [0, 1)"
        assert 0.0 <= min_lr_ratio < 1.0, "min_lr_ratio must be in [0, 1)"
        assert 0.0 <= warmup_start_ratio <= 1.0, "warmup_start_ratio must be in [0, 1]"

        self.total_steps = int(total_steps)
        self.warmup_ratio = float(warmup_ratio)
        self.min_lr_ratio = float(min_lr_ratio)
        self.warmup_start_ratio = float(warmup_start_ratio)

        # Derive warmup_steps and ensure there is at least 1 decay step
        self.warmup_steps = int(round(self.total_steps * self.warmup_ratio))
        self.warmup_steps = max(self.warmup_steps, 0)
        self.decay_steps = max(self.total_steps - self.warmup_steps, 1)

        super().__init__(optimizer, lr_lambda=self._lr_lambda, last_epoch=last_epoch)

    def _lr_lambda(self, step: int) -> float:
        # Clamp to avoid weirdness if scheduler is stepped beyond total_steps
        step = min(step, self.total_steps)

        # Warmup: linearly increase from warmup_start_ratio -> 1.0
        if self.warmup_steps > 0 and step < self.warmup_steps:
            progress = (step + 1) / self.warmup_steps  # ensures nonzero at step=0
            return self.warmup_start_ratio + progress * (1.0 - self.warmup_start_ratio)

        # Cosine decay from 1.0 -> min_lr_ratio over decay_steps
        t = min(step - self.warmup_steps, self.decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * t / self.decay_steps))

        return self.min_lr_ratio + cosine * (1.0 - self.min_lr_ratio)
