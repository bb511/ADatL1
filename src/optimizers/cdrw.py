import math
from torch.optim.lr_scheduler import _LRScheduler


class CDRW(_LRScheduler):
    def __init__(
        self,
        optimizer,
        lr0: float,
        s0: int,
        t_mul: float = 2.0,
        m_mul: float = 1.0,
        alpha: float = 0.0,
        warmup_epochs: int = 10,
        last_epoch: int = -1,
    ):
        """
        Cosine decay with restarts and linear warmup scheduler

        Args:
            optimizer (Optimizer): Wrapped optimizer
            lr0 (float): Initial learning rate
            s0 (int): Number of steps for the first decay
            t_mul (float, optional): Multiplier for decay steps. Defaults to 2.0.
            m_mul (float, optional): Multiplier for maximum learning rate. Defaults to 1.0.
            alpha (float, optional): Minimum learning rate value. Defaults to 0.0.
            warmup_epochs (int, optional): Number of warmup epochs. Defaults to 10.
            last_epoch (int, optional): The index of last epoch. Defaults to -1.
        """
        self.base_lr0 = lr0
        self.lr0 = lr0
        self.s0 = s0
        self.t_mul = t_mul
        self.m_mul = m_mul
        self.alpha = alpha
        self.warmup_epochs = warmup_epochs

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # step index (PyTorch uses last_epoch as the "step count" for most schedulers)
        step = max(self.last_epoch, 0)

        # --- Find current cycle (cycle_idx), cycle length (decay_steps), and position in cycle (t) ---
        decay_steps = self.s0
        t = step
        cycle_idx = 0

        # Walk forward through cycles until we find the active one
        while t >= decay_steps:
            t -= decay_steps
            decay_steps = int(round(decay_steps * self.t_mul))
            decay_steps = max(decay_steps, 1)
            cycle_idx += 1

        # --- Peak LR for this cycle (gamma-style decay of max LR) ---
        lr_max = self.base_lr0 * (self.m_mul ** cycle_idx)

        # --- Cosine decay within the cycle: lr_max -> alpha ---
        # Clamp t so we never exceed decay_steps
        t = min(t, decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * t / decay_steps))
        lr = (lr_max - self.alpha) * cosine + self.alpha

        # --- Warmup ---
        # Option A: global warmup
        warm = min((step + 1) / self.warmup_epochs, 1.0) if self.warmup_epochs > 0 else 1.0

        # Option B: per-cycle warmup (common for warm restarts)
        # warm = min((t + 1) / self.warmup_epochs, 1.0) if self.warmup_epochs > 0 else 1.0

        lr *= warm

        return [lr for _ in self.base_lrs]
