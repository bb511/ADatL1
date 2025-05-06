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
        self.lr0 = lr0
        self.s0 = s0
        self.t_mul = t_mul
        self.m_mul = m_mul
        self.alpha = alpha
        self.warmup_epochs = warmup_epochs

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Compute warmup ratio
        warmup_ratio = min((self.last_epoch + 1) / self.warmup_epochs, 1.0)

        # Compute cosine decay with restarts
        def cos_decay(step, initial_lr, decay_steps, t_mul, m_mul, alpha):
            step = min(step, decay_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * step / decay_steps))
            decayed_lr = (initial_lr - alpha) * cosine_decay + alpha
            return decayed_lr

        # Calculate learning rate
        current_step = self.last_epoch
        decay_steps = self.s0
        t = current_step

        # Implement cosine decay with restarts logic
        while t >= decay_steps:
            t -= decay_steps
            decay_steps *= self.t_mul
            self.lr0 *= self.m_mul

        lr = cos_decay(t, self.lr0, decay_steps, self.t_mul, self.m_mul, self.alpha)

        # Apply warmup
        lr *= warmup_ratio

        return [lr for _ in self.base_lrs]
