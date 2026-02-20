# Linear warmup scheduler.


class LinearWarmup:
    """Linear warmup of scalar value."""

    def __init__(self, final_value: float, warmup_frac: float, total_steps: int):
        self.final_value = float(final_value)
        self.warmup_frac = float(warmup_frac)
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(1, int(self.warmup_frac * total_steps))

    def __call__(self, step: int) -> float:
        if self.warmup_frac <= 0.0:
            return self.final_value
        if step >= self.warmup_steps:
            return self.final_value

        return self.final_value * (step / self.warmup_steps)
