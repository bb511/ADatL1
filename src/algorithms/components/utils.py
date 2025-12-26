# Utilities for the components of the algorithms.
import torch


class RandomNumberGenerator:
    """Random number generator implementation.

    Used primarly in the augmentation.py script to seed the augmenter layers in
    different ways, such that two samples are not augmented in the same way.
    """

    def __init__(self):
        self._seed: int | None = None
        self._gen: torch.Generator | None = None
        self._gen_device: torch.device | None = None

    def set_seed(self, seed: int):
        self._seed = int(seed)
        self._gen = None
        self._gen_device = None

    def get_generator(self, device: torch.device) -> torch.Generator:
        if self._seed is None:
            raise RuntimeError("RNG seed not set.")

        if self._gen is None or self._gen_device != device:
            self._gen = torch.Generator(device=device)
            self._gen.manual_seed(self._seed)
            self._gen_device = device

        return self._gen


class LinearWarmup:
    """Linear warmup of scalar value."""
    def __init__(self, final_value: float, warmup_frac: float, total_steps: int):
        self.final_value = float(final_value)
        self.warmup_frac = float(warmup_frac)
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(1, int(self.warmup_frac*total_steps))

    def __call__(self, step: int) -> float:
        if self.warmup_frac <= 0.0:
            return self.final_value
        if step >= self.warmup_steps:
            return self.final_value

        return self.final_value * (step / self.warmup_steps)
