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
