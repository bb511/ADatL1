import torch

from src.algorithms.svdd import DeepSVDD
from src.data.utils import unpack_batch


class ImageDeepSVDD(DeepSVDD):
    """Deep SVDD on images of shape (batch, channels, height, width), unflattened."""

    def __init__(self, *args, target_rate: float = 0.01, **kwargs):
        super().__init__(*args, target_rate=target_rate, **kwargs)

    def _prepare_input(self, batch) -> torch.Tensor:
        return unpack_batch(batch).x
