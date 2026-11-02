from src.algorithms.svdd import DeepSVDD


class ImageDeepSVDD(DeepSVDD):
    """Deep SVDD on images of shape (batch, channels, height, width), unflattened."""

    _flat_input = False

    def __init__(self, *args, target_rate: float = 0.01, **kwargs):
        super().__init__(*args, target_rate=target_rate, **kwargs)
