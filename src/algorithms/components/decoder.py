# Decoder models.
from typing import Optional, List, Callable

import torch

from src.algorithms.components.mlp import MLP


class Decoder(MLP):
    """Simple decoder model.

    :param nodes: List of ints, each int specifying the width of a layer.
    :param init_weight: Callable method to initialize the weights of the decoder nodes.
    :param init_bias: Callable method to initialize the biases of the decoder nodes.
    :param init_last_weight: Callable method to initialize the weights of the last layer.
    :param init_last_bias: Callable method to initialize the biases of the last layer.
    :param batchnorm: Whether to use batch normalization or not.
    """

    def __init__(
        self,
        nodes: List[int],
        init_weight: Optional[Callable] = None,
        init_bias: Optional[Callable] = None,
        batchnorm: bool = False,
        init_last_weight: Optional[Callable] = None,
        init_last_bias: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            nodes,
            batchnorm=batchnorm,
            init_weight=init_weight,
            init_bias=init_bias,
        )
        
        # Apply initialization to the weight of the last Linear() layer
        last_linear_layer = self.net[-1]
        if init_last_weight != None:
            init_last_weight(last_linear_layer.weight)
            if last_linear_layer.bias != None and init_last_bias != None:
                init_last_bias(last_linear_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
