from typing import Optional, Tuple, List, Callable

import torch
import torch.nn as nn

from src.models.components.mlp import MLP


class VariationalEncoder(nn.Module):
    """Simple variational encoder model.

    :param nodes: List of ints, each int specifying the width of a layer.
    :param init_weight: Callable method to initialize the weights of the encoder nodes.
    :param init_bias: Callable method to initialize the biases of the encoder nodes.
    """

    def __init__(
        self,
        nodes: List[int],
        init_weight: Optional[Callable] = None,
        init_bias: Optional[Callable] = None,
    ):
        super().__init__()

        # The encoder will be a MLP up to the last layer
        self.net = MLP(
            nodes[:-1],
            batchnorm=False,
            init_weight=init_weight,
            init_bias=init_bias
        )

        # Mean and log variance layers
        self.z_mean = nn.Linear(nodes[-2], nodes[-1])
        init_weight(self.z_mean.weight)
        self.z_log_var = nn.Linear(nodes[-2], nodes[-1])
        init_weight(self.z_log_var.weight)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        x = self.net(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        z = self.sample(z_mean, z_log_var)
        return z_mean, z_log_var, z

    def sample(self, z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
        """Performs reparametrization trick."""
        std = torch.exp(0.5 * z_log_var)
        epsilon = torch.randn_like(std)
        return z_mean + std * epsilon
    


from src.models.quantization import Quantizer
from src.models.quantization.layers import QuantizedLinear
from src.models.components.mlp import QMLP


class QuantizedVariationalEncoder(VariationalEncoder):
    """
    Quantized variational encoder model.

    :param nodes: List of ints, each int specifying the width of a layer.
    :param qdata: Quantizer for the input data.
    :param qweight: Quantizer for the weight parameters.
    :param qbias: Quantizer for the bias parameters.
    :param qactivation: Quantizer for the activation output.
    :param init_weight: Callable method to initialize the weights of the encoder nodes.
    :param init_bias: Callable method to initialize the biases of the encoder nodes.
    """
    def __init__(
        self,
        nodes: List[int],
        qdata: Optional[Quantizer] = None,
        qweight: Optional[Quantizer] = None,
        qbias: Optional[Quantizer] = None,
        qactivation: Optional[Quantizer] = None,
        init_weight: Optional[Callable] = None,
        init_bias: Optional[Callable] = None,
    ):
        nn.Module.__init__(self)

        # Input data quantization
        self.qdata = qdata or Quantizer(None, None)

        # The encoder will be a QMLP up to the last layer
        qmlp = QMLP(
            nodes,
            qweight=qweight,
            qbias=qbias,
            qactivation=qactivation,
            batchnorm=False,
            init_weight=init_weight,
            init_bias=init_bias,
        )
        self.net = nn.Sequential(*list(qmlp.net.children())[:-1])

        # Mean and log variance layers
        self.z_mean = QuantizedLinear(
            in_features=nodes[-2],
            out_features=nodes[-1],
            qweight=qweight,
            qbias=qbias,
            qactivation=qactivation,
            init_weight=nn.init.xavier_uniform_,
            init_bias=nn.init.zeros_,
        )

        # Log variance layer (initialized to zero)
        self.z_log_var = QuantizedLinear(
            in_features=nodes[-2],
            out_features=nodes[-1],
            qweight=qweight,
            qbias=qbias,
            qactivation=qactivation,
            init_weight=nn.init.xavier_uniform_,
            init_bias=nn.init.zeros_,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return super().forward(self.qdata(x))