# Encoder models.
from typing import Optional, Callable

import torch
import torch.nn as nn

from src.algorithms.components.mlp import MLP


class Encoder(nn.Module):
    """Simple vanilla encoder, i.e., just an MLP.

    :param in_dim: Integer for the input dimension to the encoder.
    :param nodes: List of ints, each int specifying the width of a layer, includes the
        output layer, i.e., the latent dimension.
    :param activation: Pytorch module that defines the activation function.
    :param init_weight: Callable method to initialize the weights of the encoder nodes.
    :param init_bias: Callable method to initialize the biases of the encoder nodes.
    """

    def __init__(
        self,
        in_dim: int,
        nodes: list[int],
        activation: str = "relu",
        init_weight: Optional[Callable] = None,
        init_bias: Optional[Callable] = None,
    ):
        super().__init__()

        self.net = MLP(
            in_dim=in_dim,
            nodes=nodes[:-1],
            out_dim=nodes[-1],
            batchnorm=False,
            activation=activation,
            final_activation=False,
            init_weight=init_weight,
            init_bias=init_bias,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return self.net(x)


class VariationalEncoder(nn.Module):
    """Simple variational encoder model.

    :param in_dim: Integer for the input dimension to the encoder.
    :param nodes: List of ints, each int specifying the width of a layer.
    :param activation: Pytorch module that defines the activation function.
    :param init_weight: Callable method to initialize the weights of the encoder nodes.
    :param init_bias: Callable method to initialize the biases of the encoder nodes.
    :param clamp_zlogvar_range: Tuple of two floats defining the range of the z_log_var
        value produced by the variational encoder.
    """

    def __init__(
        self,
        in_dim: int,
        nodes: list[int],
        activation: str = "relu",
        init_weight: Optional[Callable] = None,
        init_bias: Optional[Callable] = None,
        clamp_zlogvar_range: tuple[float, float] = (-20.0, 10.0),
    ):
        super().__init__()

        # The encoder will be a MLP up to the last layer
        self.net = MLP(
            in_dim=in_dim,
            nodes=nodes[:-2],
            out_dim=nodes[-2],
            batchnorm=False,
            activation=activation,
            final_activation=True,
            init_weight=init_weight,
            init_bias=init_bias,
        )

        # Mean and log variance layers
        self.z_mean = nn.Linear(nodes[-2], nodes[-1])
        self.z_log_var = nn.Linear(nodes[-2], nodes[-1])

        self.clamp_zlogvar_range = clamp_zlogvar_range
        if init_weight:
            init_weight(self.z_mean.weight)
            init_weight(self.z_log_var.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        x = self.net(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        z_log_var = z_log_var.clamp(*self.clamp_zlogvar_range)
        z = self.sample(z_mean, z_log_var)
        return z_mean, z_log_var, z

    def sample(self, z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
        """Performs reparametrization trick."""
        std = torch.exp(0.5 * z_log_var)
        epsilon = torch.randn_like(std)
        return z_mean + std * epsilon


class DeepSetsEncoder(nn.Module):
    """Per-type DeepSets encoder for sparse object data.

    This encoder applies a different shared object-wise network to each object type,
    pools the resulting object embeddings within each type, concatenates the pooled
    summaries across types, and maps the result to a global event representation.

    Expected input:
      - x_by_type[obj_name]: Tensor of shape [B, N_obj, F_obj]
      - m_by_type[obj_name]: Tensor of shape [B, N_obj], where 1 means real object and
        0 means padded object.

    :param object_dims: Dict mapping object type name to number of features per object.
    :param object_phi_nodes: Dict mapping object type name to hidden-layer structure of
        the object-wise encoder phi_t. The last entry is the output embedding size of
        phi_t.
    :param rho_nodes: Hidden-layer structure of the event-level network rho. The last
        entry is the output representation size of the encoder.
    :param activation: String specifying the activation used in all MLPs.
    :param pooling: String specifying pooling within object type. One of "mean", "sum",
        or "sum_max".
    :param add_counts: Bool whether to append the number of valid objects of each type
        to the pooled event representation.
    :param init_weight: Callable method to initialize the weights of the encoder nodes.
    :param init_bias: Callable method to initialize the biases of the encoder nodes.
    """

    def __init__(
        self,
        object_dims: dict[str, int],
        object_phi_nodes: dict[str, list[int]],
        rho_nodes: list[int],
        activation: str = "relu",
        pooling: str = "mean",
        add_counts: bool = False,
        init_weight: Optional[Callable] = None,
        init_bias: Optional[Callable] = None,
    ):
        super().__init__()

        if pooling not in {"mean", "sum", "sum_max"}:
            raise ValueError(f"Unsupported pooling '{pooling}'.")

        self.object_types = list(object_dims.keys())
        self.pooling = pooling
        self.add_counts = add_counts

        self.phis = nn.ModuleDict()
        pooled_dim = 0

        for obj_name in self.object_types:
            if obj_name not in object_phi_nodes:
                raise ValueError(f"Missing phi configuration for object type '{obj_name}'.")

            phi_nodes = object_phi_nodes[obj_name]
            if len(phi_nodes) < 1:
                raise ValueError(f"phi_nodes for '{obj_name}' must be non-empty.")

            self.phis[obj_name] = MLP(
                in_dim=object_dims[obj_name],
                nodes=phi_nodes[:-1],
                out_dim=phi_nodes[-1],
                batchnorm=False,
                activation=activation,
                final_activation=True,
                init_weight=init_weight,
                init_bias=init_bias,
            )

            phi_out_dim = phi_nodes[-1]
            pooled_dim += phi_out_dim if pooling in {"mean", "sum"} else 2 * phi_out_dim

        if add_counts:
            pooled_dim += len(self.object_types)

        self.rho = MLP(
            in_dim=pooled_dim,
            nodes=rho_nodes[:-1],
            out_dim=rho_nodes[-1],
            batchnorm=False,
            activation=activation,
            final_activation=True,
            init_weight=init_weight,
            init_bias=init_bias,
        )

    def encode_event(
        self,
        x_by_type: dict[str, torch.Tensor],
        m_by_type: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Encode an event into a global event representation."""
        pooled_parts = []

        for obj_name in self.object_types:
            x = x_by_type[obj_name]
            m = m_by_type[obj_name]

            bsz, n_obj, feat_dim = x.shape
            x_flat = x.reshape(bsz * n_obj, feat_dim)
            h_flat = self.phis[obj_name](x_flat)
            h = h_flat.reshape(bsz, n_obj, -1)

            m_float = m.unsqueeze(-1).float()
            h = h * m_float

            pooled = self._pool(h, m_float)
            pooled_parts.append(pooled)

            if self.add_counts:
                counts = m.float().sum(dim=1, keepdim=True)
                pooled_parts.append(counts)

        event_repr = torch.cat(pooled_parts, dim=1)
        return self.rho(event_repr)


    def forward(
        self,
        x_by_type: dict[str, torch.Tensor],
        m_by_type: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Encode a batch of events into a global event representation."""
        return self.encode_event(x_by_type, m_by_type)

    def _pool(self, h: torch.Tensor, m_float: torch.Tensor):
        """Pool the output of the phi networks."""
        if self.pooling == "sum":
            pooled = h.sum(dim=1)

        elif self.pooling == "mean":
            denom = m_float.sum(dim=1).clamp_min(1.0)
            pooled = h.sum(dim=1) / denom

        else:
            h_sum = h.sum(dim=1)
            h_max = h.masked_fill(m_float == 0, float("-inf")).max(dim=1).values
            h_max = torch.where(torch.isfinite(h_max), h_max, torch.zeros_like(h_max))
            pooled = torch.cat([h_sum, h_max], dim=1)

        return pooled

class DeepSetsVariationalEncoder(DeepSetsEncoder):
    """Per-type DeepSets variational encoder.

    The final dimension of rho_nodes is taken to be the latent dimension.

    :param clamp_zlogvar_range: Tuple of two floats defining the range of the z_log_var
        value produced by the variational encoder.
    """

    def __init__(
        self,
        object_dims: dict[str, int],
        object_phi_nodes: dict[str, list[int]],
        rho_nodes: list[int],
        activation: str = "relu",
        pooling: str = "mean",
        add_counts: bool = False,
        init_weight: Optional[Callable] = None,
        init_bias: Optional[Callable] = None,
        clamp_zlogvar_range: tuple[float, float] = (-10.0, 6.0),
    ):
        super().__init__(
            object_dims=object_dims,
            object_phi_nodes=object_phi_nodes,
            rho_nodes=rho_nodes,
            activation=activation,
            pooling=pooling,
            add_counts=add_counts,
            init_weight=init_weight,
            init_bias=init_bias,
        )

        latent_dim = rho_nodes[-1]
        self.z_mean = nn.Linear(latent_dim, latent_dim)
        self.z_log_var = nn.Linear(latent_dim, latent_dim)
        self.clamp_zlogvar_range = clamp_zlogvar_range

        if init_weight is not None:
            init_weight(self.z_mean.weight)
            init_weight(self.z_log_var.weight)
        if init_bias is not None:
            if self.z_mean.bias is not None:
                init_bias(self.z_mean.bias)
            if self.z_log_var.bias is not None:
                init_bias(self.z_log_var.bias)

    def forward(
        self,
        x_by_type: dict[str, torch.Tensor],
        m_by_type: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        event_repr = self.encode_event(x_by_type, m_by_type)

        z_mean = self.z_mean(event_repr)
        z_log_var = self.z_log_var(event_repr)
        z_log_var = z_log_var.clamp(*self.clamp_zlogvar_range)

        z = self.sample(z_mean, z_log_var)
        return z_mean, z_log_var, z

    def sample(self, z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
        """Performs reparametrization trick."""
        std = torch.exp(0.5 * z_log_var)
        epsilon = torch.randn_like(std)
        return z_mean + std * epsilon
