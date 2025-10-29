from typing import Literal

import torch
import torch.nn.functional as F

from src.losses import L1ADLoss


class AlignmentLoss(L1ADLoss):
    """
    Alignment loss to control separation between zerobias and simulation data.
    
    :param scale: Scaling factor for the loss.
    :param margin: Margin for distance-based strategies.
    :param distance: Distance metric to use ("l2", "l1", "cosine", "wasserstein").
    :param strategy: Strategy to use ("push_simulations", "cluster_zerobias", "contrastive").
    :param reduction: Reduction method to apply to the loss ("none", "mean", "sum").
    """

    name: str = "alignment"
    
    def __init__(
        self,
        scale: float = 1.0,
        margin: float = 1.0,
        distance: Literal["l2", "l1", "cosine", "wasserstein"] = "l2",
        strategy: Literal["push_simulations", "cluster_zerobias", "contrastive"] = "push_simulations",
        reduction: Literal["none", "mean", "sum"] = "mean",
    ):
        super().__init__(scale=scale, reduction=reduction)
        self.margin = margin
        self.distance = distance
        self.strategy = strategy
        
    def _compute_distance(self, z: torch.Tensor) -> torch.Tensor:
        """Compute distance/norm of latent codes."""
        if self.distance == "l2":
            return torch.norm(z, dim=1, p=2)
        elif self.distance == "l1":
            return torch.norm(z, dim=1, p=1)
        elif self.distance == "cosine":
            normalized = F.normalize(z, dim=1) # distance from unit sphere
            return 1 - torch.norm(normalized, dim=1)
        elif self.distance == "wasserstein":
            return torch.abs(z).sum(dim=1) # 1-Wasserstein distance to origin
        else:
            raise ValueError(f"Unknown distance: {self.distance}")
    
    def forward(
        self,
        z_mean: torch.Tensor,
        s: torch.LongTensor,
        **kwargs
    ) -> torch.Tensor:
        
        distances = self._compute_distance(z_mean)
        is_zerobias = s == 0 if s is not None else torch.ones(z_mean.shape[0], dtype=torch.bool, device=z_mean.device)
        is_simulation = s == 1 if s is not None else torch.ones(z_mean.shape[0], dtype=torch.bool, device=z_mean.device)
        loss = torch.zeros(z_mean.shape[0], device=z_mean.device)
        if self.strategy == "push_simulations":
            # Zerobias near origin, simulations far
            if is_zerobias.any():
                loss[is_zerobias] = distances[is_zerobias] ** 2
            if is_simulation.any():
                loss[is_simulation] = F.relu(self.margin - distances[is_simulation])
                
        elif self.strategy == "cluster_zerobias":
            # Minimize variance for zerobias, maximize for simulations
            if is_zerobias.any():
                zerobias_var = z_mean[is_zerobias].var(dim=0).sum()
                loss[is_zerobias] = zerobias_var
            if is_simulation.any():
                sim_var = z_mean[is_simulation].var(dim=0).sum()
                loss[is_simulation] = 1.0 / (sim_var + 1e-6)
                
        elif self.strategy == "contrastive":
            # Contrastive loss between zerobias and simulations
            if is_zerobias.any() and is_simulation.any():
                z_zb = z_mean[is_zerobias]
                z_sim = z_mean[is_simulation]
                
                dist_matrix = torch.cdist(z_zb, z_sim)
                contrastive_loss = F.relu(self.margin - dist_matrix.min(dim=1)[0])
                loss[is_zerobias] = contrastive_loss
                
        return self.scale * self.reduce(loss)