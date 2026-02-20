from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
from pytorch_lightning.utilities.memory import garbage_collection_cuda

from src.algorithms import L1ADLightningModule


class SVDD(L1ADLightningModule):
    """Support Vector Data Description for anomaly detection."""

    def __init__(
        self,
        encoder: nn.Module,
        features: Optional[nn.Module] = None,
        center_init_method: str = "mean",  # or "zeros"
        **kwargs,
    ):
        super().__init__(model=encoder, **kwargs)
        self.save_hyperparameters(ignore=["model", "features", "encoder", "loss"])

        self.encoder = encoder
        self.features = features if features is not None else nn.Identity()
        self.features.eval()

        # Center will be initialized after first forward pass
        self.register_buffer("center", None)
        self.center_init_method = center_init_method
        self.center_initialized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.features(x))

    def _init_center(self, z: torch.Tensor):
        """Initialize the center of the hypersphere."""
        if self.center_init_method == "mean":
            self.center = z.detach().mean(dim=0)
        elif self.center_init_method == "zeros":
            self.center = torch.zeros(z.shape[1], device=z.device)
        else:
            raise ValueError(f"Unknown center init method: {self.center_init_method}")

        # Avoid collapse by ensuring center is not too close to zero
        eps = 0.1
        self.center[(torch.abs(self.center) < eps) & (self.center >= 0)] = eps
        self.center[(torch.abs(self.center) < eps) & (self.center < 0)] = -eps
        self.center_initialized = True

    def _compute_distance(self, z: torch.Tensor) -> torch.Tensor:
        """Compute distance from center for each sample."""
        return torch.sum((z - self.center) ** 2, dim=1)

    def model_step(self, batch: Tuple[torch.Tensor]) -> Dict[str, torch.Tensor]:
        x, _ = batch
        x = torch.flatten(x, start_dim=1)
        z = self.forward(x)
        del x

        # Initialize center on first batch if not done
        if not self.center_initialized:
            with torch.no_grad():
                self._init_center(z)

        # Compute distances (anomaly scores)
        distances = self._compute_distance(z)
        total_loss = self.loss(distances=distances, z=z)
        return {
            "loss": total_loss.mean(),
            "loss/total/full": total_loss.detach(),
            "loss/svdd/mean": total_loss.mean(),
            "loss/distance/mean": distances.mean(),
        }

    def outlog(self, outdict: dict) -> dict:
        return {
            "loss": outdict.get("loss"),
            "loss_svdd": outdict.get("loss/svdd/mean"),
            "loss_distance": outdict.get("loss/distance/mean"),
        }
