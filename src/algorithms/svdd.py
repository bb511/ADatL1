# Deep SVDD model implementation.
from typing import Optional, Tuple, Dict

import torch
from torch import nn

from src.algorithms import L1ADLightningModule


class DeepSVDD(L1ADLightningModule):
    """Deep Support Vector Data Description for anomaly detection.

    This model learns an encoder that maps normal examples into a compact region of
    latent space. A center c is initialized in that latent space and training pushes
    encoded samples towards it. The anomaly score is the squared Euclidean distance of
    a sample's latent representation to that center.

    In this implementation:
      - the input is flattened before being passed to the encoder,
      - an optional feature module can be applied before the encoder,
      - the center is initialized from the first batch using either the latent mean
        or the origin,
      - the loss is computed from the latent distances and optional additional losses.

    :param encoder: Encoder network mapping flattened inputs to latent representations.
    :param features: Optional feature module applied before the encoder.
    :param center_init_method: Strategy used to initialize the latent center.
        Supported values are "mean" and "zeros".
    :param ckpt: Optional checkpoint path used to restore weights before training.
    :param operational_rate: Operational trigger/output rate used to define the tail
        quantile for diagnostics.
    :param bc_rate: Bunch crossing rate used together with operational_rate to define
        the operational quantile.
    """

    def __init__(
        self,
        encoder: nn.Module,
        features: Optional[nn.Module] = None,
        center_init_method: str = "mean",
        ckpt: str = "",
        operational_rate: float = 0.25,
        bc_rate: float = 28608.8064,
        **kwargs,
    ):
        super().__init__(model=None, **kwargs)
        self.save_hyperparameters(ignore=["model", "features", "encoder", "loss"])

        self.encoder = encoder
        self.features = features if features is not None else nn.Identity()
        self.features.eval()

        self.center_init_method = center_init_method
        self.ckpt_path = ckpt
        self.center_initialized = False

        # The latent center is initialized after the first forward pass.
        self.register_buffer("center", None)

        self.operational_quantile = 1 - operational_rate / bc_rate

    def on_fit_start(self):
        """Move optional feature module to the correct device and restore checkpoint."""
        self.features.to(self.device)

        if self.ckpt_path:
            self._load_checkpoint()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of flattened inputs into latent representations."""
        x = self.features(x)
        return self.encoder(x)

    def _init_center(self, z: torch.Tensor):
        """Initialize the hypersphere center in latent space.

        If the center is initialized too close to zero in some dimensions, those
        coordinates are moved slightly away from zero to reduce the risk of trivial
        collapse.
        """
        if self.center_init_method == "mean":
            center = z.detach().mean(dim=0)
        elif self.center_init_method == "zeros":
            center = torch.zeros(z.shape[1], device=z.device, dtype=z.dtype)
        else:
            raise ValueError(f"Unknown center_init_method: {self.center_init_method}")

        # Avoid near-zero coordinates in the center.
        eps = 0.1
        center[(torch.abs(center) < eps) & (center >= 0)] = eps
        center[(torch.abs(center) < eps) & (center < 0)] = -eps

        self.center = center
        self.center_initialized = True

    def _compute_distance(self, z: torch.Tensor) -> torch.Tensor:
        """Compute squared Euclidean distance from the latent center for each sample."""
        return torch.sum((z - self.center) ** 2, dim=1)

    def model_step(self, batch: Tuple[torch.Tensor]) -> Dict[str, torch.Tensor]:
        x, _, _, _ = batch
        x = torch.flatten(x, start_dim=1)

        z = self.forward(x)

        # Initialize the center from the first batch if needed.
        if not self.center_initialized:
            with torch.no_grad():
                self._init_center(z)

        distances = self._compute_distance(z)
        distance_raw, reg_scaled, total_loss = self.loss(distances=distances, z=z)
        total_loss = self._add_hgq_loss(total_loss)

        with torch.no_grad():
            dist_quantiles = torch.tensor(
                [0.5, 0.95, 0.99, 0.999], device=distances.device
            )
            dist_q50, dist_q95, dist_q99, dist_q999 = torch.quantile(
                distances, dist_quantiles
            ).tolist()

            n = distances.numel()
            k = max(10, int((1.0 - self.operational_quantile) * n))
            topk_vals = torch.topk(distances, k).values
            mean_top_vals = topk_vals.mean().item()

            z_squared = torch.square(z).sum(dim=1).mean().item()
            center_norm = torch.square(self.center).sum().item()

        return {
            # Used for backpropagation:
            "loss": total_loss.mean(),
            # Used for logging:
            "loss/distance_raw/mean": distance_raw.detach().mean(),
            "loss/reg_scaled/mean": reg_scaled.detach().mean(),
            "loss/svdd/mean": total_loss.detach().mean(),
            "loss/distance/mean": distances.detach().mean(),
            "loss/distance/median": dist_q50,
            "loss/distance/q95": dist_q95,
            "loss/distance/q99": dist_q99,
            "loss/distance/q999": dist_q999,
            "loss/distance/mean_top_vals": mean_top_vals,
            "loss/z_squared": z_squared,
            "center_norm": center_norm,
            # Used for callbacks:
            "loss/total/full": total_loss.detach(),
            "loss/distance/full": distances.detach(),
            "encoded_data": z.detach(),
        }

    def outlog(self, outdict: dict) -> dict:
        return {
            "loss": outdict.get("loss"),
            "loss_distance_raw": outdict.get("loss/distance_raw/mean"),
            "loss_reg_scaled": outdict.get("loss/reg_scaled/mean"),
            "loss_svdd": outdict.get("loss/svdd/mean"),
            "loss_distance": outdict.get("loss/distance/mean"),
            "loss_distance_median": outdict.get("loss/distance/median"),
            "loss_distance_q95": outdict.get("loss/distance/q95"),
            "loss_distance_q99": outdict.get("loss/distance/q99"),
            "loss_distance_q999": outdict.get("loss/distance/q999"),
            "loss_distance_mean_top_vals": outdict.get("loss/distance/mean_top_vals"),
            "z_squared": outdict.get("loss/z_squared"),
            "center_norm": outdict.get("center_norm"),
        }

    def _load_checkpoint(self):
        """Load checkpoint weights to continue the training from, if provided."""
        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"]
        self.load_state_dict(state_dict, strict=False)

    def _add_hgq_loss(self, total_loss):
        """Add additional HGQ losses from the encoder if they exist."""
        add_loss = 0.0
        if hasattr(self.encoder, "losses") and len(self.encoder.losses) > 0:
            add_loss = add_loss + torch.stack([l for l in self.encoder.losses]).sum()

        total_loss = total_loss + add_loss
        return total_loss
