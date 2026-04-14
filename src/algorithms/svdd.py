from typing import Optional

import torch
from torch import nn

from src.algorithms import ADLightningModule
from src.algorithms.losses.svdd import SVDDLoss
from src.data.utils import unpack_batch


class DeepSVDD(ADLightningModule):
    """Deep SVDD module.

    Learns a compact latent representation of normal data. The anomaly score is the
    squared Euclidean distance to a learned center in latent space.

    :param encoder: Encoder network mapping flattened inputs to latent space.
    :param features: Optional feature module applied before the encoder.
    :param center_init_method: Strategy used to initialize the latent center.
        Supported values are "mean" and "zeros".
    :param ckpt: Optional checkpoint path used to restore weights before training.
    :param target_rate: Float specifying the target background rate or false positive rate.
    :param base_rate: Float specifying the base rate used to convert target_rate into
        an FPR. If None, target_rate is interpreted directly as an FPR.
    :param nu: Float soft-boundary hyperparameter with 0 < nu <= 1.
    :param weight_decay: Float L2 penalty applied to the latent representation in
        one-class mode.
    :param soft_boundary: Bool indicating whether to use the soft-boundary SVDD
        objective instead of the one-class formulation.
    """

    def __init__(
        self,
        encoder: nn.Module,
        features: Optional[nn.Module] = None,
        center_init_method: str = "mean",
        ckpt: str = "",
        target_rate: float = 0.25,
        base_rate: float | None = None,
        nu: float = 0.1,
        weight_decay: float = 1e-6,
        soft_boundary: bool = False,
        **kwargs,
    ):
        super().__init__(model=None, **kwargs)
        self.save_hyperparameters(ignore=["model", "features", "encoder"])

        self.encoder = encoder
        self.features = features if features is not None else nn.Identity()
        self.features.eval()

        self.center_init_method = center_init_method
        self.ckpt_path = ckpt
        self.center_initialized = False

        self.register_buffer("center", None)
        self.loss = SVDDLoss(
            nu=nu,
            weight_decay=weight_decay,
            soft_boundary=soft_boundary,
            reduction="none",
        )

    def on_fit_start(self):
        self.features.to(self.device)

        if self.ckpt_path:
            self._load_checkpoint()

    @property
    def target_fpr(self) -> float:
        return self.compute_target_fpr()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.encoder(x)

    def _init_center(self, z: torch.Tensor):
        if self.center_init_method == "mean":
            center = z.detach().mean(dim=0)
        elif self.center_init_method == "zeros":
            center = torch.zeros(z.shape[1], device=z.device, dtype=z.dtype)
        else:
            raise ValueError(f"Unknown center_init_method: {self.center_init_method}")

        eps = 0.1
        center[(torch.abs(center) < eps) & (center >= 0)] = eps
        center[(torch.abs(center) < eps) & (center < 0)] = -eps

        self.center = center
        self.center_initialized = True

    def _compute_distance(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sum((z - self.center) ** 2, dim=1)

    def model_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        b = unpack_batch(batch)
        x = torch.flatten(b.x, start_dim=1)

        z = self.forward(x)

        if not self.center_initialized:
            with torch.no_grad():
                self._init_center(z)

        ascore = self._compute_distance(z)

        if ascore.ndim != 1:
            raise ValueError(
                f"Expected per-event ascores, got {tuple(ascore.shape)}."
            )

        distance_raw, reg_scaled, loss = self.loss(distances=ascore, z=z)
        loss = self._add_hgq_loss(loss)

        with torch.no_grad():
            n = ascore.numel()
            k = max(1, int(self.target_fpr * n))

            if k < 10:
                k_eff = min(max(10, k), n)
                operational_ascore = torch.topk(ascore, k_eff).values.mean().item()
            else:
                operational_ascore = torch.quantile(
                    ascore, 1.0 - self.target_fpr
                ).item()

            q50, q99 = torch.quantile(
                ascore, torch.tensor([0.5, 0.99], device=ascore.device)
            ).tolist()

            z_squared = torch.square(z).sum(dim=1).mean().item()
            center_norm = torch.square(self.center).sum().item()

        loss_mean = loss.mean()

        return {
            # Backprop
            "loss": loss_mean,
            # Logging
            "loss/mean": loss_mean,
            "loss/distance_raw/mean": distance_raw.mean(),
            "loss/reg_scaled/mean": reg_scaled.mean(),
            "ascore/operational": operational_ascore,
            "ascore/q50": q50,
            "ascore/q99": q99,
            "z_squared": z_squared,
            "center_norm": center_norm,
            # Callbacks
            "loss/full": loss.detach(),
            "ascore/full": ascore.detach(),
            "encoded_data": z.detach(),
        }

    def outlog(self, outdict: dict) -> dict:
        return {
            "loss": outdict.get("loss"),
            "loss_mean": outdict.get("loss/mean"),
            "loss_distance_raw": outdict.get("loss/distance_raw/mean"),
            "loss_reg_scaled": outdict.get("loss/reg_scaled/mean"),
            "ascore_operational": outdict.get("ascore/operational"),
            "ascore_q50": outdict.get("ascore/q50"),
            "ascore_q99": outdict.get("ascore/q99"),
            "z_squared": outdict.get("z_squared"),
            "center_norm": outdict.get("center_norm"),
        }

    def _load_checkpoint(self):
        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"]
        self.load_state_dict(state_dict, strict=False)

    def _add_hgq_loss(self, loss: torch.Tensor) -> torch.Tensor:
        add_loss = 0.0
        if hasattr(self.encoder, "losses") and len(self.encoder.losses) > 0:
            add_loss = add_loss + torch.stack([l for l in self.encoder.losses]).sum()

        return loss + add_loss
