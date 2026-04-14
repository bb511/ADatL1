from typing import Optional

import torch
from torch import nn

from src.algorithms import ADLightningModule
from src.data.utils import unpack_batch


class RealNVP(ADLightningModule):
    """RealNVP on flattened input data.

    The anomaly score is the negative log-likelihood (NLL) under the flow.

    :param flow: RealNVP module operating on flattened inputs.
    :param features: Optional feature module applied before the flow.
    :param ckpt: Optional checkpoint path for restarting training.
    :param target_rate: Target background rate or FPR.
    :param base_rate: Base rate used to convert target_rate into an FPR. If None,
        target_rate is interpreted directly as an FPR.
    """

    def __init__(
        self,
        flow: nn.Module,
        features: Optional[nn.Module] = None,
        ckpt: str = "",
        target_rate: float = 0.25,
        base_rate: float | None = None,
        **kwargs,
    ):
        super().__init__(model=None, **kwargs)
        self.save_hyperparameters(
            ignore=["model", "features", "flow"]
        )

        self.flow = flow
        self.features = features if features is not None else nn.Identity()
        self.features.eval()
        self.ckpt_path = ckpt

    def on_fit_start(self):
        """Move features to device and optionally load checkpoint."""
        self.features.to(self.device)

        if self.ckpt_path:
            self._load_checkpoint()

    @property
    def target_fpr(self) -> float:
        return self.compute_target_fpr()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the flow and return the log-probability of the input."""
        x = self.features(x)
        return self.flow.log_prob(x)

    def model_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        """Shared train/val/test step."""
        b = unpack_batch(batch)
        x = torch.flatten(b.x, start_dim=1)

        log_prob = self.forward(x)
        ascore = -log_prob

        if ascore.ndim != 1:
            raise ValueError(
                f"Expected per-event ascores, got {tuple(ascore.shape)}."
            )

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

        loss_mean = ascore.mean()

        return {
            # Used for backpropagation:
            "loss": loss_mean,
            # Used for logging:
            "loss/mean": loss_mean,
            "ascore/operational": operational_ascore,
            "ascore/q50": q50,
            "ascore/q99": q99,
            # Used for callbacks:
            "loss/full": ascore.detach(),
            "ascore/full": ascore.detach(),
        }

    def outlog(self, outdict: dict) -> dict:
        """Values logged in the standard training loop."""
        return {
            "loss": outdict.get("loss"),
            "loss_mean": outdict.get("loss/mean"),
            "ascore_operational": outdict.get("ascore/operational"),
            "ascore_q50": outdict.get("ascore/q50"),
            "ascore_q99": outdict.get("ascore/q99"),
        }

    def _load_checkpoint(self):
        """Load a previously saved checkpoint."""
        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"]
        self.load_state_dict(state_dict, strict=False)
