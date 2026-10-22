from typing import Optional

import torch
from torch import nn

from src.algorithms import ADLightningModule
from src.data.utils import unpack_batch


def _present(mask: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor:
    """The batch's data mask as 1 (data) and 0 (padding), flattened like x."""
    if mask is None:
        raise ValueError("mask_aware=True needs batches that carry the data mask.")
    return torch.flatten(mask, start_dim=1).to(x.dtype)


class RealNVP(ADLightningModule):
    """RealNVP on flattened input data.

    The anomaly score is the negative log-likelihood (NLL) under the flow.

    :param flow: RealNVP module operating on flattened inputs.
    :param features: Optional feature module applied before the flow.
    :param ckpt: Optional checkpoint path for restarting training.
    :param target_rate: Target background rate or FPR.
    :param base_rate: Base rate used to convert target_rate into an FPR. If None,
        target_rate is interpreted directly as an FPR.
    :param mask_aware: Whether the coordinates that the batch's data mask marks as
        padding are left out of the NLL, which is then the NLL of the present
        coordinates alone. Only the L1AD batches carry a mask.
    """

    def __init__(
        self,
        flow: nn.Module,
        features: Optional[nn.Module] = None,
        ckpt: str = "",
        target_rate: float = 0.25,
        base_rate: float | None = None,
        mask_aware: bool = False,
        **kwargs,
    ):
        super().__init__(model=None, **kwargs)
        self.save_hyperparameters(ignore=["model", "features", "flow"])

        self.flow = flow
        self.features = features if features is not None else nn.Identity()
        self.features.eval()
        self.ckpt_path = ckpt
        self.mask_aware = mask_aware

    def on_fit_start(self):
        """Move features to device and optionally load checkpoint."""
        self.features.to(self.device)

        if self.ckpt_path:
            self._load_checkpoint()

    def forward(
        self, x: torch.Tensor, present: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Run the flow and return the log-probability of the input."""
        x = self.features(x)
        return self.flow.log_prob(x, present)

    def model_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        """Shared train/val/test step."""
        b = unpack_batch(batch)
        x = torch.flatten(b.x, start_dim=1)
        present = _present(b.mask, x) if self.mask_aware else None

        log_prob = self.forward(x, present)
        ascore = -log_prob

        if ascore.ndim != 1:
            raise ValueError(f"Expected per-event ascores, got {tuple(ascore.shape)}.")

        with torch.no_grad():
            operational_ascore = self.compute_operational_ascore(ascore)

            q50, q99 = self.compute_score_quantiles(ascore)

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
