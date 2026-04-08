from typing import Optional, Tuple, Dict

import torch
from torch import nn

from src.algorithms import L1ADLightningModule
from src.utils import pylogger

log = pylogger.RankedLogger(__name__)


class RealNVP(L1ADLightningModule):
    """RealNVP on flattened input data.

    The anomaly score is the negative log-likelihood (NLL) under the flow.

    :param flow: RealNVP module operating on flattened inputs.
    :param features: Optional feature module applied before the flow.
    :param ckpt: Optional checkpoint path for restarting training.
    :param operational_rate: Float operational trigger rate.
    :param bc_rate: Float bunch crossing rate used to derive the operational quantile.
    """

    def __init__(
        self,
        flow: nn.Module,
        features: Optional[nn.Module] = None,
        ckpt: str = "",
        operational_rate: float = 0.25,
        bc_rate: float = 28608.8064,
        **kwargs,
    ):
        super().__init__(model=None, **kwargs)

        # Save non-heavy hyperparameters. Avoid serializing full modules here.
        self.save_hyperparameters(
            ignore=["model", "features", "flow", "loss"]
        )

        self.flow = flow

        self.features = features if features is not None else nn.Identity()
        self.features.eval()

        self.ckpt_path = ckpt
        self.operational_quantile = 1 - operational_rate / bc_rate

    def on_fit_start(self):
        """Move features to device and optionally load checkpoint."""
        self.features.to(self.device)

        if self.ckpt_path:
            self._load_checkpoint()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the flow and return the log-probability of the input."""
        log_prob = self.flow.log_prob(x)
        return log_prob

    def model_step(self, batch: Tuple[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Shared train/val/test step.

        The input is flattened, optionally transformed by a feature module,
        and then scored by the flow.
        """
        x, _, _, _ = batch

        x = torch.flatten(x, start_dim=1)
        x = self.features(x)

        # Flow anomaly score: negative log-likelihood.
        log_prob = self.forward(x)
        nll = -log_prob

        with torch.no_grad():
            # Number of per-sample NLL values in the batch.
            n = nll.numel()

            # Number of extreme-tail samples used for the operational proxy.
            k = max(10, int((1.0 - self.operational_quantile) * n))

            topk_vals = torch.topk(nll, k).values
            mean_top_vals = topk_vals.mean().item()

            nll_q50, nll_q95, nll_q99, nll_q999 = torch.quantile(
                nll,
                torch.tensor([0.5, 0.95, 0.99, 0.999], device=nll.device),
            ).tolist()

        return {
            # Used for backpropagation:
            "loss": nll.mean(),
            # Used for logging:
            "loss/nll/mean": nll.mean(),
            "loss/nll/median": nll_q50,
            "loss/nll/q95": nll_q95,
            "loss/nll/q99": nll_q99,
            "loss/nll/q999": nll_q999,
            "loss/nll/mean_top_vals": mean_top_vals,
            # Used for callbacks:
            "loss/nll/full": nll.detach(),
        }

    def outlog(self, outdict: dict) -> dict:
        """Values logged in the standard training loop."""
        return {
            "loss": outdict.get("loss"),
            "loss_nll": outdict.get("loss/nll/mean"),
            "loss_nll_median": outdict.get("loss/nll/median"),
            "loss_nll_q95": outdict.get("loss/nll/q95"),
            "loss_nll_q99": outdict.get("loss/nll/q99"),
            "loss_nll_q999": outdict.get("loss/nll/q999"),
            "loss_nll_mean_top_vals": outdict.get("loss/nll/mean_top_vals"),
        }

    def _load_checkpoint(self):
        """Load a previously saved checkpoint."""
        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"]
        self.load_state_dict(state_dict, strict=False)
