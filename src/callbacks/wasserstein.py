# Wasserstein callback.
import math

import torch
from pytorch_lightning import Callback


class WassersteinCallback(Callback):
    """Compute a validation Wasserstein distance between two dataloaders.

    This callback collects all scores from two validation dataloaders and computes
    the empirical 1D Wasserstein-1 distance between their score distributions.

    For the two empirical score samples:
        - x from dataset_1
        - y from dataset_2

    the Wasserstein distance is computed by:
        1) sorting both samples
        2) averaging absolute quantile differences
        3) interpolating onto a common quantile grid if sample sizes differ

    Optionally, a log1p transform is applied before the distance computation.

    :param output_name: Key in outputs dict containing per-event anomaly scores / losses.
    :param dataset_1: Name of the first validation dataloader.
    :param dataset_2: Name of the second validation dataloader.
    :param apply_log1p: Whether to apply log1p transform before distance computation.
    :beta: Float that sets parameter of EMA metrics compute here.
    """

    def __init__(
        self,
        output_name: str,
        dataset_1: str,
        dataset_2: str,
        apply_log1p: bool = True,
        beta: float = 0.9,
    ):
        super().__init__()
        self.output_name = output_name
        self.dataset_1_name = dataset_1
        self.dataset_2_name = dataset_2
        self.apply_log1p = bool(apply_log1p)
        self.beta = beta

        self.log_kwargs = dict(
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )

    def on_fit_start(self, trainer, pl_module):
        """Instantiate useful quantities."""
        self.w1dist_ema = None

    def on_validation_epoch_start(self, trainer, pl_module):
        """Set the device and check that both required dataloaders are present."""
        self.device = pl_module.device

        dset_names = list(trainer.val_dataloaders.keys())
        missing = [
            dset_name
            for dset_name in (self.dataset_1_name, self.dataset_2_name)
            if dset_name not in dset_names
        ]
        if missing:
            raise ValueError(
                f"{self.__class__.__name__} requires validation dataloaders {missing}. "
                f"Available validation dataloaders: {dset_names}"
            )

        self.dataset_1_scores = []
        self.dataset_2_scores = []

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Aggregate anomaly scores from the two designated validation data sets."""
        dset_name = list(trainer.val_dataloaders.keys())[dataloader_idx]
        if dset_name not in (self.dataset_1_name, self.dataset_2_name):
            return

        loss = outputs[self.output_name]
        if loss.ndim == 0:
            raise ValueError(f"outputs['{self.output_name}'] is scalar. Need a tensor.")

        loss = loss.detach().view(-1)

        if dset_name == self.dataset_1_name:
            self.dataset_1_scores.append(loss)

        if dset_name == self.dataset_2_name:
            self.dataset_2_scores.append(loss)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Compute and log the Wasserstein distance across the two data sets."""
        if not self.dataset_1_scores:
            raise RuntimeError(
                f"No validation scores were collected for '{self.dataset_1_name}'."
            )
        if not self.dataset_2_scores:
            raise RuntimeError(
                f"No validation scores were collected for '{self.dataset_2_name}'."
            )

        wasserstein = self._compute_wasserstein()
        self._compute_w1dist_ema(wasserstein)

        ds1 = self.dataset_1_name.replace("/", "_")
        ds2 = self.dataset_2_name.replace("/", "_")

        pl_module.log_dict(
            {f"val/summary/w1dist_ema_{ds1}_vs_{ds2}": float(self.w1dist_ema)},
            **self.log_kwargs,
        )

    def _compute_wasserstein(self) -> float:
        """Compute the 1D Wasserstein distance between the two score sets."""
        x = torch.cat(self.dataset_1_scores, dim=0).view(-1).cpu().float()
        y = torch.cat(self.dataset_2_scores, dim=0).view(-1).cpu().float()

        if self.apply_log1p:
            x = torch.log1p(torch.clamp(x, min=0.0))
            y = torch.log1p(torch.clamp(y, min=0.0))

        return self._wasserstein_1d_sorted(x, y)

    def _wasserstein_1d_sorted(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute empirical 1D Wasserstein-1 distance."""
        x = torch.sort(x).values
        y = torch.sort(y).values

        nx = int(x.numel())
        ny = int(y.numel())

        if nx == 0 or ny == 0:
            raise RuntimeError("Cannot compute Wasserstein distance with an empty set.")

        if nx == ny:
            return torch.mean(torch.abs(x - y)).item()

        n = min(nx, ny)
        q = torch.linspace(0.0, 1.0, n, dtype=torch.float32)

        xq = self._interp_quantiles(x, q)
        yq = self._interp_quantiles(y, q)

        return torch.mean(torch.abs(xq - yq)).item()

    def _interp_quantiles(
        self, sorted_vals: torch.Tensor, q: torch.Tensor
    ) -> torch.Tensor:
        """Interpolate sorted samples at quantiles q in [0, 1]."""
        sorted_vals = sorted_vals.view(-1)
        n = int(sorted_vals.numel())

        if n == 0:
            raise RuntimeError("Cannot interpolate quantiles from an empty tensor.")

        if n == 1:
            return sorted_vals.repeat(len(q))

        pos = q * (n - 1)
        lo = torch.floor(pos).long()
        hi = torch.ceil(pos).long()
        w = pos - lo.float()

        return (1.0 - w) * sorted_vals[lo] + w * sorted_vals[hi]

    def _compute_w1dist_ema(self, w1dist: float):
        """Compute the cvar estimated moving average."""
        if self.w1dist_ema is None:
            self.w1dist_ema = float(w1dist)
        else:
            self.w1dist_ema = self.beta * self.w1dist_ema + (1 - self.beta) * float(
                w1dist
            )
