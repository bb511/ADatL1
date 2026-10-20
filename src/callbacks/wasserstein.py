# Wasserstein callback.
import torch
from pytorch_lightning import Callback


def _wasserstein_1d(x: torch.Tensor, y: torch.Tensor) -> float:
    """Return the exact W1 between the empirical distributions of x and y, the
    integral of |F_x - F_y| over the merged sorted values, in float64."""
    if x.numel() == 0 or y.numel() == 0:
        raise RuntimeError("Cannot compute Wasserstein distance with an empty set.")

    x = torch.sort(x.double()).values
    y = torch.sort(y.double()).values
    merged = torch.sort(torch.cat([x, y])).values
    cdf_x = torch.searchsorted(x, merged[:-1], right=True).double() / x.numel()
    cdf_y = torch.searchsorted(y, merged[:-1], right=True).double() / y.numel()
    return torch.sum(torch.abs(cdf_x - cdf_y) * torch.diff(merged)).item()


class WassersteinCallback(Callback):
    """Compute a validation Wasserstein distance between two dataloaders.

    This callback collects all scores from two validation dataloaders and computes the
    exact empirical 1D Wasserstein-1 distance between their score distributions, the
    integral of the absolute difference of their empirical CDFs.

    Optionally, the scores are mapped through sign(x) log1p(|x|) before the distance
    computation, which equals log1p(x) for non-negative scores.

    :param output_name: Key in outputs dict containing per-event anomaly scores /
        losses.
    :param dataset_1: Name of the first validation dataloader.
    :param dataset_2: Name of the second validation dataloader.
    :param apply_log1p: Whether to map scores through sign(x) log1p(|x|) first.
    :param ema_weight: Weight of the previous value in the exponential moving average.
    """

    def __init__(
        self,
        output_name: str,
        dataset_1: str,
        dataset_2: str,
        apply_log1p: bool = True,
        ema_weight: float = 0.9,
    ):
        super().__init__()
        self.output_name = output_name
        self.dataset_1_name = dataset_1
        self.dataset_2_name = dataset_2
        self.apply_log1p = bool(apply_log1p)
        self.ema_weight = ema_weight

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
        """Check that both required dataloaders are present."""
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
        if trainer.sanity_checking:
            return
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
            # Sign-preserving because RealNVP's negative log-likelihood is unbounded
            # below: log1p is undefined below -1, and a clamp at zero would merge
            # every negative score into 0.
            x = torch.sign(x) * torch.log1p(torch.abs(x))
            y = torch.sign(y) * torch.log1p(torch.abs(y))

        return _wasserstein_1d(x, y)

    def _compute_w1dist_ema(self, w1dist: float):
        """Update the exponential moving average of the W1 distance."""
        if self.w1dist_ema is None:
            self.w1dist_ema = float(w1dist)
        else:
            self.w1dist_ema = self.ema_weight * self.w1dist_ema + (
                1 - self.ema_weight
            ) * float(w1dist)
