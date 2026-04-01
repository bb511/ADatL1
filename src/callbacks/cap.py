import torch
from pytorch_lightning import Callback

from src.callbacks.metrics.cap.metric import ApproximationCapacity
from src.callbacks.metrics.cap.binary import get_pairing_fn


class CAPCallback(Callback):
    """Compute a validation approximation capacity metric between two datasets.

    This callback collects all scores from two validation dataloaders, pairs the
    samples according to the selected pairing function, and computes:
        - CAP metric
        - Spearman rank correlation

    The metric is computed on the paired score samples from:
        - dataset_1
        - dataset_2

    This is appropriate as a validation-side proxy objective for HPO.

    :param loss_name: Key in outputs dict containing per-event anomaly scores / losses.
    :param dataset_1: Name of the first validation dataloader.
    :param dataset_2: Name of the second validation dataloader.
    :param pairing_type: Pairing scheme name passed to get_pairing_fn.
    :param cap_metric_config: Keyword arguments used to instantiate
        ApproximationCapacity.
    :beta: Float that sets parameter of EMA metrics compute here.
    """

    def __init__(
        self,
        metric_name: str,
        dataset_1: str,
        dataset_2: str,
        pairing_type: str,
        cap_metric_config: dict,
        beta: float = 0.9
    ):
        super().__init__()
        self.device = None
        self.metric_name = metric_name
        self.dataset_1_name = dataset_1
        self.dataset_2_name = dataset_2
        self.cap_metric_config = cap_metric_config
        self.pairing_fn = get_pairing_fn(pairing_type)
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
        self.cap_ema = None

    def on_validation_epoch_start(self, trainer, pl_module):
        """Set device, validate dataloaders, and initialize metric state."""
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

        self.capmetric = ApproximationCapacity(**self.cap_metric_config, device=self.device)
        self.capmetric.to(self.device)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Aggregate anomaly scores from the two designated validation data sets."""
        dset_name = list(trainer.val_dataloaders.keys())[dataloader_idx]
        if dset_name not in (self.dataset_1_name, self.dataset_2_name):
            return

        loss = outputs[self.metric_name]
        if loss.ndim == 0:
            raise ValueError(f"outputs['{self.metric_name}'] is scalar. Need a tensor.")

        loss = loss.detach().view(-1)

        if dset_name == self.dataset_1_name:
            self.dataset_1_scores.append(loss)

        if dset_name == self.dataset_2_name:
            self.dataset_2_scores.append(loss)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Compute and log CAP and rank-correlation across the two data sets."""
        if not self.dataset_1_scores:
            raise RuntimeError(
                f"No validation scores were collected for '{self.dataset_1_name}'."
            )
        if not self.dataset_2_scores:
            raise RuntimeError(
                f"No validation scores were collected for '{self.dataset_2_name}'."
            )

        cap_value, rankcorr_value = self._compute_cap()

        ds1 = self.dataset_1_name.replace("/", "_")
        ds2 = self.dataset_2_name.replace("/", "_")
        self._compute_cap_ema(cap_value)

        pl_module.log_dict(
            {
                f"val/summary/cap_ema_{ds1}_vs_{ds2}": self.cap_ema,
                f"val/summary/rankcorr_{ds1}_vs_{ds2}": float(rankcorr_value),
            },
            **self.log_kwargs,
        )

    def _compute_cap(self):
        """Compute CAP and Spearman rank correlation between the two data sets."""
        dataset_1_scores = torch.cat(self.dataset_1_scores, dim=0).view(-1)
        dataset_2_scores = torch.cat(self.dataset_2_scores, dim=0).view(-1)

        idxs1, idxs2 = self.pairing_fn(dataset_1_scores, dataset_2_scores)
        ds1_scores = dataset_1_scores[idxs1]
        ds2_scores = dataset_2_scores[idxs2]

        n = min(len(ds1_scores), len(ds2_scores))
        if n <= 0:
            raise RuntimeError("No paired validation samples available for CAP.")

        ds1_scores = ds1_scores[:n]
        ds2_scores = ds2_scores[:n]

        rankcorr_value = self._spearman_corr(ds1_scores, ds2_scores)

        with torch.inference_mode(False):
            with torch.enable_grad():
                ds1 = ds1_scores.clone().requires_grad_(True)
                ds2 = ds2_scores.clone().requires_grad_(True)
                self.capmetric.update(ds1, ds2)

        cap_value = self.capmetric.compute()
        if isinstance(cap_value, torch.Tensor):
            cap_value = cap_value.detach().item()

        return float(cap_value), float(rankcorr_value)

    def _spearman_corr(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute the Spearman correlation between paired anomaly scores."""
        x = x.detach().view(-1).cpu()
        y = y.detach().view(-1).cpu()

        if x.numel() == 0 or y.numel() == 0:
            raise RuntimeError("Cannot compute Spearman correlation on empty tensors.")

        rx = torch.argsort(torch.argsort(x)).float()
        ry = torch.argsort(torch.argsort(y)).float()

        rx = rx - rx.mean()
        ry = ry - ry.mean()

        denom = torch.sqrt((rx ** 2).sum() * (ry ** 2).sum())
        if denom < 1e-12:
            return float("nan")

        return ((rx * ry).sum() / denom).item()

    def _compute_cap_ema(self, cap: float):
        """Compute the cvar estimated moving average."""
        if self.cap_ema is None:
            self.cap_ema = float(cap)
        else:
            self.cap_ema = \
                self.beta * self.cap_ema + (1 - self.beta) * float(cap)
