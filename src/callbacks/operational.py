"""The fit Q'' of validation normal as a strategy callback."""

import torch
from pytorch_lightning import Callback

from src.callbacks.components.selection import EMA_WEIGHT, ema_update


class FitCallback(Callback):
    """Compute the fit Q'' of the validation normal data set and its EMA.

    The value is the mean of the per-event ``fit/full`` over every row of validation
    normal, so it depends on no batch size. It is logged under ``val/summary/<name>``
    and its EMA under ``val/summary/<name>_ema``, from the first epoch after the sanity
    pass.

    :param name: Name in the logged keys; None takes "fit".
    :param output_name: Key in outputs dict containing the per-event fit.
    :param ema_weight: Weight of the previous value in the exponential moving average.
    """

    def __init__(
        self,
        name: str | None = None,
        output_name: str = "fit/full",
        ema_weight: float = EMA_WEIGHT,
    ):
        super().__init__()
        self.name = "fit" if name is None else name
        self.output_name = output_name
        self.ema_weight = ema_weight

        self.log_kwargs = dict(
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            add_dataloader_idx=False,
            batch_size=1,
        )

    def on_fit_start(self, trainer, pl_module):
        """Instantiate useful quantities."""
        self.fit_ema = None

    def on_validation_epoch_start(self, trainer, pl_module):
        """Make sure the normal data is in the used data sets."""
        dset_names = list(trainer.val_dataloaders.keys())
        if "normal" not in dset_names:
            raise ValueError(
                f"{self.__class__.__name__} requires a dataloader named 'normal'. "
                f"Available validation dataloaders: {dset_names}"
            )

        self.normal_fits = []

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Aggregate the per-event fit of the normal validation data set."""
        dset_name = list(trainer.val_dataloaders.keys())[dataloader_idx]
        if dset_name != "normal":
            return

        fit = torch.as_tensor(outputs[self.output_name]).detach()
        self.normal_fits.append(fit.reshape(-1).cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        """Compute and log the fit and its EMA."""
        if trainer.sanity_checking:
            return

        fit = torch.cat(self.normal_fits).double().mean().item()
        self.fit_ema = ema_update(self.fit_ema, fit, self.ema_weight)

        metrics = {
            f"val/summary/{self.name}": fit,
            f"val/summary/{self.name}_ema": self.fit_ema,
        }
        pl_module.log_dict(metrics, **self.log_kwargs)
