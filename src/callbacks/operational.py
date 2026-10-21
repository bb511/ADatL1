"""The operational score Q'' of validation normal as a strategy callback."""

import torch
from pytorch_lightning import Callback

from src.callbacks.components.selection import EMA_WEIGHT, ema_update


class OperationalScoreCallback(Callback):
    """Compute the EMA of the operational score Q'' of the validation normal data set.

    The value is the mean over the batches of ``ascore/operational``, which the module
    logs itself as ``val/normal/ascore_operational``; only the EMA is logged here, under
    ``val/summary/<name>_ema``, from the first epoch after the sanity pass.

    :param name: Name in the logged key; None takes "ascore_operational".
    :param output_name: Key in outputs dict containing the per-batch operational score.
    :param ema_weight: Weight of the previous value in the exponential moving average.
    """

    def __init__(
        self,
        name: str | None = None,
        output_name: str = "ascore/operational",
        ema_weight: float = EMA_WEIGHT,
    ):
        super().__init__()
        self.name = "ascore_operational" if name is None else name
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
        self.operational_ema = None

    def on_validation_epoch_start(self, trainer, pl_module):
        """Make sure the normal data is in the used data sets."""
        dset_names = list(trainer.val_dataloaders.keys())
        if "normal" not in dset_names:
            raise ValueError(
                f"{self.__class__.__name__} requires a dataloader named 'normal'. "
                f"Available validation dataloaders: {dset_names}"
            )

        self.normal_scores = []

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Aggregate the operational scores of the normal validation data set."""
        dset_name = list(trainer.val_dataloaders.keys())[dataloader_idx]
        if dset_name != "normal":
            return

        # The module returns a Python float per batch, held here as a float32 tensor.
        score = torch.as_tensor(outputs[self.output_name]).detach()
        self.normal_scores.append(score.reshape(-1))

    def on_validation_epoch_end(self, trainer, pl_module):
        """Compute and log the EMA of the operational score."""
        if trainer.sanity_checking:
            return

        operational = torch.cat(self.normal_scores).cpu().double().mean().item()
        self.operational_ema = ema_update(
            self.operational_ema, operational, self.ema_weight
        )

        metrics = {f"val/summary/{self.name}_ema": self.operational_ema}
        pl_module.log_dict(metrics, **self.log_kwargs)
