from typing import Optional
from collections import defaultdict

import torch
from torch.utils.data import TensorDataset
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.memory import garbage_collection_cuda

from src.models.metrics.cap import CAP


class CAP_Callback(Callback):
    """
    Callback to compute the CAP (Conditional Average Precision) metric during training.

    Args:
        beta0 (float): Initial value for the beta parameter.
    """

    def __init__(
        self,
        loss_name: str = "loss/total/full",
        beta0: Optional[float] = 1.0,
        n_epochs: Optional[int] = 100,
        lr: Optional[float] = 0.01,
        batch_size: Optional[int] = 64,
    ):
        super().__init__()
        self.loss_name = loss_name
        self.capmetric = CAP(
            beta0=beta0,
            n_epochs=n_epochs,
            lr=lr,
            batch_size=batch_size,
        )

        self.loss_cache = defaultdict(list)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self.loss_cache[dataloader_idx].append(outputs[self.loss_name].detach().cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        # TODO: Decice how the stability will be computed... In this case simply consider that there will
        # be two validation dataloaders, one for the normal and one for the anomaly data.
        loss1 = torch.cat(self.loss_cache[0])
        loss2 = torch.cat(self.loss_cache[1])
        self.loss_cache = None
        garbage_collection_cuda()

        loss_dataset = TensorDataset(
            loss1[: min(len(loss1), len(loss2))], loss2[: min(len(loss1), len(loss2))]
        )
        self.capmetric.update(loss_dataset)
        del loss_dataset
        garbage_collection_cuda()
        cap = self.capmetric.compute()

        pl_module.log_dict(
            {
                "val/cap": cap,
            },
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
