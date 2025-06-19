# Callback that calculates the rate of anomalies.

import torch
from pytorch_lightning.callbacks import Callback

from src.models.metrics.rate import AnomalyRate


class AnomalyRateCallback(Callback):
    """Calculates the how many anomalies are detected given a certain trigger rate.

    :target_rate: Float specifying the target rate of anomalies.
    :bc_rate: Float containing the bunch crossing rate in kHz. This is the
        revolution frequency of the LHC times the number of proton bunches in the
        LHC tunnel, which gives the total rate of events that are processed by
        the L1 trigger.
    """
    def __init__(self, target_rates: list[int], bc_rate: int):
        super().__init__()
        self.target_rates = target_rates
        self.bc_rate = bc_rate

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        main_val = trainer.val_dataloaders["main_val"]
        print(main_val.size())
        exit(1)


        # self.rate_metrics = {}
        # for target_rate in self.target_rates:
        #     metric = AnomalyRate(target_rate, self.bc_rate)
        #     metric.set_threshold(z_mean)
        #     self.rate_metrics.update({target_rate: metric})

        # for data_name, data in trainer.val_dataloaders.items():
        #     for rate in self.target_rates:

        #         for batch in data:
        #             batch = batch.flatten(start_dim=1).to(dtype=torch.float32)

        #             z_mean,_,_,_ = pl_module.forward(batch)
        #             self.rate_metrics[rate].update(z_mean)

        #         pl_module.log_dict(
        #             {f"val/{data_name}/{rate}": self.rate_metrics[rate].compute()},
        #             prog_bar=False,
        #             on_step=False,
        #             on_epoch=True,
        #             logger=True,
        #             sync_dist=True,
        #             add_dataloader_idx=False,
        #         )

    # def _set_thresholds(self):
