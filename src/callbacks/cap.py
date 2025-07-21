from typing import Optional
from collections import defaultdict
import torch
from pytorch_lightning.utilities.memory import garbage_collection_cuda

from capmetric.callback import ApproximationCapacityCallback
import wandb


class ApproximationCapacityCallbackDebug(ApproximationCapacityCallback):
    """
    Callback to compute the CAP (Conditional Average Precision) metric during validation.

    Args:
        beta0 (float): Initial value for the beta parameter.
    """

    def __init__(
        self,
        log_optimization: Optional[bool] = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.log_optimization = log_optimization

    @staticmethod
    def symlog(x, linthresh=1.0):
        return torch.sign(x) * torch.log1p(torch.abs(x / linthresh))

    def on_validation_epoch_end(self, trainer, pl_module):
        """Compute CAP for the two first datasets"""

        if (pl_module.current_epoch + 1) % self.log_every_n_epochs != 0:
            return

        # Obtain the loss from the cache:
        list_dset_key = list(self.cache.keys())
        for pair_name, (ds_name1, ds_name2) in self.data_pairs.items():
            self.capmetric.reset()
            if ds_name1 not in list_dset_key or ds_name2 not in list_dset_key:
                print(f"CAP: {pair_name} datasets not found in cache")
                continue

            # Concatenate losses and pair:
            loss1 = torch.cat(self.cache[ds_name1])
            loss2 = torch.cat(self.cache[ds_name2])
            indices1, indices2 = self.pairing_fn(loss1, loss2)
            # if self.log_optimization:
            #     try:
            #         pl_module.logger.experiment.log({
            #                 f"cap-debug/{pair_name}/loss1": wandb.Histogram(loss1.cpu().numpy().flatten()),
            #                 f"cap-debug/{pair_name}/loss2": wandb.Histogram(loss2.cpu().numpy().flatten()),
            #                 f"cap-debug/{pair_name}/loss1_filtered": wandb.Histogram(loss1[indices1].cpu().numpy().flatten()),
            #                 f"cap-debug/{pair_name}/loss2_filtered": wandb.Histogram(loss2[indices2].cpu().numpy().flatten()),
            #             },
            #             step=pl_module.global_step
            #         )
            #     except Exception as e:
            #         print(f"It was not possible to log {e}.")
            #         pass

            loss1, loss2 = loss1[indices1], loss2[indices2]

            # Compute the metric:
            self.capmetric.update(
                loss1[: min(len(loss1), len(loss2))],
                loss2[: min(len(loss1), len(loss2))]
            )
            pl_module.log_dict(
                {
                    f"cap/{pair_name}": self.capmetric.compute(),
                },
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )
            
            # Log arrays with the debugging plots:
            if self.log_optimization:
                try:
                    cap_values = [log['cap'] for log in self.capmetric.epoch_logs]
                    beta_values = [log['beta'] for log in self.capmetric.epoch_logs]
                   
                    pl_module.logger.experiment.log({
                        f"cap-{pair_name}/cap@e={pl_module.current_epoch}": wandb.plot.line_series(
                            xs=list(range(len(cap_values))), 
                            ys=[self.symlog(torch.tensor(cap_values)).tolist()], 
                            keys=["CAP"], 
                            title=f"CAP@{pl_module.current_epoch}", 
                            xname="epoch"
                        ),
                        f"cap-{pair_name}/beta@e={pl_module.current_epoch}": wandb.plot.line_series(
                            xs=list(range(len(beta_values))), 
                            ys=[beta_values], 
                            keys=["Beta"], 
                            title=f"Beta@{pl_module.current_epoch}", 
                            xname="epoch"
                        ),
                    },
                    step=pl_module.global_step
                )
                except Exception as e:
                    print(f"It was not possible to log {e}.")
                    pass
            
        self.cache = defaultdict(list)
        del loss1, loss2, indices1, indices2; garbage_collection_cuda()
