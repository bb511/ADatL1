import math

import torch
from pytorch_lightning import Callback

from src.callbacks.components.cap.bank import build_couplings
from src.callbacks.components.cap.metric import CAPBank
from src.callbacks.components.selection import EMA_WEIGHT, ema_update
from src.data.utils import unpack_batch


def _flat_inputs(batch) -> torch.Tensor:
    x = unpack_batch(batch).x.detach()
    return x.reshape(len(x), -1).cpu()


def _logged_keys(bank_config, dataset_1: str, dataset_2: str) -> dict[str, str]:
    """Return the logged key of every registered statistic, by its key in the bank."""
    pair_name = "_vs_".join(name.replace("/", "_") for name in (dataset_1, dataset_2))
    return {
        f"{p}_{m}_{s}": f"val/summary/cap_{p}_{m}_{s}_ema_{pair_name}"
        for p in bank_config["pairings"]
        for m in bank_config["maps"]
        for s in bank_config["statistics"]
    }


def _bank_values(scores_1, scores_2, couplings: dict, bank_config) -> dict[str, float]:
    bank = CAPBank(**bank_config)
    bank.update(scores_1, scores_2, couplings)
    return bank.compute()


class CAPBankCallback(Callback):
    """Log the exponential moving average of every statistic of the CAP bank.

    The scores of dataset_1 and dataset_2 are collected batch by batch at every
    validation epoch after the sanity pass. At the first such epoch the inputs of the
    two sets and of the nn caliper's reference set are collected too, flattened to
    [n, d], in the same batch order as the scores; that epoch's end builds the couplings
    of ``bank_config["pairings"]`` once, in memory, and drops the inputs. Every epoch
    then logs, for each registered pairing, evidence map and statistic,
    ``val/summary/cap_<pairing>_<map>_<statistic>_ema_<dataset_1>_vs_<dataset_2>``,
    averaged as CAPCallback averages: a non-finite value leaves the average unchanged.

    :param output_name: Key in outputs dict containing per-event anomaly scores.
    :param dataset_1: Name of the validation dataloader of X1.
    :param dataset_2: Name of the validation dataloader of X2.
    :param reference: Name of the validation dataloader that sets the nn caliper.
    :param bank_config: The bank's configuration, as in configs/cap/bank.yaml.
    :param ema_weight: Weight of the previous value in the exponential moving average.
    """

    def __init__(
        self,
        output_name: str,
        dataset_1: str,
        dataset_2: str,
        reference: str,
        bank_config: dict,
        ema_weight: float = EMA_WEIGHT,
    ):
        super().__init__()
        self.output_name, self.ema_weight = output_name, ema_weight
        self.datasets, self.reference = (dataset_1, dataset_2), reference
        self.bank_config = bank_config
        self.keys = _logged_keys(bank_config, dataset_1, dataset_2)
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
        self.emas = {name: math.nan for name in self.keys}
        self.couplings = None

    def on_validation_epoch_start(self, trainer, pl_module):
        """Check the dataloaders and reset the scores and, until the couplings exist,
        the inputs."""
        dset_names = list(trainer.val_dataloaders.keys())
        missing = [n for n in (*self.datasets, self.reference) if n not in dset_names]
        if missing:
            raise ValueError(
                f"{self.__class__.__name__} requires validation dataloaders {missing}. "
                f"Available validation dataloaders: {dset_names}"
            )
        self.scores = {name: [] for name in self.datasets}
        sets = (*self.datasets, self.reference) if self.couplings is None else ()
        self.inputs = {name: [] for name in sets}

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Collect the scores of the two sets and, until the couplings exist, the
        inputs."""
        if trainer.sanity_checking:
            return
        dset_name = list(trainer.val_dataloaders.keys())[dataloader_idx]
        if dset_name in self.scores:
            score = outputs[self.output_name].detach().reshape(-1)
            self.scores[dset_name].append(score)
        if dset_name in self.inputs:
            self.inputs[dset_name].append(_flat_inputs(batch))

    def on_validation_epoch_end(self, trainer, pl_module):
        """Compute the bank and log the moving average of every statistic."""
        if trainer.sanity_checking:
            return
        if self.couplings is None:
            self.couplings = self._build_couplings()
        scores_1, scores_2 = (torch.cat(self.scores[n]) for n in self.datasets)
        values = _bank_values(scores_1, scores_2, self.couplings, self.bank_config)
        pl_module.log_dict(self._averaged(values), **self.log_kwargs)

    def _averaged(self, values: dict[str, float]) -> dict[str, float]:
        """Update the moving averages with the epoch's values; return those to log."""
        for name in self.keys:
            self.emas[name] = ema_update(
                self.emas[name], values[name], self.ema_weight, skip_nonfinite=True
            )
        return {key: self.emas[name] for name, key in self.keys.items()}

    def _build_couplings(self) -> dict:
        """Build the couplings from the collected inputs, which are then dropped."""
        x_1, x_2, x_ref = (
            torch.cat(self.inputs[name]) for name in (*self.datasets, self.reference)
        )
        self.inputs = {}
        return build_couplings(x_1, x_2, x_ref, self.bank_config["pairings"])
