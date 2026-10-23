from pathlib import Path

import torch
from pytorch_lightning.callbacks import Callback

from src.data.utils import unpack_batch
from src.evaluation.callbacks import utils
from src.evaluation.callbacks.components.cap.bank import build_couplings
from src.evaluation.callbacks.components.cap.metric import CAPBank


def _flat_inputs(batch) -> torch.Tensor:
    x = unpack_batch(batch).x.detach()
    return x.reshape(len(x), -1).cpu()


class CAPBankCallback(Callback):
    """Evaluate every statistic of the CAP bank between two test sets.

    The scores of dataset_1 and dataset_2 are collected batch by batch at every test
    epoch, and every key of ``CAPBank.compute()`` is recorded under '<name>/<key>'
    through ``log_metrics_to_mlflow``. The couplings of ``bank_config["pairings"]`` are
    built once per instance, at the end of its first test epoch, from the inputs of the
    two sets and of the nn caliper's reference set, flattened to [n, d] and collected
    in the same batch order as the scores. When the trainer carries ``out_dir``, as the
    dump stage's trainers do, they are cached in ``<out_dir>/cap_bank`` (see
    ``bank.build_couplings``), so that the runs of a lattice served the same inputs
    share them; otherwise they are kept in memory only. Nothing is plotted.

    :param output_name: Name of the model output used as anomaly score.
    :param dataset_1: Name of the test dataloader of X1.
    :param dataset_2: Name of the test dataloader of X2.
    :param reference: Name of the test dataloader that sets the nn caliper.
    :param bank_config: The bank's configuration, as in configs/cap/bank.yaml.
    :param log_raw_mlflow: Unused, accepted for config compatibility.
    :param name: Callback name, the prefix of the recorded scalars.
    """

    def __init__(
        self,
        output_name: str,
        dataset_1: str,
        dataset_2: str,
        reference: str,
        bank_config: dict,
        log_raw_mlflow: bool = True,
        name: str = "cap_bank",
    ):
        super().__init__()
        self.output_name = output_name
        self.datasets, self.reference = (dataset_1, dataset_2), reference
        self.bank_config = bank_config
        self.name = name
        self.couplings = None

    def on_test_epoch_start(self, trainer, pl_module):
        """Reset the scores and, until the couplings exist, the inputs."""
        self.scores = {name: [] for name in self.datasets}
        sets = (*self.datasets, self.reference) if self.couplings is None else ()
        self.inputs = {name: [] for name in sets}

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Cache the scores of the two sets and, until the couplings exist, the
        inputs."""
        dataset_name = list(trainer.test_dataloaders.keys())[dataloader_idx]
        if dataset_name in self.scores:
            score = outputs[self.output_name].detach().reshape(-1)
            self.scores[dataset_name].append(score)
        if dataset_name in self.inputs:
            self.inputs[dataset_name].append(_flat_inputs(batch))

    def on_test_epoch_end(self, trainer, pl_module):
        """Compute the bank for the current checkpoint and record every value."""
        ckpt_name = Path(pl_module._ckpt_path).stem
        if self.couplings is None:
            self.couplings = self._build_couplings(trainer)
        bank = CAPBank(**self.bank_config)
        bank.update(*(torch.cat(self.scores[n]) for n in self.datasets), self.couplings)
        utils.mlflow.log_metrics_to_mlflow(
            trainer, bank.compute(), ckpt_name=ckpt_name, cb_name=self.name
        )

    def _build_couplings(self, trainer) -> dict:
        """Build or read the couplings of the collected inputs, which are then
        dropped."""
        out_dir = getattr(trainer, "out_dir", None)
        cache_dir = None if out_dir is None else Path(out_dir) / "cap_bank"
        names = (*self.datasets, self.reference)
        x_1, x_2, x_ref = (torch.cat(self.inputs[name]) for name in names)
        self.inputs = {}
        label = dict(zip(("dataset_1", "dataset_2", "reference"), names))
        pairings = self.bank_config["pairings"]
        return build_couplings(x_1, x_2, x_ref, pairings, cache_dir, label)
