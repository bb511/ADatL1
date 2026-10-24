import math
from pathlib import Path

import torch
from pytorch_lightning.callbacks import Callback

from src.data.utils import unpack_batch
from src.evaluation.callbacks import utils
from src.evaluation.callbacks.components.cap.bank import build_couplings
from src.evaluation.callbacks.components.cap.metric import CAPBank
from src.evaluation.callbacks.components.cap.representation import (
    REPRESENTATIONS,
    represent,
)

# The loader whose inputs a representation is standardised on, when one is served:
# the training set the dump stage appends under extra_sets.train.
FIT_SET = "train"


def _inputs(batch) -> torch.Tensor:
    return unpack_batch(batch).x.detach().cpu()


class CAPBankCallback(Callback):
    """Evaluate every statistic of the CAP bank between two test sets.

    The scores of dataset_1 and dataset_2 are collected batch by batch at every test
    epoch, and every key of ``CAPBank.compute()`` is recorded under '<name>/<key>'
    through ``log_metrics_to_mlflow``. The couplings of ``bank_config["pairings"]`` are
    built once per instance, at the end of its first test epoch, from the inputs of the
    two sets and of the nn caliper's reference set, collected in the same batch order
    as the scores and mapped by the representation to [n, d]. When the trainer
    carries ``out_dir``, as the dump stage's trainers do, they are cached in
    ``<out_dir>/cap_bank`` (see ``bank.build_couplings``), so that the runs of a
    lattice served the same inputs share them; otherwise they are kept in memory only.
    The scores of the two sets at the checkpoint named ``initial``, which the dump
    stage replays first, are kept: ``disp`` reads them as the initial state's at that
    checkpoint and every later one. Before such a checkpoint they are NaN, and so are
    disp's values. Nothing is plotted.

    :param output_name: Name of the model output used as anomaly score.
    :param dataset_1: Name of the test dataloader of X1.
    :param dataset_2: Name of the test dataloader of X2.
    :param reference: Name of the test dataloader that sets the nn caliper.
    :param bank_config: The bank's configuration, as in configs/cap/bank.yaml.
    :param log_raw_mlflow: Unused, accepted for config compatibility.
    :param name: Callback name, the prefix of the recorded scalars.
    :param representation: Name in ``representation.REPRESENTATIONS`` of the fixed map
        from the inputs to the coordinates the couplings read; the identity flattens
        them. Every other representation is standardised on the inputs of the
        dataloader named 'train', which the dump stage appends when
        ``extra_sets.train`` is set, otherwise on X1's; its name, parameters and fit
        set enter the coupling cache's hash and label.
    :param representation_params: Keyword arguments of the representation.
    :param x2_blocks: With k, dataset_2's rows are read as k equal consecutive blocks,
        each one ot subsample (the members of a pool, each a changed copy of X1), as
        ``bank.build_couplings`` takes its ``blocks``.
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
        representation: str = "identity",
        representation_params: dict | None = None,
        x2_blocks: int | None = None,
    ):
        super().__init__()
        if representation not in REPRESENTATIONS:
            raise ValueError(f"Unknown representation '{representation}'.")
        self.representation = representation
        self.representation_params = dict(representation_params or {})
        self.x2_blocks = x2_blocks
        self.output_name = output_name
        self.datasets, self.reference = (dataset_1, dataset_2), reference
        self.bank_config = bank_config
        self.name = name
        self.couplings = None
        self.initial = None

    def on_test_epoch_start(self, trainer, pl_module):
        """Reset the scores and, until the couplings exist, the inputs."""
        self.scores = {name: [] for name in self.datasets}
        loaders = list(trainer.test_dataloaders.keys())
        sets = self._coupling_sets(loaders) if self.couplings is None else ()
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
            self.inputs[dataset_name].append(_inputs(batch))

    def on_test_epoch_end(self, trainer, pl_module):
        """Compute the bank for the current checkpoint and record every value."""
        ckpt_name = Path(pl_module._ckpt_path).stem
        if self.couplings is None:
            self.couplings = self._build_couplings(trainer)
        scores = [torch.cat(self.scores[name]) for name in self.datasets]
        if ckpt_name == "initial":
            self.initial = scores
        bank = CAPBank(**self.bank_config)
        bank.update(*scores, self.couplings, *self._initial(scores))
        utils.mlflow.log_metrics_to_mlflow(
            trainer, bank.compute(), ckpt_name=ckpt_name, cb_name=self.name
        )

    def _initial(self, scores: list[torch.Tensor]) -> list[torch.Tensor]:
        # NaN initial scores give disp NaN values under its usual keys.
        if self.initial is None:
            return [torch.full_like(s, math.nan) for s in scores]
        return self.initial

    def _coupling_sets(self, loaders: list[str]) -> tuple[str, ...]:
        """Return the sets whose inputs the couplings read, the fit set included."""
        sets = (*self.datasets, self.reference)
        if self.representation != "identity" and FIT_SET in loaders:
            return (*sets, FIT_SET)
        return sets

    def _represented(self, names: tuple[str, ...]) -> tuple[list, dict | None]:
        """Return the inputs of names in the representation, and its record for the
        cache, None for the identity; the collected inputs are dropped."""
        inputs = {name: torch.cat(batches) for name, batches in self.inputs.items()}
        self.inputs = {}
        fit = FIT_SET if FIT_SET in inputs else names[0]
        sets = [inputs[name] for name in names]
        params = self.representation_params
        represented = represent(self.representation, params, sets, inputs[fit])
        if self.representation == "identity":
            return represented, None
        return represented, {"name": self.representation, "params": params, "fit": fit}

    def _build_couplings(self, trainer) -> dict:
        """Build or read the couplings of the collected inputs, which are then
        dropped."""
        out_dir = getattr(trainer, "out_dir", None)
        cache_dir = None if out_dir is None else Path(out_dir) / "cap_bank"
        names = (*self.datasets, self.reference)
        (x_1, x_2, x_ref), record = self._represented(names)
        label = dict(zip(("dataset_1", "dataset_2", "reference"), names))
        pairings, blocks = self.bank_config["pairings"], self.x2_blocks
        return build_couplings(
            x_1, x_2, x_ref, pairings, cache_dir, label, blocks, record
        )
