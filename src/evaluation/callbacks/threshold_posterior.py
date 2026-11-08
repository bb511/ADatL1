"""ANCAP as an evaluation callback, the mirror of src/callbacks/threshold_posterior.py
over the named loaders of a test loop."""

from pathlib import Path

import torch
from pytorch_lightning.callbacks import Callback

from src.callbacks.threshold_posterior import REFERENCE, ancap, ancap_w
from src.evaluation.callbacks import utils


class ThresholdPosterior(Callback):
    """Evaluate ANCAP of a test set against the view of every grade.

    The scores of dataset_1, of the views and of the reference, a loader that
    ``extra_sets.sds`` of the dump stage appends to its validation pass, are collected
    batch by batch at every test epoch. Every key of ``ancap``, and with
    ``working_point`` every key of ``ancap_w``, is recorded under '<name>/<key>'
    through ``log_metrics_to_mlflow``. Nothing is plotted.

    Args:
        output_name: Name of the model output used as anomaly score.
        dataset_1: Name of the test dataloader of X1.
        views: Name of the test dataloader of X2, by grade.
        posterior_config: The posterior's configuration, as in
            ``configs/cap/posterior.yaml``.
        reference: Name of the test dataloader of the reference.
        name: Callback name, the prefix of the recorded scalars.
        working_point: Whether ANCAP-W's columns, which the ``ancap3`` strategy reads,
            are recorded too.
    """

    def __init__(
        self,
        output_name: str,
        dataset_1: str,
        views: dict[str, str],
        posterior_config: dict,
        reference: str = REFERENCE,
        name: str = "ancap",
        working_point: bool = False,
    ):
        super().__init__()
        self.output_name, self.name = output_name, name
        self.dataset_1, self.views = dataset_1, dict(views)
        self.reference, self.config = reference, posterior_config
        self.sets = [dataset_1, *self.views.values(), reference]
        self.working_point = working_point

    def on_test_epoch_start(self, trainer, pl_module):
        """Check that every required dataloader is present and reset the scores."""
        dset_names = list(trainer.test_dataloaders.keys())
        missing = [name for name in self.sets if name not in dset_names]
        if missing:
            raise ValueError(
                f"{self.__class__.__name__} requires test dataloaders {missing}. "
                f"Available test dataloaders: {dset_names}"
            )
        # The empty tensor stands for a set whose loader serves no batch.
        self.scores = {name: [torch.empty(0)] for name in self.sets}

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Cache the scores of the views and of the reference."""
        dset_name = list(trainer.test_dataloaders.keys())[dataloader_idx]
        if dset_name in self.scores:
            score = torch.as_tensor(outputs[self.output_name]).detach()
            self.scores[dset_name].append(score.reshape(-1).cpu())

    def on_test_epoch_end(self, trainer, pl_module):
        """Compute every column at the current checkpoint and record its values."""
        ckpt_name = Path(pl_module._ckpt_path).stem
        utils.mlflow.log_metrics_to_mlflow(
            trainer, self._values(), ckpt_name=ckpt_name, cb_name=self.name
        )

    def _values(self) -> dict[str, float]:
        scores = {name: torch.cat(batches) for name, batches in self.scores.items()}
        views = {grade: scores[name] for grade, name in self.views.items()}
        args = (scores[self.dataset_1], views, scores[self.reference], self.config)
        values = ancap(*args)
        return values | ancap_w(*args) if self.working_point else values
