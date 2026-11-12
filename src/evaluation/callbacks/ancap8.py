"""ANCAP v8 as an evaluation callback, the mirror of src/callbacks/ancap8.py over the
named loaders of a test loop."""

from pathlib import Path

import torch
from pytorch_lightning.callbacks import Callback

from src.callbacks.ancap8 import ancap8
from src.evaluation.callbacks import utils


class ANCAP8(Callback):
    """Evaluate ANCAP v8's values of every cell at a checkpoint.

    The scores of dataset_1, of each cell's view and of the points are collected batch
    by batch at every test epoch; the points' loaders are joined in list order into P.
    Every key of ``src.callbacks.ancap8.ancap8``, ``<key>_<cell>``, is recorded under
    '<name>/<key>_<cell>' through ``log_metrics_to_mlflow``. The selection reads them at
    the lattice step (``src/evaluation/components/ancap8.py``).

    Args:
        output_name: Name of the model output used as anomaly score.
        dataset_1: Name of the test dataloader of S_1.
        views: Name of the test dataloader of S_2, by cell.
        points: Names of the test dataloaders of the points, joined in list order.
        kinds: Whether the k-th loader of points holds the points of kind k, counted
            from 1.
        sizes: Declared sizes, any of ``n1``, ``n2`` and ``n_p``, ``n2`` one size or
            one per cell; an array of another length makes its candidate unusable.
        name: Callback name, the prefix of the recorded scalars.
    """

    def __init__(
        self,
        output_name: str,
        dataset_1: str,
        views: dict[str, str],
        points: list[str],
        kinds: bool = False,
        sizes: dict[str, int | dict[str, int]] | None = None,
        name: str = "ancap8",
    ):
        super().__init__()
        self.output_name, self.name = output_name, name
        self.dataset_1, self.views = dataset_1, dict(views)
        self.points, self.kinds = list(points), kinds
        self.sizes = None if sizes is None else dict(sizes)
        self.sets = [dataset_1, *self.views.values(), *self.points]

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
        """Cache the scores of S_1, of the views and of the points."""
        dset_name = list(trainer.test_dataloaders.keys())[dataloader_idx]
        if dset_name in self.scores:
            score = torch.as_tensor(outputs[self.output_name]).detach()
            self.scores[dset_name].append(score.reshape(-1).cpu())

    def on_test_epoch_end(self, trainer, pl_module):
        """Compute every cell's values at the current checkpoint and record them."""
        ckpt_name = Path(pl_module._ckpt_path).stem
        utils.mlflow.log_metrics_to_mlflow(
            trainer, self._values(), ckpt_name=ckpt_name, cb_name=self.name
        )

    def _values(self) -> dict[str, float]:
        scores = {name: torch.cat(batches) for name, batches in self.scores.items()}
        views = {cell: scores[name] for cell, name in self.views.items()}
        points = [scores[name] for name in self.points]
        return ancap8(scores[self.dataset_1], views, points, self.kinds, self.sizes)
