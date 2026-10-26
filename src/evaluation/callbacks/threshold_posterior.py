"""The threshold posterior with a probe as an evaluation callback, the mirror of
src/callbacks/threshold_posterior.py over the named loaders of a test loop."""

from pathlib import Path

import torch
from pytorch_lightning.callbacks import Callback

from src.callbacks.threshold_posterior import declared_betas, threshold_posterior
from src.evaluation.callbacks import utils


class ThresholdPosterior(Callback):
    """Evaluate the threshold posterior of two test sets against every probe.

    The scores of dataset_1, dataset_2 and of the probes named in
    ``posterior_config["probes"]``, among the loaders ``gaussian``, ``uniform`` and
    ``autouad`` that the dump stage appends to its validation pass when
    ``extra_sets.aux`` is set, are collected batch by batch at every test epoch. Every
    key of ``threshold_posterior`` is recorded per probe under '<name>/<probe>_<key>'
    through ``log_metrics_to_mlflow``. Nothing is plotted.

    Args:
        output_name: Name of the model output used as anomaly score.
        dataset_1: Name of the test dataloader of X1.
        dataset_2: Name of the test dataloader of X2.
        posterior_config: The posterior's configuration, as in
            ``configs/cap/posterior.yaml``. A null beta_fix of a registered probe
            raises at the first epoch end, so that the configuration composes and
            instantiates before the test bed declares it.
        name: Callback name, the prefix of the recorded scalars.
    """

    def __init__(
        self,
        output_name: str,
        dataset_1: str,
        dataset_2: str,
        posterior_config: dict,
        name: str = "threshold_posterior",
    ):
        super().__init__()
        self.output_name = output_name
        self.datasets, self.config = (dataset_1, dataset_2), posterior_config
        self.probes = list(posterior_config["probes"])
        self.name = name

    def on_test_epoch_start(self, trainer, pl_module):
        """Check that every required dataloader is present and reset the scores."""
        dset_names = list(trainer.test_dataloaders.keys())
        sets = [*self.datasets, *self.probes]
        missing = [name for name in sets if name not in dset_names]
        if missing:
            raise ValueError(
                f"{self.__class__.__name__} requires test dataloaders {missing}. "
                f"Available test dataloaders: {dset_names}"
            )
        self.scores = {name: [] for name in sets}

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Cache the scores of the two sets and of the probes."""
        dset_name = list(trainer.test_dataloaders.keys())[dataloader_idx]
        if dset_name in self.scores:
            score = torch.as_tensor(outputs[self.output_name]).detach()
            self.scores[dset_name].append(score.reshape(-1).cpu())

    def on_test_epoch_end(self, trainer, pl_module):
        """Compute the posterior of every probe at the current checkpoint and record its
        values."""
        ckpt_name = Path(pl_module._ckpt_path).stem
        utils.mlflow.log_metrics_to_mlflow(
            trainer, self._values(), ckpt_name=ckpt_name, cb_name=self.name
        )

    def _values(self) -> dict[str, float]:
        """Return every key of every probe, '<probe>_<key>'."""
        betas = declared_betas(self.config)
        scores = {name: torch.cat(batches) for name, batches in self.scores.items()}
        typical = [scores[name] for name in self.datasets]
        values = {}
        for probe, beta in betas.items():
            stats = threshold_posterior(*typical, scores[probe], beta, self.config)
            values |= {f"{probe}_{key}": value for key, value in stats.items()}
        return values
