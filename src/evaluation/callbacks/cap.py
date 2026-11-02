# Compute the approximation capacity metric.
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning.callbacks import Callback

from src.evaluation.callbacks.components.cap.metric import ApproximationCapacity
from src.evaluation.callbacks.components.paired_scores import PairedScores


class CAP(Callback):
    """Compute a test approximation capacity metric between two datasets.

    This callback collects all scores from two test dataloaders, pairs the
    samples according to the selected pairing function, and computes the CAP
    metric on the paired score samples from:
        - dataset_1
        - dataset_2

    This is appropriate as a validation-side proxy objective for HPO.

    For every checkpoint the CAP(beta) curve is written to `cap_curve.npz` (arrays
    `betas`, `curve`) in the checkpoint's plot folder, unless the trainer's
    `plot_checkpoints` is False, and the maximising beta and whether it sits at the
    upper bound of the grid are logged beside CAP.

    :param output_name: Key in outputs dict containing per-event anomaly scores / losses.
    :param dataset_1: Name of the first validation dataloader.
    :param dataset_2: Name of the second validation dataloader.
    :param pairing_type: Pairing scheme name passed to get_pairing_fn.
    :param cap_metric_config: Keyword arguments used to instantiate
        ApproximationCapacity.
    :param log_raw_mlflow: Unused, accepted for config compatibility.
    :param name: String specifying the name of the callback for identification in
        later methods that manipulate callbacks.
    """

    def __init__(
        self,
        output_name: str,
        dataset_1: str,
        dataset_2: str,
        pairing_type: str,
        cap_metric_config: dict,
        log_raw_mlflow: bool = True,
        name: str = "cap",
    ):
        super().__init__()
        self.cap_metric_config = cap_metric_config
        self.name = name
        self.pairs = PairedScores(
            name, "CAP", output_name, dataset_1, dataset_2, pairing_type
        )

    def on_test_epoch_start(self, trainer, pl_module):
        """Initialise useful quantities."""
        self.pairs.reset()
        self.capmetric = ApproximationCapacity(**self.cap_metric_config)
        self.capmetric.to(pl_module.device)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Cache the data to compute approximation capacity on.

        The two data sets need to be paired and going through two data sets at once in
        not compatible with the lightning workflow, and hence we cache the whole data
        sets here.
        """
        self.pairs.collect(trainer, outputs, dataloader_idx)

    def _compute_cap(self) -> float:
        """Compute the cap metric between the two given data sets."""
        ds1_scores, ds2_scores = self.pairs.paired()
        with torch.inference_mode(False):
            with torch.enable_grad():
                # If capmetric needs gradients w.r.t. these tensors, they must require grad
                ds1 = ds1_scores.clone().requires_grad_(True)
                ds2 = ds2_scores.clone().requires_grad_(True)
                self.capmetric.update(ds1, ds2)
        return self.capmetric.compute()

    def on_test_epoch_end(self, trainer, pl_module):
        """Compute the CAP metric on the designated dataset."""
        cap_metric_value = self._compute_cap()
        curve = self._store_curve(trainer, pl_module)
        self.pairs.record(trainer, pl_module, cap_metric_value, curve)

    def _store_curve(self, trainer, pl_module) -> dict:
        """Save the checkpoint's curve; return its maximising beta and bound flag."""
        ckpt_path = Path(pl_module._ckpt_path)
        plot_folder = ckpt_path.parent / "plots" / trainer.split / ckpt_path.stem
        plot_folder = plot_folder / self.name
        if getattr(trainer, "plot_checkpoints", True):
            plot_folder.mkdir(parents=True, exist_ok=True)
            np.savez(
                plot_folder / "cap_curve.npz",
                betas=self.capmetric.betas.cpu().numpy(),
                curve=self.capmetric.curve.numpy(),
            )
        return {
            f"{self.name}_beta_star": self.capmetric.beta_star,
            f"{self.name}_at_bound": float(self.capmetric.at_bound),
        }

    def plot_summary(self, trainer, root_folder: Path):
        """Plot the CAP of every evaluated checkpoint."""
        self.pairs.plot_summary(trainer, root_folder)

    def get_optimized_metric(self):
        """Get one number that one should optimize on this callback.

        Here, it's the maximum of the summary metric across checkpoints corresponding to
        a certain checkpointing criterion.
        """
        return self.pairs.best()

    def clear_crit_summary(self):
        self.pairs.summary.clear()
