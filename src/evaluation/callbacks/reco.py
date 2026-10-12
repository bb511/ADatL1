# Callback that plots overlaid histograms of the input and reconstructed data.
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pathvalidate import sanitize_filename
from pytorch_lightning.callbacks import Callback

from src.evaluation.callbacks import utils
from src.plot import overlaid_hist
from src.data.utils import unpack_batch

#: Subfolder of reco/<dataset>/ with the histogram data behind the plots.
DATA_DIR = "data"


def counts_with_flow(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """``[underflow, counts per bin..., overflow]`` of ``values`` in ``edges``."""
    counts = np.histogram(values, bins=edges)[0]
    under = np.count_nonzero(values < edges[0])
    over = np.count_nonzero(values > edges[-1])
    return np.concatenate([[under], counts, [over]]).astype(np.float64)


def histogram_frame(edges: np.ndarray, input_counts: np.ndarray, reco_counts: np.ndarray) -> pd.DataFrame:
    """One row per bin, plus an underflow row first and an overflow row last."""
    low = np.concatenate([[-np.inf], edges])
    high = np.concatenate([edges, [np.inf]])
    return pd.DataFrame({"bin_low": low, "bin_high": high,
                         "input": input_counts, "reco": reco_counts})


class ReconstructionPlots(Callback):
    """Stream overlaid histograms of input vs reconstruction.

    Besides the plots, every histogram is written as data to
    ``reco/<dataset>/data/<object>_<feature>.csv`` (``bin_low``, ``bin_high``,
    ``input``, ``reco``; first row underflow, last row overflow). Those use their
    own binning: Doane edges of the INPUT values of the warmup batches only, so two
    models evaluated on the same split get identical bins and can be overlaid and
    subtracted (src/evaluation/pareto/run_vs_gamma0.py). They count every event,
    warmup batches included. The plots keep their edges from input and
    reconstruction together.

    :param output_name: String specifying the key in the output dictionary of the
        output you'd like to plot.
    :param datasets: List of strings specifying the datasets that you'd like to make the
        histogram for.
    :param ckpts: Dictionary of the checkpoints that you'd like to plot the histograms
        for.
    :param warmup_batches: Int specifying the number of batches used to determine the
        range of the histogram. If this is a float, it should be a float between 0 and
        1, since then this is interpreted as the fraction of batches that should be used
        for warmup.
    :param name: String specifying the name of the callback and the name of the gallery.
    :param log_raw_mlflow: Boolean to decide whether to log the raw plots produced by
        this callback to mlflow artifacts. Default is True. An html gallery of all the
        plots is made by default, so if this is set to false, one can still view the
        plots produced with this callback in the gallery.
    """

    def __init__(
        self,
        warmup_batches: int | float,
        output_name: str,
        ckpts: dict,
        datasets: list[str] = [],
        name: str = "reco",
        log_raw_mlflow: bool = True,
    ):
        self.warmup_batches = warmup_batches
        self.output_name = output_name
        self.log_raw_mlflow = log_raw_mlflow
        self.datasets = datasets
        self.ckpts = ckpts
        self.name = name

    def on_test_epoch_start(self, trainer, pl_module):
        """Determine if this callback should run for current ckpt and initialise q."""
        self._active = self._should_run_for_current_ckpt(trainer)
        if not self._active:
            return

        self.object_feature_map = getattr(pl_module, "object_feature_map", None)

        if self.object_feature_map is None:
            raise RuntimeError(
                "object_feature_map not found on module. "
                "Make sure inject_object_feature_map(self) called in on_test_start. "
                "This needs to happen in the current L1ADLightningModule you are using."
            )

        self._buffers = {}
        self._edges = {}
        self._hist_input = {}
        self._hist_output = {}
        self._batch_counts = {}
        # Histogram data with run-independent (input-only) binning.
        self._data_edges = {}
        self._data_input = {}
        self._data_output = {}

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Mask and stream the data to overlaid histograms."""
        if not self._active:
            return

        dset_name = list(trainer.test_dataloaders.keys())[dataloader_idx]
        if self.datasets and dset_name not in self.datasets:
            return

        b = unpack_batch(batch)
        x = b.x
        m = b.mask

        x = torch.flatten(x, start_dim=1)
        m = torch.flatten(m, start_dim=1).bool()

        if hasattr(pl_module, "features"):
            x = pl_module.features(x)

        yhat = outputs[self.output_name]

        if hasattr(pl_module, "features") and not isinstance(
            pl_module.features, torch.nn.Identity
        ):
            self._update_features(trainer, dset_name, x, yhat)
        else:
            self._update_objects(trainer, dset_name, x, yhat, m)

    def on_test_epoch_end(self, trainer, pl_module):
        """Plot the accummulated histograms for the relevant data sets."""
        if not self._active:
            return

        ckpts_dir = Path(pl_module._ckpt_path).parent
        ckpt_name = Path(pl_module._ckpt_path).stem

        for dset_name in self._edges:
            split = trainer.split
            plot_folder = (
                ckpts_dir / "plots" / split / ckpt_name / self.name / dset_name
            )
            plot_folder.mkdir(parents=True, exist_ok=True)

            for key, edges in self._edges[dset_name].items():
                c1 = self._hist_input[dset_name].get(key)
                c2 = self._hist_output[dset_name].get(key)

                if c1 is None or c2 is None:
                    continue

                obj_name, feat_name = key

                overlaid_hist.plot_streamed(
                    counts1=c1,
                    counts2=c2,
                    edges=edges,
                    obj_name=obj_name,
                    feat_name=feat_name,
                    save_dir=plot_folder,
                    label1="input",
                    label2="reco",
                )
                self._write_histogram_data(plot_folder, dset_name, key)

            utils.mlflow.log_plots_to_mlflow(
                trainer,
                ckpt_name,
                "recos",
                plot_folder,
                log_raw=self.log_raw_mlflow,
                gallery_name=f"{dset_name}_reco",
            )

    def _write_histogram_data(self, plot_folder: Path, dset_name: str, key) -> None:
        """Write the run-independent histogram of ``key`` beside its plot."""
        edges = self._data_edges.get(dset_name, {}).get(key)
        if edges is None:
            return
        obj_name, feat_name = key
        filename = sanitize_filename(f"{obj_name}_{feat_name}").replace(" ", "_")
        folder = plot_folder / DATA_DIR
        folder.mkdir(parents=True, exist_ok=True)
        histogram_frame(
            edges,
            self._data_input[dset_name][key],
            self._data_output[dset_name][key],
        ).to_csv(folder / f"{filename}.csv", index=False)

    def _warmup_n(self, trainer, dset_name):
        """Get the number of batches to use in determining the range of the histogram."""
        if isinstance(self.warmup_batches, float):
            total = len(trainer.test_dataloaders[dset_name])
            return max(1, int(self.warmup_batches * total))
        return int(self.warmup_batches)

    def _update_hist_pair(self, trainer, dset_name, key, x1, x2):
        """Update the overlaid histogram with a new batch of data."""
        self._buffers.setdefault(dset_name, {})
        self._edges.setdefault(dset_name, {})
        self._hist_input.setdefault(dset_name, {})
        self._hist_output.setdefault(dset_name, {})
        self._batch_counts.setdefault(dset_name, {})
        self._data_edges.setdefault(dset_name, {})
        self._data_input.setdefault(dset_name, {})
        self._data_output.setdefault(dset_name, {})

        if key not in self._batch_counts[dset_name]:
            self._batch_counts[dset_name][key] = 0

        x1 = x1.detach().flatten().float().cpu().numpy()
        x2 = x2.detach().flatten().float().cpu().numpy()

        x1 = x1[np.isfinite(x1)]
        x2 = x2[np.isfinite(x2)]

        if key not in self._edges[dset_name]:

            self._buffers[dset_name].setdefault(key, [])
            self._buffers[dset_name][key].append((x1, x2))
            self._batch_counts[dset_name][key] += 1

            if self._batch_counts[dset_name][key] >= self._warmup_n(trainer, dset_name):

                warm_input = np.concatenate([a for a, _ in self._buffers[dset_name][key]])
                warm_output = np.concatenate([b for _, b in self._buffers[dset_name][key]])
                vals = np.concatenate([warm_input, warm_output])
                edges = np.histogram_bin_edges(vals, bins="doane")

                self._edges[dset_name][key] = edges

                c1 = np.histogram(x1, bins=edges)[0]
                c2 = np.histogram(x2, bins=edges)[0]

                self._hist_input[dset_name][key] = c1.astype(np.float64)
                self._hist_output[dset_name][key] = c2.astype(np.float64)

                if warm_input.size:
                    data_edges = np.histogram_bin_edges(warm_input, bins="doane")
                    self._data_edges[dset_name][key] = data_edges
                    self._data_input[dset_name][key] = counts_with_flow(warm_input, data_edges)
                    self._data_output[dset_name][key] = counts_with_flow(warm_output, data_edges)

                self._buffers[dset_name][key] = []

        else:

            edges = self._edges[dset_name][key]

            self._hist_input[dset_name][key] += np.histogram(x1, bins=edges)[0]
            self._hist_output[dset_name][key] += np.histogram(x2, bins=edges)[0]

            data_edges = self._data_edges[dset_name].get(key)
            if data_edges is not None:
                self._data_input[dset_name][key] += counts_with_flow(x1, data_edges)
                self._data_output[dset_name][key] += counts_with_flow(x2, data_edges)

    def _update_objects(self, trainer, dset_name, x, yhat, m):
        """If the inputs are from object_feature_map, use that to label the hist."""
        for object_name, feature_map in self.object_feature_map.items():
            for feat_name, feat_idxs in feature_map.items():

                x_feat = x[:, feat_idxs]
                y_feat = yhat[:, feat_idxs]

                m_feat = m[:, feat_idxs]

                x_feat = x_feat[m_feat]
                y_feat = y_feat[m_feat]

                self._update_hist_pair(
                    trainer,
                    dset_name,
                    (object_name, feat_name),
                    x_feat,
                    y_feat,
                )

    def _update_features(self, trainer, dset_name, x, yhat):
        """If the inputs are some contrastive space, use that to label the hist."""
        for feat_idx in range(x.size(-1)):

            self._update_hist_pair(
                trainer,
                dset_name,
                ("noobject", f"feat_{feat_idx}"),
                x[:, feat_idx],
                yhat[:, feat_idx],
            )

    def _should_run_for_current_ckpt(self, trainer):
        """Determine whether this callback should run for the current ckpt."""
        strat = getattr(trainer, "strat_name", None)
        metric = getattr(trainer, "metric_name", None)
        crit = getattr(trainer, "criterion_name", None)

        if strat is None:
            return False

        strat_cfg = self.ckpts.get(strat, None)
        if isinstance(strat_cfg, bool):
            return strat_cfg

        if not isinstance(strat_cfg, Mapping) or metric is None or crit is None:
            return False

        allowed_criteria = strat_cfg.get(metric, None)

        if not isinstance(allowed_criteria, Sequence) or isinstance(
            allowed_criteria,
            (str, bytes),
        ):
            return False

        return crit in allowed_criteria

    def get_optimized_metric(self, ckpt_name: str, test_ds: str):
        raise NotImplementedError(
            "Callback 'ReconstructionPlots' does not have the get_optimized_metric "
            "method implemented, hence cannot return something to optuna."
        )
