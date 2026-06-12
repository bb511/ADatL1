# Callback that saves and plots correlation matrices of selected event-level variables.
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_lightning.callbacks import Callback

import matplotlib.pyplot as plt
import mplhep as hep

from src.data.utils import unpack_batch
from src.evaluation.callbacks import utils


class CorrelationMatrixCallback(Callback):
    """Save event-level variable tables and plot correlation matrices.

    This callback is intended for the post-training evaluator, in the same integration
    pattern as ReconstructionPlots. It collects selected object-level features from the
    test batches, aggregates multi-candidate objects to one scalar per event, then saves
    both the variable table and the resulting correlation matrices.

    :param variables: Variables to extract, written as '<object>.<feature>'. Singular
        aliases such as 'muon.Et' are accepted and resolved through object_feature_map.
    :param datasets: Test dataloader names to process. Empty means all dataloaders.
    :param output_name: Key in the model output dictionary containing reconstructed data.
    :param ckpts: Checkpoint selection dictionary, same convention as reco.py.
    :param aggregate: Reduction for multi-candidate object features. Supported values:
        'sum', 'mean', 'max', 'first'.
    :param correlation_methods: Pandas correlation methods to save/plot, e.g.
        ['pearson'] or ['pearson', 'spearman'].
    :param include_input: Whether to save/plot correlations of input variables.
    :param include_reconstruction: Whether to save/plot correlations of reconstructed
        variables. This is the gamma-dependent table for an autoencoder.
    :param include_residual: Whether to save/plot correlations of reco - input.
    :param max_events: Optional cap on events collected per dataset to keep CSV files
        small. None means use all events.
    :param name: Name of the callback and output subfolder.
    :param log_raw_mlflow: Whether to log generated image files to MLflow.
    """

    def __init__(
        self,
        variables: list[str] | None = None,
        datasets: list[str] | None = None,
        output_name: str = "reconstructed_data",
        ckpts: dict | None = None,
        aggregate: str = "sum",
        correlation_methods: list[str] | None = None,
        include_input: bool = True,
        include_reconstruction: bool = True,
        include_residual: bool = False,
        max_events: int | None = None,
        name: str = "correlation_matrix",
        log_raw_mlflow: bool = True,
    ):
        super().__init__()
        self.variables = variables or [
            "muons.Et",
            "jets.Et",
            "egammas.Et",
            "taus.Et",
            "FET.Et",
        ]
        self.datasets = datasets or []
        self.output_name = output_name
        self.ckpts = ckpts or {"last": True}
        self.aggregate = aggregate.lower()
        self.correlation_methods = correlation_methods or ["pearson"]
        self.include_input = include_input
        self.include_reconstruction = include_reconstruction
        self.include_residual = include_residual
        self.max_events = max_events
        self.name = name
        self.log_raw_mlflow = log_raw_mlflow

        if self.aggregate not in {"sum", "mean", "max", "first"}:
            raise ValueError(
                "aggregate must be one of {'sum', 'mean', 'max', 'first'}, "
                f"got {aggregate!r}."
            )

    def on_test_epoch_start(self, trainer, pl_module):
        """Determine whether to run and initialise per-dataset buffers."""
        self._active = self._should_run_for_current_ckpt(trainer)
        if not self._active:
            return

        self.object_feature_map = getattr(pl_module, "object_feature_map", None)
        if self.object_feature_map is None:
            raise RuntimeError(
                "object_feature_map not found on module. "
                "Make sure inject_object_feature_map(self) was called in on_test_start."
            )

        if hasattr(pl_module, "features") and not isinstance(
            pl_module.features, torch.nn.Identity
        ):
            raise RuntimeError(
                "CorrelationMatrixCallback expects the raw flattened L1 feature layout. "
                "It does not support a non-Identity pl_module.features transform."
            )

        self._resolved_variables = self._resolve_variables()
        self._buffers = {}
        self._event_counts = {}

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Collect selected variables from input and reconstruction tensors."""
        if not self._active:
            return

        dset_name = list(trainer.test_dataloaders.keys())[dataloader_idx]
        if self.datasets and dset_name not in self.datasets:
            return

        b = unpack_batch(batch)
        x = torch.flatten(b.x, start_dim=1)
        mask = None if b.mask is None else torch.flatten(b.mask, start_dim=1).bool()

        n_keep = self._num_events_to_keep(dset_name, x.size(0))
        if n_keep <= 0:
            return

        x = x[:n_keep]
        mask = None if mask is None else mask[:n_keep]

        input_table = None
        if self.include_input or self.include_residual:
            input_table = self._make_variable_table(x, mask)

        if self.include_input:
            self._append_table(dset_name, "input", input_table)

        reco_table = None
        if self.include_reconstruction or self.include_residual:
            if self.output_name not in outputs:
                raise KeyError(
                    f"outputs does not contain {self.output_name!r}. "
                    f"Available keys: {list(outputs.keys())}"
                )
            yhat = outputs[self.output_name]
            yhat = torch.flatten(yhat, start_dim=1)[:n_keep]
            reco_table = self._make_variable_table(yhat, mask)

        if self.include_reconstruction:
            self._append_table(dset_name, "reconstruction", reco_table)

        if self.include_residual:
            residual_table = {
                key: reco_table[key] - input_table[key] for key in input_table.keys()
            }
            self._append_table(dset_name, "residual", residual_table)

        self._event_counts[dset_name] = self._event_counts.get(dset_name, 0) + n_keep

    def on_test_epoch_end(self, trainer, pl_module):
        """Write tables, correlation matrices, FET summaries, and heatmaps."""
        if not self._active:
            return

        ckpts_dir = Path(pl_module._ckpt_path).parent
        ckpt_name = Path(pl_module._ckpt_path).stem
        split = trainer.split

        for dset_name, space_buffers in self._buffers.items():
            plot_folder = (
                ckpts_dir / "plots" / split / ckpt_name / self.name / dset_name
            )
            plot_folder.mkdir(parents=True, exist_ok=True)
            self._write_metadata(plot_folder, dset_name)

            for space_name, tables in space_buffers.items():
                df = self._to_dataframe(tables)
                df.to_csv(plot_folder / f"{space_name}_variables.csv", index=False)

                clean_df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
                if clean_df.empty:
                    continue

                for method in self.correlation_methods:
                    corr = clean_df.corr(method=method)
                    corr_path = plot_folder / f"{space_name}_{method}_correlation_matrix.csv"
                    corr.to_csv(corr_path)

                    self._plot_heatmap(
                        corr=corr,
                        save_path=plot_folder
                        / f"{space_name}_{method}_correlation_matrix.png",
                        title=f"{space_name} {method} correlation: {dset_name}",
                    )

                    self._write_fet_summary(
                        corr=corr,
                        save_path=plot_folder
                        / f"{space_name}_{method}_correlation_with_FET_Et.csv",
                    )

            utils.mlflow.log_plots_to_mlflow(
                trainer,
                ckpt_name,
                self.name,
                plot_folder,
                log_raw=self.log_raw_mlflow,
                gallery_name=f"{dset_name}_{self.name}",
            )

    def _resolve_variables(self) -> list[tuple[str, str, str, list[int]]]:
        """Resolve configured variable labels to object_feature_map indices."""
        resolved = []
        available = {
            f"{obj}.{feat}"
            for obj, feature_map in self.object_feature_map.items()
            for feat in feature_map.keys()
        }

        for variable in self.variables:
            if "." not in variable:
                raise ValueError(
                    f"Variable {variable!r} must have format '<object>.<feature>'."
                )

            object_key, feature_name = variable.split(".", 1)

            if object_key not in self.object_feature_map:
                raise KeyError(
                    f"Object {object_key!r} is not in object_feature_map. "
                    f"Available variables: {sorted(available)}"
                )

            feature_map = self.object_feature_map[object_key]
            if feature_name not in feature_map:
                raise KeyError(
                    f"Feature {feature_name!r} is not available for object "
                    f"{object_key!r}. Available features: {sorted(feature_map.keys())}"
                )

            resolved.append(
                (variable, object_key, feature_name, list(feature_map[feature_name]))
            )

        return resolved

    def _make_variable_table(
        self, x: torch.Tensor, mask: torch.Tensor | None
    ) -> dict[str, np.ndarray]:
        """Extract and aggregate selected variables from one flattened batch."""
        return {
            label: self._aggregate_feature(x, mask, indices).detach().cpu().numpy()
            for label, _, _, indices in self._resolved_variables
        }

    def _aggregate_feature(
        self, x: torch.Tensor, mask: torch.Tensor | None, indices: list[int]
    ) -> torch.Tensor:
        """Reduce one object feature from candidate-level values to event scalars."""
        values = x[:, indices].float()
        valid = None if mask is None else mask[:, indices]

        if self.aggregate == "first":
            out = values[:, 0]
            if valid is not None:
                out = out.masked_fill(~valid[:, 0], float("nan"))
            return out

        if valid is None:
            if self.aggregate == "sum":
                return values.sum(dim=1)
            if self.aggregate == "mean":
                return values.mean(dim=1)
            if self.aggregate == "max":
                return values.max(dim=1).values

        counts = valid.sum(dim=1)
        values_masked = values.masked_fill(~valid, 0.0)

        if self.aggregate == "sum":
            return values_masked.sum(dim=1)

        if self.aggregate == "mean":
            out = values_masked.sum(dim=1) / counts.clamp_min(1)
            return out.masked_fill(counts == 0, float("nan"))

        if self.aggregate == "max":
            minus_inf = torch.full_like(values, -float("inf"))
            out = torch.where(valid, values, minus_inf).max(dim=1).values
            return out.masked_fill(counts == 0, float("nan"))

        raise RuntimeError(f"Unhandled aggregate mode: {self.aggregate}")

    def _append_table(self, dset_name: str, space_name: str, table: dict[str, np.ndarray]):
        """Append a batch table to the in-memory buffers."""
        self._buffers.setdefault(dset_name, {})
        self._buffers[dset_name].setdefault(space_name, [])
        self._buffers[dset_name][space_name].append(table)

    def _num_events_to_keep(self, dset_name: str, batch_size: int) -> int:
        """Respect max_events while keeping input/reco rows aligned."""
        if self.max_events is None:
            return batch_size

        already_seen = self._event_counts.get(dset_name, 0)
        remaining = int(self.max_events) - already_seen
        return max(0, min(batch_size, remaining))

    def _to_dataframe(self, tables: list[dict[str, np.ndarray]]) -> pd.DataFrame:
        """Concatenate batch dictionaries into the table saved to disk."""
        columns = {}
        for label, _, _, _ in self._resolved_variables:
            columns[label] = np.concatenate([table[label] for table in tables], axis=0)
        return pd.DataFrame(columns)

    def _write_metadata(self, plot_folder: Path, dset_name: str):
        """Write a small text file documenting how the table was produced."""
        lines = [
            f"dataset: {dset_name}",
            f"aggregate: {self.aggregate}",
            f"max_events: {self.max_events}",
            "variables:",
        ]
        for label, object_key, feature_name, indices in self._resolved_variables:
            lines.append(
                f"  {label} -> object_feature_map[{object_key!r}][{feature_name!r}] "
                f"= {indices}"
            )
        (plot_folder / "metadata.txt").write_text("\n".join(lines) + "\n")

    def _write_fet_summary(self, corr: pd.DataFrame, save_path: Path):
        """Save correlations of all selected variables with FET.Et."""
        fet_label = self._find_fet_label(corr.columns)
        if fet_label is None:
            return

        summary = corr[fet_label].drop(labels=[fet_label], errors="ignore")
        summary = summary.rename("corr_with_FET.Et").to_frame()
        summary["abs_corr_with_FET.Et"] = summary["corr_with_FET.Et"].abs()
        summary = summary.sort_values("abs_corr_with_FET.Et", ascending=False)
        summary.to_csv(save_path)

    def _find_fet_label(self, labels) -> str | None:
        """Find the configured FET.Et label, allowing for exact configured spelling."""
        for label in labels:
            obj, _, feat = label.partition(".")
            if obj == "FET" and feat == "Et":
                return label
        return None

    def _plot_heatmap(self, corr: pd.DataFrame, save_path: Path, title: str):
        """Plot one correlation matrix as a PNG heatmap."""
        plt.style.use(hep.style.CMS)

        labels = list(corr.columns)
        mat = corr.to_numpy(dtype=float)
        n = len(labels)
        fig_size = max(6, 0.9 * n)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=140)

        im = ax.imshow(mat, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_title(title)

        for i in range(n):
            for j in range(n):
                value = mat[i, j]
                if np.isnan(value):
                    text = "nan"
                    color = "black"
                else:
                    text = f"{value:.2f}"
                    color = "white" if abs(value) > 0.55 else "black"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("correlation")
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        fig.clear()
        plt.close(fig)

    def _should_run_for_current_ckpt(self, trainer):
        """Determine whether this callback should run for the current checkpoint."""
        strat = getattr(trainer, "strat_name", None)
        metric = getattr(trainer, "metric_name", None)
        crit = getattr(trainer, "criterion_name", None)

        if strat is None:
            return False

        if strat == "last":
            return bool(self.ckpts.get("last", False))

        strat_cfg = self.ckpts.get(strat, None)

        if not isinstance(strat_cfg, dict) or metric is None or crit is None:
            return False

        allowed_criteria = strat_cfg.get(metric, None)

        if not isinstance(allowed_criteria, (list, tuple)):
            return False

        return crit in allowed_criteria

    def get_optimized_metric(self, ckpt_name: str, test_ds: str):
        raise NotImplementedError(
            "Callback 'CorrelationMatrixCallback' does not define an optimized metric."
        )