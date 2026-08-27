# Callback that saves and plots correlation matrices of selected event-level variables.
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_lightning.callbacks import Callback

from src.data.utils import unpack_batch
from src.evaluation.callbacks import utils
from src.plot import matrix


CORRELATION_SOURCE_FILENAMES = {
    "input": "input_variables.csv",
    "reconstruction": "reconstruction_variables.csv",
}


class CorrelationMatrixCallback(Callback):
    """Save event-level variable tables and plot correlation matrices.

    This callback is intended for the post-training evaluator, in the same integration
    pattern as ReconstructionPlots. It collects selected object-level features from the
    test batches, aggregates multi-candidate objects to one scalar per event, then saves
    the input/reconstruction source tables and the resulting correlation matrices.

    :param variables: Variables to extract, written as '<object>.<feature>'. Singular
        aliases such as 'muon.Et' are accepted and resolved through object_feature_map.
    :param datasets: Test dataloader names to process. Empty means all dataloaders.
    :param output_name: Key in the model output dictionary containing reconstructed data.
    :param ckpts: Checkpoint selection dictionary, same convention as reco.py.
    :param aggregate: Reduction for multi-candidate object features. Supported values:
        'sum', 'mean', 'max', 'first'.
    :param correlation_methods: Pandas correlation methods to save/plot, e.g.
        ['pearson'] or ['pearson', 'spearman'].
    :param sensitive_variable: Variable whose mean correlation with every other
        configured variable is written for Pareto-front construction.
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
        sensitive_variable: str = "FET.Et",
        include_input: bool = True,
        include_reconstruction: bool = True,
        include_residual: bool = False,
        max_events: int | None = None,
        name: str = "correlation_matrix",
        log_raw_mlflow: bool = True,
    ):
        super().__init__()
        self.variables = variables or [
            "FET.phi",
            "egammas.phi",
            "jets.phi",
            "muons.phi",
            "taus.phi",
            "FET.Et",
            "egammas.Et",
            "jets.Et",
            "muons.Et",
            "taus.Et",
            "FET.eta",
            "egammas.eta",
            "jets.eta",
            "muons.eta",
            "taus.eta",
        ]
        self.datasets = datasets or []
        self.output_name = output_name
        self.ckpts = ckpts or {"last": True}
        self.aggregate = aggregate.lower()
        self.correlation_methods = correlation_methods or ["pearson"]
        self.sensitive_variable = sensitive_variable
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
        if not any(
            variable.lower() == self.sensitive_variable.lower()
            for variable in self.variables
        ):
            raise ValueError(
                f"Sensitive variable {self.sensitive_variable!r} must be included in "
                "the configured correlation variables."
            )

    def on_test_epoch_start(self, trainer, pl_module):
        """Determine whether to run and initialise per-dataset buffers."""
        self._active = self._should_run_for_current_ckpt(trainer)
        if not self._active:
            return

        self.object_feature_map = getattr(pl_module, "object_feature_map", None)
        self.control_object_feature_map = getattr(
            pl_module,
            "control_object_feature_map",
            self.object_feature_map,
        )
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
        if self.include_residual:
            control_only = [
                item["label"]
                for item in self._resolved_variables
                if item["model_indices"] is None
            ]

            if control_only:
                raise RuntimeError(
                    "Cannot compute residuals for variables that are not reconstructed: "
                    f"{control_only}. Disable include_residual or remove them."
                )
        self._buffers = {}
        self._event_counts = {}

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Collect selected variables from input, control, and reconstruction tensors."""
        if not self._active:
            return

        dset_name = list(trainer.test_dataloaders.keys())[dataloader_idx]
        if self.datasets and dset_name not in self.datasets:
            return

        b = unpack_batch(batch)

        # Model-input tensor: this is what the AE actually sees.
        # After the FET.Et exclusion, this should have 116 flattened features.
        x = torch.flatten(b.x, start_dim=1)
        mask = None if b.mask is None else torch.flatten(b.mask, start_dim=1).bool()

        # Control tensor: this is the full raw/control tensor.
        # It should still contain FET.Et, so correlation_matrix can still use FET.Et.
        needs_control_x = any(
            item["model_indices"] is None for item in self._resolved_variables
        )

        if b.control_x is None and needs_control_x:
            raise RuntimeError(
                "CorrelationMatrixCallback needs control_x for at least one control-only "
                "variable, but the batch does not contain control_x. Check that "
                "L1ADDataset.__iter__ yields the 6-tuple "
                "(x, mask, l1bit, y, control_x, control_mask)."
            )

        control_x = b.control_x if b.control_x is not None else b.x
        control_mask = b.control_mask if b.control_mask is not None else b.mask

        control_x = torch.flatten(control_x, start_dim=1)
        control_mask = (
            None
            if control_mask is None
            else torch.flatten(control_mask, start_dim=1).bool()
        )

        n_keep = self._num_events_to_keep(dset_name, x.size(0))
        if n_keep <= 0:
            return

        x = x[:n_keep]
        mask = None if mask is None else mask[:n_keep]

        control_x = control_x[:n_keep]
        control_mask = None if control_mask is None else control_mask[:n_keep]

        input_table = None
        if self.include_input or self.include_residual:
            input_table = self._make_variable_table(
                model_x=x,
                model_mask=mask,
                control_x=control_x,
                control_mask=control_mask,
                space_name="input",
            )

        if self.include_input:
            self._append_table(dset_name, "input", input_table)

        reco_table = None
        if self.include_reconstruction or self.include_residual:
            if self.output_name not in outputs:
                raise KeyError(
                    f"outputs does not contain {self.output_name!r}. "
                    f"Available keys: {list(outputs.keys())}"
                )

            # Reconstruction tensor: output of the AE.
            # This has the same layout as the 116-feature model input.
            yhat = outputs[self.output_name]
            yhat = torch.flatten(yhat, start_dim=1)[:n_keep]

            reco_table = self._make_variable_table(
                model_x=yhat,
                model_mask=mask,
                control_x=control_x,
                control_mask=control_mask,
                space_name="reconstruction",
            )

        if self.include_reconstruction:
            self._append_table(dset_name, "reconstruction", reco_table)

        if self.include_residual:
            residual_table = {
                key: reco_table[key] - input_table[key] for key in input_table.keys()
            }
            self._append_table(dset_name, "residual", residual_table)

        self._event_counts[dset_name] = self._event_counts.get(dset_name, 0) + n_keep

    def on_test_epoch_end(self, trainer, pl_module):
        """Write tables, correlation matrices, correlation changes, and heatmaps."""
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
            correlations: dict[tuple[str, str], pd.DataFrame] = {}
            space_dataframes: dict[str, pd.DataFrame] = {}

            for space_name, tables in space_buffers.items():
                df = self._to_dataframe(tables)
                space_dataframes[space_name] = df

                clean_df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
                if clean_df.empty:
                    continue

                for method in self.correlation_methods:
                    corr = self._exclude_nan_variables(clean_df.corr(method=method))
                    if corr.empty:
                        continue

                    correlations[(space_name, method)] = corr

            # Keep one parent-level copy for existing analysis utilities and write a
            # copy into every method folder so each correlation result is standalone.
            self._write_correlation_source_tables(space_dataframes, plot_folder)

            mean_correlations: dict[str, dict[str, dict[str, float | int]]] = {}
            for method in self.correlation_methods:
                method_name = method.capitalize()
                method_folder = plot_folder / method_name
                method_folder.mkdir(parents=True, exist_ok=True)
                self._write_correlation_source_tables(
                    space_dataframes,
                    method_folder,
                )

                for space_name in space_dataframes:
                    corr = correlations.get((space_name, method))
                    if corr is None:
                        continue

                    corr.to_csv(
                        method_folder
                        / f"{space_name}_{method}_correlation_matrix.csv"
                    )

                    mean_value, num_other_variables = (
                        self._mean_sensitive_variable_correlation(corr)
                    )
                    mean_correlations.setdefault(space_name, {})[method] = {
                        "mean_correlation": mean_value,
                        "num_other_variables": num_other_variables,
                    }

                    title = {
                        "input": f"{method_name} correlation matrix before training",
                        "reconstruction": (
                            f"{method_name} correlation matrix after training"
                        ),
                    }.get(
                        space_name,
                        f"{method_name} correlation matrix: {space_name}",
                    )

                    self._write_correlation_matrix_variants(
                        corr=corr,
                        plot_folder=method_folder,
                        stem=f"{space_name}_{method}_correlation_matrix",
                        title=title,
                    )

                corr_before = correlations.get(("input", method))
                corr_after = correlations.get(("reconstruction", method))
                if corr_before is not None and corr_after is not None:
                    common_labels = [
                        label
                        for label in corr_before.index
                        if label in corr_after.index
                    ]
                    if not common_labels:
                        continue

                    correlation_change = corr_after.loc[
                        common_labels,
                        common_labels,
                    ].abs()
                    correlation_change -= corr_before.loc[
                        common_labels,
                        common_labels,
                    ].abs()
                    correlation_change = self._exclude_nan_variables(
                        correlation_change
                    )
                    if correlation_change.empty:
                        continue

                    change_stem = (
                        f"abs_reconstruction_minus_input_{method}_correlation_matrix"
                    )

                    self._write_correlation_matrix_variants(
                        corr=correlation_change,
                        plot_folder=method_folder,
                        stem=change_stem,
                        title=(
                            f"Change in {method_name} correlation: "
                            "|corr_after| - |corr_before|"
                        ),
                    )

                    for direction, ascending in (
                        ("increase", False),
                        ("decrease", True),
                    ):
                        self._write_correlation_matrix_variants(
                            corr=correlation_change,
                            plot_folder=method_folder,
                            stem=f"{change_stem}_sorted_by_{direction}",
                            title=(
                                f"Change in {method_name} correlation: "
                                f"variables sorted by mean {direction}"
                            ),
                            sort_ascending=ascending,
                        )

                utils.mlflow.log_plots_to_mlflow(
                    trainer,
                    ckpt_name,
                    f"{self.name}/{method_name}",
                    method_folder,
                    log_raw=self.log_raw_mlflow,
                    gallery_name=f"{dset_name}_{self.name}_{method}",
                )

            self._write_mean_correlations(
                mean_correlations,
                plot_folder / "mean_correlations.json",
            )

        # The source tables can contain millions of event values. They are no longer
        # needed after the CSVs and summaries have been written, and retaining them
        # here keeps the entire validation sample alive while the test split is loaded.
        self._buffers.clear()
        self._event_counts.clear()

    def _resolve_variables(self) -> list[dict]:
        """Resolve labels against model-input and full/control feature maps."""
        resolved = []

        available = {
            f"{obj}.{feat}"
            for feature_map_source in (
                self.object_feature_map,
                self.control_object_feature_map,
            )
            for obj, feature_map in feature_map_source.items()
            for feat in feature_map.keys()
        }

        for variable in self.variables:
            model_resolved = self._resolve_variable_in_map(
                variable,
                self.object_feature_map,
            )

            control_resolved = self._resolve_variable_in_map(
                variable,
                self.control_object_feature_map,
            )

            if model_resolved is None and control_resolved is None:
                raise KeyError(
                    f"Variable {variable!r} is not available in the model-input or "
                    f"control feature maps. Available variables: {sorted(available)}"
                )

            reference = model_resolved or control_resolved

            resolved.append(
                {
                    "label": variable,
                    "object_key": reference[0],
                    "feature_name": reference[1],
                    "model_indices": None if model_resolved is None else model_resolved[2],
                    "control_indices": (
                        None if control_resolved is None else control_resolved[2]
                    ),
                }
            )

        return resolved
    
    def _resolve_variable_in_map(
        self,
        variable: str,
        object_feature_map: dict,
    ) -> tuple[str, str, list[int]] | None:
        if "." not in variable:
            raise ValueError(
                f"Variable {variable!r} must have format '<object>.<feature>'."
            )

        requested_object, requested_feature = variable.split(".", 1)

        object_key = None
        for candidate in object_feature_map.keys():
            if str(candidate).lower() == requested_object.lower():
                object_key = candidate
                break

        if object_key is None:
            return None

        feature_map = object_feature_map[object_key]

        feature_key = None
        for candidate in feature_map.keys():
            if str(candidate).lower() == requested_feature.lower():
                feature_key = candidate
                break

        if feature_key is None:
            return None

        return object_key, feature_key, list(feature_map[feature_key])

    def _make_variable_table(
        self,
        model_x: torch.Tensor,
        model_mask: torch.Tensor | None,
        control_x: torch.Tensor,
        control_mask: torch.Tensor | None,
        space_name: str,
    ) -> dict[str, np.ndarray]:
        """Extract and aggregate selected variables from one flattened batch."""
        table = {}

        for item in self._resolved_variables:
            indices = item["model_indices"]
            x = model_x
            mask = model_mask

            if indices is None:
                indices = item["control_indices"]
                x = control_x
                mask = control_mask

            if indices is None:
                raise RuntimeError(f"Could not resolve indices for {item['label']!r}.")

            table[item["label"]] = (
                self._aggregate_feature(x, mask, indices).detach().cpu().numpy()
            )

        return table

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
        for item in self._resolved_variables:
            label = item["label"]
            columns[label] = np.concatenate([table[label] for table in tables], axis=0)
        return pd.DataFrame(columns)

    @staticmethod
    def _exclude_nan_variables(corr: pd.DataFrame) -> pd.DataFrame:
        """Remove variables that would leave NaN values in a correlation matrix."""
        corr = corr.replace([np.inf, -np.inf], np.nan)

        while corr.isna().to_numpy().any():
            nan_counts = corr.isna().sum(axis=0) + corr.isna().sum(axis=1)
            label = nan_counts.idxmax()
            corr = corr.drop(index=label, columns=label)

        return corr

    def _write_metadata(self, plot_folder: Path, dset_name: str):
        """Write a small text file documenting how the table was produced."""
        lines = [
            f"dataset: {dset_name}",
            f"aggregate: {self.aggregate}",
            f"max_events: {self.max_events}",
            f"sensitive_variable: {self.sensitive_variable}",
            "variables:",
        ]
        for item in self._resolved_variables:
            model_indices = item["model_indices"]
            control_indices = item["control_indices"]

            if model_indices is None:
                source = "control only; not reconstructed"
                shown_indices = control_indices
            else:
                source = "model input/reconstruction"
                shown_indices = model_indices

            lines.append(
                f"  {item['label']} -> {source}; "
                f"object={item['object_key']!r}, feature={item['feature_name']!r}, "
                f"indices={shown_indices}, control_indices={control_indices}"
            )
        (plot_folder / "metadata.txt").write_text("\n".join(lines) + "\n")

    def _mean_sensitive_variable_correlation(
        self,
        corr: pd.DataFrame,
    ) -> tuple[float, int]:
        """Mean absolute correlation of the sensitive variable with all others."""
        matching_labels = [
            label
            for label in corr.index
            if str(label).lower() == self.sensitive_variable.lower()
        ]
        if not matching_labels:
            raise ValueError(
                f"Sensitive variable {self.sensitive_variable!r} is missing from "
                f"the correlation matrix. Available variables: {list(corr.index)}"
            )

        sensitive_label = matching_labels[0]
        other_labels = [label for label in corr.columns if label != sensitive_label]
        if not other_labels:
            raise ValueError(
                "At least one non-sensitive variable is required to calculate the "
                "mean sensitive-variable correlation."
            )

        values = corr.loc[sensitive_label, other_labels].dropna()
        if values.empty:
            raise ValueError(
                f"No finite correlations remain for {self.sensitive_variable!r}."
            )

        return float(values.abs().mean()), int(values.size)

    def _write_mean_correlations(
        self,
        mean_correlations: dict[str, dict[str, dict[str, float | int]]],
        output_path: Path,
    ) -> None:
        """Write sensitive-variable means and reconstructed Pareto metric ``C``."""
        reconstruction = mean_correlations.get("reconstruction", {})
        required_methods = ("pearson", "spearman")
        missing = [
            method for method in required_methods if method not in reconstruction
        ]
        if missing:
            raise ValueError(
                "Cannot calculate Pareto metric C because reconstructed mean "
                f"correlations are missing for: {missing}."
            )

        mean_pearson = float(reconstruction["pearson"]["mean_correlation"])
        mean_spearman = float(reconstruction["spearman"]["mean_correlation"])
        pareto_c = max(max(0.0, mean_pearson), max(0.0, mean_spearman))

        payload = {
            "sensitive_variable": self.sensitive_variable,
            "definition": (
                "Arithmetic mean of absolute correlations between the sensitive "
                "variable and every other variable; self-correlation is excluded."
            ),
            "spaces": mean_correlations,
            "mean_pearson_correlation": mean_pearson,
            "mean_spearman_correlation": mean_spearman,
            "C": pareto_c,
            "C_definition": (
                "max(max(0, mean_pearson_correlation), "
                "max(0, mean_spearman_correlation)) using reconstruction means"
            ),
        }
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def _write_correlation_matrix_variants(
        self,
        corr: pd.DataFrame,
        plot_folder: Path,
        stem: str,
        title: str,
        sort_ascending: bool | None = None,
    ) -> None:
        """Save full-variable and ``*.Et``-only PNG correlation matrices."""
        variants = [("", corr, 1.0)]

        et_labels = [label for label in corr.columns if str(label).endswith(".Et")]
        if not et_labels:
            raise RuntimeError(
                "Cannot create the required *.Et-only correlation matrix because "
                "the configured correlation variables contain no labels ending in '.Et'."
            )

        et_corr = corr.loc[et_labels, et_labels]
        variants.append(("_et_only", et_corr, 0.6))

        for suffix, variant, figure_scale in variants:
            if sort_ascending is not None:
                variant = self._sort_correlation_change_matrix(
                    variant,
                    ascending=sort_ascending,
                )

            variant_stem = f"{stem}{suffix}"
            matrix.plot(
                data=variant.to_dict(orient="index"),
                value_name=title,
                save_dir=plot_folder,
                cmap="coolwarm",
                vmin=-1.0,
                vmax=1.0,
                filename=f"{variant_stem}.png",
                figure_scale=figure_scale,
            )

    @staticmethod
    def _sort_correlation_change_matrix(
        corr: pd.DataFrame,
        ascending: bool,
    ) -> pd.DataFrame:
        """Order both axes by each variable's mean off-diagonal correlation change.

        Positive scores mean that a variable became more strongly correlated on
        average after reconstruction; negative scores mean that it became less
        strongly correlated. The diagonal is excluded because self-correlation does
        not describe a relationship between variables.
        """
        if list(corr.index) != list(corr.columns):
            raise ValueError(
                "Cannot sort a correlation-change matrix whose row and column "
                "labels differ."
            )

        off_diagonal = corr.copy()
        np.fill_diagonal(off_diagonal.values, np.nan)
        mean_change = off_diagonal.mean(axis=1).fillna(0.0)
        ordered_labels = mean_change.sort_values(
            ascending=ascending,
            kind="stable",
        ).index

        return corr.loc[ordered_labels, ordered_labels]

    @staticmethod
    def _write_correlation_source_tables(
        space_dataframes: dict[str, pd.DataFrame],
        plot_folder: Path,
    ) -> None:
        """Save the input and reconstruction event tables as the two source CSVs."""
        for space, filename in CORRELATION_SOURCE_FILENAMES.items():
            dataframe = space_dataframes.get(space)
            if dataframe is not None:
                dataframe.to_csv(plot_folder / filename, index=False)

    def _should_run_for_current_ckpt(self, trainer):
        """Determine whether this callback should run for the current checkpoint."""
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
            "Callback 'CorrelationMatrixCallback' does not define an optimized metric."
        )
