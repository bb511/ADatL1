import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.evaluation.callbacks.correlation_matrix import CorrelationMatrixCallback


def test_checkpoint_selection_supports_named_root_checkpoint() -> None:
    callback = CorrelationMatrixCallback(
        correlation_methods=["pearson", "spearman"],
        ckpts={"loss_total": True},
    )

    loss_total = SimpleNamespace(
        strat_name="loss_total",
        metric_name=None,
        criterion_name=None,
    )
    last = SimpleNamespace(
        strat_name="last",
        metric_name=None,
        criterion_name=None,
    )

    assert callback._should_run_for_current_ckpt(loss_total)
    assert not callback._should_run_for_current_ckpt(last)


def test_test_epoch_end_writes_method_folders_sources_means_and_sorted_matrices(
    tmp_path: Path,
    monkeypatch,
) -> None:
    labels = ["a.Et", "b.Et", "c.Et"]
    input_table = {
        "a.Et": np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        "b.Et": np.array([0.0, 1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0]),
        "c.Et": np.array([1.0, 3.0, 0.0, 4.0, 2.0, 7.0, 5.0, 6.0]),
    }
    reconstruction_table = {
        "a.Et": np.array([0.0, 2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 7.0]),
        "b.Et": np.array([7.0, 5.0, 6.0, 3.0, 4.0, 1.0, 2.0, 0.0]),
        "c.Et": np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
    }
    plot_paths = []
    gallery_folders = []

    monkeypatch.setattr(
        "src.evaluation.callbacks.correlation_matrix.matrix.plot",
        lambda **kwargs: plot_paths.append(
            Path(kwargs["save_dir"]) / kwargs["filename"]
        ),
    )
    monkeypatch.setattr(
        "src.evaluation.callbacks.correlation_matrix.utils.mlflow.log_plots_to_mlflow",
        lambda *args, **kwargs: gallery_folders.append(Path(args[3])),
    )

    callback = CorrelationMatrixCallback(
        variables=labels,
        correlation_methods=["pearson", "spearman"],
        sensitive_variable="a.Et",
    )
    callback._active = True
    callback._resolved_variables = [{"label": label} for label in labels]
    callback._buffers = {
        "normal": {
            "input": [input_table],
            "reconstruction": [reconstruction_table],
        }
    }
    callback._event_counts = {"normal": len(input_table[labels[0]])}
    monkeypatch.setattr(callback, "_write_metadata", lambda *args, **kwargs: None)

    callback.on_test_epoch_end(
        trainer=SimpleNamespace(split="test"),
        pl_module=SimpleNamespace(_ckpt_path=tmp_path / "last.ckpt"),
    )

    output_dir = tmp_path / "plots/test/last/correlation_matrix/normal"
    assert {path.name for path in output_dir.glob("*.csv")} == {
        "input_variables.csv",
        "reconstruction_variables.csv",
    }

    for method in ("pearson", "spearman"):
        method_dir = output_dir / method.capitalize()
        change_stem = (
            f"abs_reconstruction_minus_input_{method}_correlation_matrix"
        )
        expected_sorted_plots = {
            method_dir / f"{change_stem}_sorted_by_increase.png",
            method_dir / f"{change_stem}_sorted_by_increase_et_only.png",
            method_dir / f"{change_stem}_sorted_by_decrease.png",
            method_dir / f"{change_stem}_sorted_by_decrease_et_only.png",
        }

        input_variables = pd.read_csv(method_dir / "input_variables.csv")
        reconstruction_variables = pd.read_csv(
            method_dir / "reconstruction_variables.csv"
        )
        input_correlation = pd.read_csv(
            method_dir / f"input_{method}_correlation_matrix.csv",
            index_col=0,
        )
        reconstruction_correlation = pd.read_csv(
            method_dir / f"reconstruction_{method}_correlation_matrix.csv",
            index_col=0,
        )

        assert list(input_variables.columns) == labels
        assert list(reconstruction_variables.columns) == labels
        assert list(input_correlation.index) == labels
        assert list(reconstruction_correlation.index) == labels
        assert expected_sorted_plots <= set(plot_paths)

    assert gallery_folders == [output_dir / "Pearson", output_dir / "Spearman"]

    summary = json.loads((output_dir / "mean_correlations.json").read_text())
    reconstructed = pd.DataFrame(reconstruction_table)
    expected_pearson = reconstructed.corr(method="pearson").loc[
        "a.Et", ["b.Et", "c.Et"]
    ].abs().mean()
    expected_spearman = reconstructed.corr(method="spearman").loc[
        "a.Et", ["b.Et", "c.Et"]
    ].abs().mean()
    assert summary["sensitive_variable"] == "a.Et"
    assert set(summary["spaces"]) == {"input", "reconstruction"}
    assert set(summary["spaces"]["input"]) == {"pearson", "spearman"}
    assert set(summary["spaces"]["reconstruction"]) == {"pearson", "spearman"}
    assert summary["spaces"]["reconstruction"]["pearson"][
        "num_other_variables"
    ] == 2
    assert summary["mean_pearson_correlation"] == pytest.approx(expected_pearson)
    assert summary["mean_spearman_correlation"] == pytest.approx(expected_spearman)
    assert summary["C"] == pytest.approx(
        max(max(0.0, expected_pearson), max(0.0, expected_spearman))
    )


def test_write_mean_correlations_uses_larger_absolute_mean_for_pareto_c(
    tmp_path: Path,
) -> None:
    callback = CorrelationMatrixCallback(sensitive_variable="FET.Et")
    output_path = tmp_path / "mean_correlations.json"
    callback._write_mean_correlations(
        {
            "reconstruction": {
                "pearson": {
                    "mean_correlation": 0.4,
                    "num_other_variables": 3,
                },
                "spearman": {
                    "mean_correlation": 0.25,
                    "num_other_variables": 3,
                },
            }
        },
        output_path,
    )

    summary = json.loads(output_path.read_text())
    assert summary["mean_pearson_correlation"] == 0.4
    assert summary["mean_spearman_correlation"] == 0.25
    assert summary["C"] == 0.4


def test_sort_correlation_change_matrix_orders_both_axes_by_off_diagonal_mean() -> None:
    correlation_change = pd.DataFrame(
        [
            [100.0, 0.8, -0.2],
            [0.8, -100.0, 0.1],
            [-0.2, 0.1, 50.0],
        ],
        index=["a", "b", "c"],
        columns=["a", "b", "c"],
    )

    by_increase = CorrelationMatrixCallback._sort_correlation_change_matrix(
        correlation_change,
        ascending=False,
    )
    by_decrease = CorrelationMatrixCallback._sort_correlation_change_matrix(
        correlation_change,
        ascending=True,
    )

    assert list(by_increase.index) == ["b", "a", "c"]
    assert list(by_increase.columns) == ["b", "a", "c"]
    assert list(by_decrease.index) == ["c", "a", "b"]
    assert list(by_decrease.columns) == ["c", "a", "b"]


def test_write_correlation_matrix_variants_sorts_full_and_et_matrices_without_csvs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    correlation_change = pd.DataFrame(
        [
            [0.0, 0.8, -0.2],
            [0.8, 0.0, 0.1],
            [-0.2, 0.1, 0.0],
        ],
        index=["a.Et", "b.Et", "c.Et"],
        columns=["a.Et", "b.Et", "c.Et"],
    )
    plot_calls = []

    def capture_plot(**kwargs) -> None:
        plot_calls.append(kwargs)

    monkeypatch.setattr(
        "src.evaluation.callbacks.correlation_matrix.matrix.plot",
        capture_plot,
    )

    callback = CorrelationMatrixCallback()
    stem = (
        "abs_reconstruction_minus_input_pearson_correlation_matrix_"
        "sorted_by_increase"
    )
    callback._write_correlation_matrix_variants(
        corr=correlation_change,
        plot_folder=tmp_path,
        stem=stem,
        title="Sorted correlation increase",
        sort_ascending=False,
    )

    expected_order = ["b.Et", "a.Et", "c.Et"]
    assert list(plot_calls[0]["data"].keys()) == expected_order
    assert list(plot_calls[0]["data"]["b.Et"].keys()) == expected_order
    assert list(plot_calls[1]["data"].keys()) == expected_order
    assert [call["filename"] for call in plot_calls] == [
        f"{stem}.png",
        f"{stem}_et_only.png",
    ]
    assert not list(tmp_path.glob("*.csv"))


def test_sort_correlation_change_matrix_rejects_misaligned_labels() -> None:
    correlation_change = pd.DataFrame(
        [[0.0, 0.2], [0.2, 0.0]],
        index=["a", "b"],
        columns=["b", "a"],
    )

    with pytest.raises(ValueError, match="row and column labels differ"):
        CorrelationMatrixCallback._sort_correlation_change_matrix(
            correlation_change,
            ascending=False,
        )
