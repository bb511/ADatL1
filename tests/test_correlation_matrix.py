from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.evaluation.callbacks.correlation_matrix import CorrelationMatrixCallback


def test_test_epoch_end_automatically_writes_sorted_change_matrices(
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
    plot_filenames = []

    monkeypatch.setattr(
        "src.evaluation.callbacks.correlation_matrix.matrix.plot",
        lambda **kwargs: plot_filenames.append(kwargs["filename"]),
    )
    monkeypatch.setattr(
        "src.evaluation.callbacks.correlation_matrix.utils.mlflow.log_plots_to_mlflow",
        lambda *args, **kwargs: None,
    )

    callback = CorrelationMatrixCallback(
        variables=labels,
        correlation_methods=["pearson"],
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
    change_stem = "abs_reconstruction_minus_input_pearson_correlation_matrix"
    expected_sorted_files = {
        f"{change_stem}_sorted_by_increase.csv",
        f"{change_stem}_sorted_by_increase_et_only.csv",
        f"{change_stem}_sorted_by_decrease.csv",
        f"{change_stem}_sorted_by_decrease_et_only.csv",
    }

    assert expected_sorted_files <= {path.name for path in output_dir.glob("*.csv")}
    assert {path.replace(".csv", ".png") for path in expected_sorted_files} <= set(
        plot_filenames
    )


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


def test_write_correlation_matrix_variants_saves_sorted_full_and_et_matrices(
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
    plot_filenames = []

    def capture_plot(**kwargs) -> None:
        plot_filenames.append(kwargs["filename"])

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

    full_matrix = pd.read_csv(tmp_path / f"{stem}.csv", index_col=0)
    et_matrix = pd.read_csv(tmp_path / f"{stem}_et_only.csv", index_col=0)

    assert list(full_matrix.index) == ["b.Et", "a.Et", "c.Et"]
    assert list(full_matrix.columns) == ["b.Et", "a.Et", "c.Et"]
    pd.testing.assert_frame_equal(full_matrix, et_matrix)
    assert plot_filenames == [f"{stem}.png", f"{stem}_et_only.png"]


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
