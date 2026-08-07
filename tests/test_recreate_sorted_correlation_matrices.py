from pathlib import Path

import pandas as pd

from src.analysis.scripts.recreate_sorted_correlation_matrices import (
    recreate_experiment,
)


def _write_mlflow_run(experiment_dir: Path, run_id: str, run_name: str) -> None:
    run_dir = experiment_dir / run_id
    (run_dir / "tags").mkdir(parents=True)
    (run_dir / "meta.yaml").write_text(f"run_id: {run_id}\nrun_name: {run_name}\n")
    (run_dir / "tags" / "mlflow.runName").write_text(run_name)


def test_recreate_experiment_writes_each_checkpoint_target_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    experiment_id = "123456789"
    experiment_name = "physics_ae_models"
    run_name = "Run_01"
    experiment_dir = tmp_path / "mlruns" / experiment_id
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "meta.yaml").write_text(f"name: {experiment_name}\n")

    _write_mlflow_run(
        experiment_dir,
        "0" * 32,
        run_name,
    )
    _write_mlflow_run(
        experiment_dir,
        "1" * 32,
        run_name,
    )

    matrix_dir = (
        tmp_path
        / "checkpoints"
        / experiment_name
        / run_name
        / "plots/test/last/correlation_matrix/normal"
    )
    matrix_dir.mkdir(parents=True)
    labels = ["a.Et", "b.Et", "c.Et"]
    corr_before = pd.DataFrame(
        [[1.0, 0.2, 0.4], [0.2, 1.0, 0.1], [0.4, 0.1, 1.0]],
        index=labels,
        columns=labels,
    )
    corr_after = pd.DataFrame(
        [[1.0, 0.8, 0.1], [0.8, 1.0, 0.3], [0.1, 0.3, 1.0]],
        index=labels,
        columns=labels,
    )
    corr_before.to_csv(matrix_dir / "input_pearson_correlation_matrix.csv")
    corr_after.to_csv(matrix_dir / "reconstruction_pearson_correlation_matrix.csv")

    plot_filenames = []
    monkeypatch.setattr(
        "src.evaluation.callbacks.correlation_matrix.matrix.plot",
        lambda **kwargs: plot_filenames.append(kwargs["filename"]),
    )

    summary = recreate_experiment(
        experiment_id=experiment_id,
        mlruns_root=tmp_path / "mlruns",
        checkpoints_root=tmp_path / "checkpoints",
        splits=["test"],
    )

    stem = "abs_reconstruction_minus_input_pearson_correlation_matrix"
    assert (matrix_dir / f"{stem}_sorted_by_increase.csv").is_file()
    assert (matrix_dir / f"{stem}_sorted_by_decrease.csv").is_file()
    assert (matrix_dir / f"{stem}_sorted_by_increase_et_only.csv").is_file()
    assert (matrix_dir / f"{stem}_sorted_by_decrease_et_only.csv").is_file()
    assert set(plot_filenames) == {
        f"{stem}_sorted_by_increase.png",
        f"{stem}_sorted_by_decrease.png",
        f"{stem}_sorted_by_increase_et_only.png",
        f"{stem}_sorted_by_decrease_et_only.png",
    }
    assert summary.discovered_runs == 2
    assert summary.unique_run_names == 1
    assert summary.planned_targets == 0
    assert summary.recreated_targets == 1
    assert summary.duplicate_targets == 1
    assert summary.missing_targets == 0
    assert summary.failed_targets == 0


def test_recreate_experiment_dry_run_does_not_write_files(tmp_path: Path) -> None:
    experiment_id = "123456789"
    experiment_name = "physics_ae_models"
    run_name = "Run_01"
    experiment_dir = tmp_path / "mlruns" / experiment_id
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "meta.yaml").write_text(f"name: {experiment_name}\n")
    _write_mlflow_run(experiment_dir, "a" * 32, run_name)

    matrix_dir = (
        tmp_path
        / "checkpoints"
        / experiment_name
        / run_name
        / "plots/test/last/correlation_matrix/normal"
    )
    matrix_dir.mkdir(parents=True)
    labels = ["a.Et", "b.Et"]
    corr = pd.DataFrame([[1.0, 0.2], [0.2, 1.0]], index=labels, columns=labels)
    corr.to_csv(matrix_dir / "input_pearson_correlation_matrix.csv")
    corr.to_csv(matrix_dir / "reconstruction_pearson_correlation_matrix.csv")

    summary = recreate_experiment(
        experiment_id=experiment_id,
        mlruns_root=tmp_path / "mlruns",
        checkpoints_root=tmp_path / "checkpoints",
        splits=["test"],
        dry_run=True,
    )

    assert summary.planned_targets == 1
    assert summary.recreated_targets == 0
    assert not list(matrix_dir.glob("*sorted_by_*.csv"))
