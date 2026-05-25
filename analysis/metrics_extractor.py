from pathlib import Path
import csv
import re


def safe_filename(name: str) -> str:
    """
    Convert a run name into a safe filename.
    """
    name = name.strip()
    name = re.sub(r"[^\w\-_.]", "_", name)
    return name


def read_run_name(run_dir: Path) -> str | None:
    """
    Reads the run name from:
        <run_dir>/tags/mlflow.runName
    """
    run_name_file = run_dir / "tags" / "mlflow.runName"

    if not run_name_file.exists():
        return None

    run_name = run_name_file.read_text(encoding="utf-8").strip()

    if not run_name:
        return None

    return run_name


def read_last_metric_value(metric_file: Path) -> float | None:
    """
    Reads the second column from the last row of a metric file.

    Expected metric file format:
        <column_1> <column_2> <column_3>
        <column_1> <column_2> <column_3>
        ...

    We need:
        last row, second column
    """
    try:
        lines = metric_file.read_text(encoding="utf-8").strip().splitlines()
    except UnicodeDecodeError:
        lines = metric_file.read_text(encoding="latin-1").strip().splitlines()

    if not lines:
        return None

    last_line = lines[-1].strip()

    if not last_line:
        return None

    columns = last_line.split()

    if len(columns) < 2:
        return None

    return float(columns[1])


def export_run_metrics(run_dir: Path, output_dir: Path) -> None:
    """
    Creates one CSV file for one MLflow run.
    """
    run_name = read_run_name(run_dir)

    if run_name is None:
        return

    metrics_train_dir = run_dir / "metrics" / "train"

    if not metrics_train_dir.exists():
        print(f"Skipping {run_dir}: no metrics/train directory found")
        return

    rows = []

    for metric_file in metrics_train_dir.iterdir():
        if not metric_file.is_file():
            continue

        metric_name = metric_file.name
        metric_value = read_last_metric_value(metric_file)

        if metric_value is None:
            print(f"Skipping metric {metric_file}: could not read value")
            continue

        rows.append({
            "metric_name": metric_name,
            "value": metric_value,
        })

    if not rows:
        print(f"Skipping {run_name}: no metrics found")
        return

    rows.sort(key=lambda row: row["metric_name"])

    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{safe_filename(run_name)}.csv"

    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["metric_name", "value"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created {output_file}")


def export_all_runs(
    mlruns_experiment_dir: str | Path,
    output_dir: str | Path = "exported_metrics",
) -> None:
    """
    Sweeps through all runs inside one MLflow experiment directory.

    Example:
        logs/mlflow/mlruns/307730738260916655/
    """
    mlruns_experiment_dir = Path(mlruns_experiment_dir)
    output_dir = Path(output_dir)

    if not mlruns_experiment_dir.exists():
        raise FileNotFoundError(f"Directory not found: {mlruns_experiment_dir}")

    for run_dir in mlruns_experiment_dir.iterdir():
        if not run_dir.is_dir():
            continue

        export_run_metrics(run_dir, output_dir)


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    export_all_runs(
        mlruns_experiment_dir=project_root / "logs" / "mlflow" / "mlruns" / "307730738260916655",
        output_dir=script_dir / "data" / "metrics",
    )