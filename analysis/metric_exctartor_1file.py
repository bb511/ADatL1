from pathlib import Path
import csv
import re


GAMMA_PATTERN = re.compile(r"Gamma_([0-9.eE+-]+)")


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


def extract_gamma_from_run_name(run_name: str) -> float | None:
    """
    Extract gamma from run names like:
        MI_Aware_AE_Run_1_Gamma_0
        MI_Aware_AE_Run_1_Gamma_1e-03
        MI_Aware_AE_Run_1_Gamma_5e-07
    """
    match = GAMMA_PATTERN.search(run_name)

    if match is None:
        return None

    return float(match.group(1))


def read_last_metric_value(metric_file: Path) -> float | None:
    """
    Reads the second column from the last row of a metric file.

    Expected metric file format:
        <column_1> <column_2> <column_3>

    We need:
        last row, second column
    """
    try:
        lines = metric_file.read_text(encoding="utf-8").strip().splitlines()
    except UnicodeDecodeError:
        lines = metric_file.read_text(encoding="latin-1").strip().splitlines()

    if not lines:
        return None

    columns = lines[-1].strip().split()

    if len(columns) < 2:
        return None

    try:
        return float(columns[1])
    except ValueError:
        return None


def extract_train_metrics_for_run(run_dir: Path) -> dict | None:
    """
    Extracts all train metrics for one MLflow run and returns one row:

        {
            "gamma": 0.001,
            "ascore_operational": ...,
            "latent_entropy": ...,
            ...
        }
    """
    run_name = read_run_name(run_dir)

    if run_name is None:
        return None

    gamma = extract_gamma_from_run_name(run_name)

    if gamma is None:
        print(f"Skipping run without Gamma in name: {run_name}")
        return None

    metrics_train_dir = run_dir / "metrics" / "train"

    if not metrics_train_dir.exists():
        print(f"Skipping {run_name}: no metrics/train directory found")
        return None

    row = {
        "gamma": gamma,
    }

    for metric_file in sorted(metrics_train_dir.iterdir()):
        if not metric_file.is_file():
            continue

        metric_name = metric_file.name
        metric_value = read_last_metric_value(metric_file)

        if metric_value is None:
            print(f"Skipping metric {metric_file}: could not read value")
            continue

        row[metric_name] = metric_value

    if len(row) == 1:
        print(f"Skipping {run_name}: no readable metrics found")
        return None

    return row


def export_all_train_metrics_by_gamma(
    mlruns_experiment_dir: str | Path,
    output_csv: str | Path,
) -> None:
    """
    Sweeps through all runs and creates one combined CSV:

        gamma,<metric_1>,<metric_2>,<metric_3>,...

    One row = one gamma/run.
    """
    mlruns_experiment_dir = Path(mlruns_experiment_dir)
    output_csv = Path(output_csv)

    if not mlruns_experiment_dir.exists():
        raise FileNotFoundError(f"Directory not found: {mlruns_experiment_dir}")

    rows = []

    for run_dir in sorted(mlruns_experiment_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        row = extract_train_metrics_for_run(run_dir)

        if row is not None:
            rows.append(row)

    if not rows:
        raise RuntimeError("No valid runs found. No CSV was created.")

    rows.sort(key=lambda row: row["gamma"])

    metric_columns = sorted(
        {
            column
            for row in rows
            for column in row.keys()
            if column != "gamma"
        }
    )

    fieldnames = ["gamma"] + metric_columns

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created combined CSV: {output_csv}")
    print(f"Rows written: {len(rows)}")
    print(f"Columns written: {len(fieldnames)}")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    mlruns_dir = (
        project_root
        / "logs"
        / "mlflow"
        / "mlruns"
        / "307730738260916655"
    )

    output_csv = script_dir / "data" / "metrics" / "train_metrics_by_gamma.csv"

    print(f"Reading MLflow runs from: {mlruns_dir}")
    print(f"Writing combined CSV to: {output_csv}")

    export_all_train_metrics_by_gamma(
        mlruns_experiment_dir=mlruns_dir,
        output_csv=output_csv,
    )