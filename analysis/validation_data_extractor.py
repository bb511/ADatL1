from pathlib import Path
import csv
import re


EXPECTED_METRICS = [
    "ascore_operational",
    "eff__ascore_full__brate_operational",
    "latent_entropy",
    "latent_mean_prob",
    "latent_std_prob",
    "loss",
    "loss_mean",
    "loss_mi",
    "loss_reconstruction",
]


def safe_filename(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^\w\-_.]", "_", name)
    return name


def read_run_name(run_dir: Path) -> str | None:
    run_name_file = run_dir / "tags" / "mlflow.runName"

    if not run_name_file.exists():
        return None

    run_name = run_name_file.read_text(encoding="utf-8").strip()
    return run_name if run_name else None


def read_last_metric_value(metric_file: Path) -> float | None:
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


def extract_validation_metrics_for_run(run_dir: Path) -> list[dict]:
    val_dir = run_dir / "metrics" / "val"

    if not val_dir.exists():
        return []

    rows = []

    for sample_dir in sorted(val_dir.iterdir()):
        if not sample_dir.is_dir():
            continue

        if sample_dir.name == "summary":
            continue

        row = {
            "val_sample": sample_dir.name,
        }

        for metric_name in EXPECTED_METRICS:
            metric_file = sample_dir / metric_name

            if not metric_file.exists():
                row[metric_name] = None
                print(f"Missing metric file: {metric_file}")
                continue

            metric_value = read_last_metric_value(metric_file)

            if metric_value is None:
                row[metric_name] = None
                print(f"Could not read metric value: {metric_file}")
                continue

            row[metric_name] = metric_value

        rows.append(row)

    return rows


def export_validation_metrics_for_run(run_dir: Path, output_dir: Path) -> None:
    run_name = read_run_name(run_dir)

    if run_name is None:
        return

    rows = extract_validation_metrics_for_run(run_dir)

    if not rows:
        print(f"Skipping {run_name}: no validation sample metrics found")
        return

    fieldnames = ["val_sample"] + EXPECTED_METRICS

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{safe_filename(run_name)}.csv"

    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created {output_file}")


def export_all_validation_runs(
    mlruns_experiment_dir: str | Path,
    output_dir: str | Path,
) -> None:
    mlruns_experiment_dir = Path(mlruns_experiment_dir)
    output_dir = Path(output_dir)

    if not mlruns_experiment_dir.exists():
        raise FileNotFoundError(f"Directory not found: {mlruns_experiment_dir}")

    for run_dir in sorted(mlruns_experiment_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        export_validation_metrics_for_run(run_dir, output_dir)


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

    output_dir = script_dir / "data" / "validation_metrics"

    print(f"Reading MLflow runs from: {mlruns_dir}")
    print(f"Writing validation CSV files to: {output_dir}")

    export_all_validation_runs(
        mlruns_experiment_dir=mlruns_dir,
        output_dir=output_dir,
    )