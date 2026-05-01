#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from mlflow.tracking import MlflowClient
from mlflow.entities import Run


def sanitize_filename(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^\w.\-]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name[:200] or "unnamed"


def get_experiment(client: MlflowClient, experiment_name: str | None, experiment_id: str | None):
    if experiment_id is not None:
        exp = client.get_experiment(experiment_id)
        if exp is None:
            raise ValueError(f"Experiment id '{experiment_id}' not found.")
        return exp
    if experiment_name is None:
        raise ValueError("Provide either --experiment-name or --experiment-id.")
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment name '{experiment_name}' not found.")
    return exp


def get_run_study_name(run: Run) -> str:
    tags = run.data.tags
    for key in ("mlflow.runName", "run_name", "study_name"):
        if tags.get(key):
            return str(tags[key])
    return run.info.run_id


def list_all_runs(
    client: MlflowClient,
    experiment_id: str,
    max_results: int = 5000,
    filter_string: str = "",
) -> list[Run]:
    runs: list[Run] = []
    page_token = None
    while True:
        page = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=filter_string,
            max_results=max_results,
            page_token=page_token,
            order_by=["attributes.start_time ASC"],
        )
        runs.extend(page)
        page_token = getattr(page, "token", None)
        if not page_token:
            break
    return runs


def fetch_metric_history_df(client: MlflowClient, run_id: str, metric_name: str) -> pd.DataFrame:
    history = client.get_metric_history(run_id, metric_name)
    rows = [
        {"key": m.key, "value": m.value, "step": m.step, "timestamp": m.timestamp}
        for m in history
    ]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["step", "timestamp"]).reset_index(drop=True)
    return df


def save_run_info(run: Run, out_path: Path) -> None:
    payload = {
        "run_id": run.info.run_id,
        "run_name": run.data.tags.get("mlflow.runName"),
        "status": run.info.status,
        "artifact_uri": run.info.artifact_uri,
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
        "tags": dict(run.data.tags),
        "params": dict(run.data.params),
        "metrics_latest": dict(run.data.metrics),
    }
    out_path.write_text(json.dumps(payload, indent=2))


def compile_patterns(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p) for p in patterns]


def metric_matches(name: str, patterns: list[re.Pattern]) -> bool:
    return any(p.search(name) for p in patterns)


def select_metric_names(
    run: Run,
    include_patterns: list[re.Pattern],
    exclude_patterns: list[re.Pattern] | None = None,
) -> list[str]:
    metric_names = sorted(run.data.metrics.keys())
    selected = []
    for m in metric_names:
        if include_patterns and not metric_matches(m, include_patterns):
            continue
        if exclude_patterns and metric_matches(m, exclude_patterns):
            continue
        selected.append(m)
    return selected


def export_metrics_for_run(
    client: MlflowClient,
    run: Run,
    output_root: Path,
    include_patterns: list[re.Pattern],
    exclude_patterns: list[re.Pattern] | None = None,
    skip_missing: bool = True,
) -> None:
    study_name = sanitize_filename(get_run_study_name(run))
    run_dir = output_root / study_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_run_info(run, run_dir / "run_info.json")

    selected_metrics = select_metric_names(run, include_patterns, exclude_patterns)
    (run_dir / "selected_metrics.json").write_text(json.dumps(selected_metrics, indent=2))

    for metric_name in selected_metrics:
        df = fetch_metric_history_df(client, run.info.run_id, metric_name)
        if df.empty and skip_missing:
            continue
        metric_file = sanitize_filename(metric_name) + ".csv"
        df.to_csv(run_dir / metric_file, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export MLflow metric histories for all runs in an experiment."
    )
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--output-dir", default="mlflow_metric_export")
    parser.add_argument("--filter-string", default="")
    parser.add_argument("--include-failed", action="store_true")
    parser.add_argument("--overwrite-experiment-folder", action="store_true")
    parser.add_argument(
        "--include-regex",
        nargs="+",
        default=[r"^val/summary/"],
        help="Regex patterns for metric names to export. Default: all val/summary/* metrics.",
    )
    parser.add_argument(
        "--exclude-regex",
        nargs="*",
        default=[],
        help="Optional regex patterns for metric names to skip.",
    )
    args = parser.parse_args()

    client = MlflowClient(tracking_uri=args.tracking_uri)
    exp = get_experiment(client, args.experiment_name, args.experiment_id)

    experiment_label = sanitize_filename(exp.name or exp.experiment_id)
    experiment_dir = Path(args.output_dir) / experiment_label

    if experiment_dir.exists() and not args.overwrite_experiment_folder:
        raise FileExistsError(
            f"Output folder already exists: {experiment_dir}\n"
            "Use --overwrite-experiment-folder if intended."
        )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    filter_string = args.filter_string.strip()
    if not args.include_failed:
        status_filter = "attributes.status = 'FINISHED'"
        filter_string = f"({filter_string}) and {status_filter}" if filter_string else status_filter

    runs = list_all_runs(client, exp.experiment_id, filter_string=filter_string)
    if not runs:
        print("No runs found.")
        return

    include_patterns = compile_patterns(args.include_regex)
    exclude_patterns = compile_patterns(args.exclude_regex) if args.exclude_regex else None

    print(f"Experiment: {exp.name} ({exp.experiment_id})")
    print(f"Runs found: {len(runs)}")
    print(f"Output: {experiment_dir}")

    for i, run in enumerate(runs, start=1):
        study_name = get_run_study_name(run)
        print(f"[{i}/{len(runs)}] Exporting run {study_name} ({run.info.run_id})")
        export_metrics_for_run(
            client=client,
            run=run,
            output_root=experiment_dir,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )

    print("Done.")


if __name__ == "__main__":
    main()
