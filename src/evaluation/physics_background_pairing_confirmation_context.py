"""Authenticated 50-to-200 persistence and MLflow training-curve evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.evaluation.physics_background_pairing_confirmation_analysis import (
    _content_hash,
)

DISCOVERY_ENDPOINT_ANALYSIS_SHA256 = (
    "b8baede48d8da59aecd6acdd816120ef754cda9edfb25e4c15d89c92afec9d1d"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty context table: {path}")
    return rows


def _finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite context value: {value!r}")
    return result


def _metric_basename(logical_objective: str) -> tuple[str, str]:
    if logical_objective.startswith("native__cap__"):
        pairing = logical_objective.removeprefix("native__cap__")
        return f"cap_native_{pairing}_ema_", "maximize"
    if logical_objective == "native__wasserstein":
        return "w1dist_native_ema_", "minimize"
    if logical_objective == "native__drift":
        return "native_operational_drift_ema", "minimize"
    raise ValueError(f"Unknown confirmation objective: {logical_objective}")


def _history(path: Path) -> list[tuple[int, float]]:
    """Read MLflow's append-only metric file, retaining the latest value per step."""
    latest: dict[int, tuple[int, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        timestamp, value, step = line.split()
        parsed_step = int(step)
        parsed = (int(timestamp), _finite(value))
        if parsed_step not in latest or parsed[0] > latest[parsed_step][0]:
            latest[parsed_step] = parsed
    return [(step, latest[step][1]) for step in sorted(latest)]


def build_confirmation_context(
    confirmation_analysis_dir: Path,
    discovery_analysis_dir: Path,
    mlflow_root: Path,
) -> dict[str, Any]:
    """Build hash-bound discovery persistence and full validation trajectories."""
    confirmation_analysis_dir = confirmation_analysis_dir.resolve(strict=True)
    discovery_analysis_dir = discovery_analysis_dir.resolve(strict=True)
    mlflow_root = mlflow_root.resolve(strict=True)
    confirmation_manifest_path = confirmation_analysis_dir / "confirmation_endpoint_analysis.json"
    confirmation_manifest = json.loads(confirmation_manifest_path.read_text(encoding="utf-8"))
    digest = confirmation_manifest.pop("content_sha256", None)
    if digest != _content_hash(confirmation_manifest):
        raise ValueError("Confirmation analysis hash differs for context extraction.")
    confirmation_manifest["content_sha256"] = digest
    confirmation_table = confirmation_analysis_dir / "endpoints_across_seeds.csv"
    if confirmation_manifest["artifacts"].get(confirmation_table.name) != _sha256(
        confirmation_table
    ):
        raise ValueError("Confirmation across-seed table changed.")

    discovery_manifest_path = discovery_analysis_dir / "endpoint_analysis.json"
    if _sha256(discovery_manifest_path) != DISCOVERY_ENDPOINT_ANALYSIS_SHA256:
        raise ValueError("Discovery endpoint analysis is not the frozen authoritative package.")
    discovery_manifest = json.loads(discovery_manifest_path.read_text(encoding="utf-8"))
    discovery_table = discovery_analysis_dir / "primary_front_summary.csv"
    if discovery_manifest.get("artifact_hashes", {}).get(discovery_table.name) != _sha256(
        discovery_table
    ):
        raise ValueError("Discovery primary-front summary changed.")

    discovery_rows = [row for row in _read_csv(discovery_table) if row["score"] == "native"]
    confirmation_rows = _read_csv(confirmation_table)
    discovery_index = {
        (row["model"], row["metric"], row["strategy"] or "not_applicable"): row
        for row in discovery_rows
    }
    confirmation_index = {
        (row["model"], row["selector"], row["pairing"], row["endpoint"]): row
        for row in confirmation_rows
    }
    if len(discovery_index) != 42 or len(confirmation_index) != 84:
        raise ValueError("50-to-200 context does not contain the exact 42 native cells.")
    persistence = []
    for key in sorted(discovery_index):
        discovery = discovery_index.get(key)
        if discovery is None:
            raise ValueError(f"Discovery cell is absent: {key}")
        for endpoint, discovery_field in (
            ("efficiency_250hz", "mean_efficiency"),
            ("auprc", "mean_auprc"),
        ):
            confirmation = confirmation_index[(key[0], key[1], key[2], endpoint)]
            value_50 = _finite(discovery[discovery_field])
            value_200 = _finite(confirmation["mean"])
            persistence.append(
                {
                    "model": key[0],
                    "selector": key[1],
                    "pairing": key[2],
                    "endpoint": endpoint,
                    "mean_50_epoch": value_50,
                    "mean_200_epoch": value_200,
                    "delta_200_minus_50": value_200 - value_50,
                }
            )

    values = {
        (row["model"], row["selector"], row["pairing"], row["endpoint"]): row
        for row in persistence
    }
    hypotheses = []
    for model in sorted({key[0] for key in confirmation_index}):
        for endpoint in ("efficiency_250hz", "auprc"):
            contrasts = [
                ("cdf_cap_minus_wasserstein", ("cap", "cdf"), ("wasserstein", "not_applicable")),
                ("cdf_cap_minus_drift", ("cap", "cdf"), ("drift", "not_applicable")),
                *[
                    (f"{pairing}_minus_cdf", ("cap", pairing), ("cap", "cdf"))
                    for pairing in (
                        "flat_physical",
                        "physics_summary",
                        "typed_sliced_wasserstein",
                        "jetclr",
                    )
                ],
            ]
            for name, left, right in contrasts:
                left_row = values[(model, *left, endpoint)]
                right_row = values[(model, *right, endpoint)]
                delta_50 = left_row["mean_50_epoch"] - right_row["mean_50_epoch"]
                delta_200 = left_row["mean_200_epoch"] - right_row["mean_200_epoch"]
                outcome = (
                    "reversed"
                    if delta_50 * delta_200 < 0
                    else "survived_directionally"
                    if delta_50 * delta_200 > 0
                    else "zero_or_tied"
                )
                hypotheses.append(
                    {
                        "model": model,
                        "endpoint": endpoint,
                        "contrast": name,
                        "delta_50_epoch": delta_50,
                        "delta_200_epoch": delta_200,
                        "direction_outcome": outcome,
                    }
                )

    endpoint_receipt_path = Path(confirmation_manifest["inputs"]["endpoint_receipt"])
    if (
        _sha256(endpoint_receipt_path)
        != confirmation_manifest["inputs"]["endpoint_receipt_sha256"]
    ):
        raise ValueError("Endpoint receipt changed before curve extraction.")
    endpoint_receipt = json.loads(endpoint_receipt_path.read_text(encoding="utf-8"))
    curve_rows, curve_inputs = [], []
    for source in endpoint_receipt.get("source_markers", []):
        marker_path = Path(source["marker"]).resolve(strict=True)
        if _sha256(marker_path) != source["marker_sha256"]:
            raise ValueError("Training-curve marker changed.")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        run_id = str(marker["mlflow_run_id"])
        run_dirs = list(mlflow_root.glob(f"*/{run_id}"))
        if len(run_dirs) != 1:
            raise ValueError(f"MLflow run ID does not resolve uniquely: {run_id}")
        run_dir = run_dirs[0].resolve(strict=True)
        if (run_dir / "tags/mlflow.runName").read_text(encoding="utf-8").strip() != marker[
            "run_name"
        ]:
            raise ValueError(f"MLflow run-name identity differs: {run_id}")
        for objective in marker["row"]["logical_objectives"]:
            prefix, direction = _metric_basename(objective)
            candidates = [
                path
                for path in (run_dir / "metrics/val/summary").iterdir()
                if path.name == prefix or path.name.startswith(prefix)
            ]
            if len(candidates) != 1:
                raise ValueError(f"Metric history is not unique for {(run_id, objective)}")
            metric_path = candidates[0]
            history = _history(metric_path)
            if len(history) != 200:
                raise ValueError(
                    f"Expected 200 validation epochs for {(run_id, objective)}, found {len(history)}."
                )
            curve_inputs.append(
                {
                    "run_id": run_id,
                    "logical_objective": objective,
                    "metric_path": str(metric_path),
                    "metric_sha256": _sha256(metric_path),
                }
            )
            for epoch, (step, value) in enumerate(history, start=1):
                curve_rows.append(
                    {
                        "model": marker["row"]["model"],
                        "trial_number": marker["row"]["trial_number"],
                        "training_seed": marker["row"]["seed"],
                        "run_id": run_id,
                        "logical_objective": objective,
                        "direction": direction,
                        "epoch": epoch,
                        "mlflow_step": step,
                        "value": value,
                    }
                )
    if len(curve_inputs) != int(endpoint_receipt["cardinality"]["checkpoint_selections"]):
        raise ValueError("Training-curve objective coverage differs from checkpoint selections.")

    curve_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in curve_rows:
        curve_groups[(row["model"], row["logical_objective"])].append(row)
    curve_summary = []
    for key, rows in sorted(curve_groups.items()):
        by_epoch: dict[int, list[float]] = defaultdict(list)
        for row in rows:
            by_epoch[int(row["epoch"])].append(float(row["value"]))
        direction = rows[0]["direction"]
        mean_50 = sum(by_epoch[50]) / len(by_epoch[50])
        mean_200 = sum(by_epoch[200]) / len(by_epoch[200])
        favorable = (mean_200 - mean_50) * (1 if direction == "maximize" else -1)
        curve_summary.append(
            {
                "model": key[0],
                "logical_objective": key[1],
                "direction": direction,
                "runs": len({row["run_id"] for row in rows}),
                "mean_epoch_50": mean_50,
                "mean_epoch_200": mean_200,
                "favorable_change_50_to_200": favorable,
                "mean_last_40_change": (
                    sum(by_epoch[200]) / len(by_epoch[200])
                    - sum(by_epoch[160]) / len(by_epoch[160])
                ),
            }
        )
    return {
        "inputs": {
            "confirmation_analysis": str(confirmation_manifest_path),
            "confirmation_analysis_sha256": _sha256(confirmation_manifest_path),
            "confirmation_analysis_content_sha256": digest,
            "discovery_analysis": str(discovery_manifest_path),
            "discovery_analysis_sha256": _sha256(discovery_manifest_path),
            "endpoint_receipt": str(endpoint_receipt_path),
            "endpoint_receipt_sha256": _sha256(endpoint_receipt_path),
            "mlflow_root": str(mlflow_root),
            "metric_files": curve_inputs,
            "metric_files_content_sha256": _content_hash(curve_inputs),
        },
        "persistence": persistence,
        "hypotheses": hypotheses,
        "training_curves": curve_rows,
        "training_curve_summary": curve_summary,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty context table: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary = Path(temporary_name)
        if path.exists() and path.read_bytes() != temporary.read_bytes():
            raise FileExistsError(f"Refusing to replace different context table: {path}")
        if not path.exists():
            os.replace(temporary, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def write_confirmation_context(
    confirmation_analysis_dir: Path,
    discovery_analysis_dir: Path,
    mlflow_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    result = build_confirmation_context(
        confirmation_analysis_dir, discovery_analysis_dir, mlflow_root
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    names = ("persistence", "hypotheses", "training_curves", "training_curve_summary")
    for name in names:
        _write_csv(output_dir / f"{name}.csv", result[name])
    receipt = {
        "schema_version": 1,
        "kind": "physics_background_pairing_confirmation_context",
        "inputs": result["inputs"],
        "cardinality": {name: len(result[name]) for name in names},
        "artifacts": {f"{name}.csv": _sha256(output_dir / f"{name}.csv") for name in names},
    }
    receipt["content_sha256"] = _content_hash(receipt)
    encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    path = output_dir / "confirmation_context.json"
    if path.exists() and path.read_text(encoding="utf-8") != encoded:
        raise FileExistsError(f"Refusing to replace different context receipt: {path}")
    if not path.exists():
        path.write_text(encoded, encoding="utf-8")
    return receipt


__all__ = ["build_confirmation_context", "write_confirmation_context"]
