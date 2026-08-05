#!/usr/bin/env python3
"""Export and validate the physics background-pairing campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle  # nosec B403 -- restricted to authenticated campaign checkpoint roots
import shutil
import statistics
import subprocess  # nosec B404 -- fixed read-only git commands
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_ORDER = ("ae", "vae", "dsae", "dsvae", "svdd", "realnvp")
TARGET_RATE_KHZ = 0.25
BASE_RATE_KHZ = 28608.8064
EXPECTED_FRONTS = 56
EXPECTED_ANOMALIES = 20
TARGET_TRIALS = 600
CAP_STRATEGIES = (
    "flat_physical",
    "physics_summary",
    "typed_sliced_wasserstein",
    "jetclr",
    "cdf",
)
SCORES_BY_MODEL = {
    "ae": ("native", "residual_oas"),
    "vae": ("native", "residual_oas"),
    "dsae": ("native",),
    "dsvae": ("native",),
    "svdd": ("native",),
    "realnvp": ("native",),
}
DOWNSTREAM_DATASETS = (
    "GluGluHTo2B_Par-MH-125",
    "GluGluHto2G_Par-MH-125",
    "GluGluHto2G_Par-MH-90",
    "GluGluHto2Tau_Par-MH-125",
    "GluGlutoHHto2B2WtoLNu2Q_Par-c2-0-kl-1-kt-1",
    "HHHto4B2Tau_Par-c3-0-d4-0",
    "HHHto6B_Par-c3-0-d4-0",
    "SUSYGluGluToBBHTo2B_Par-M-1200",
    "SUSYGluGluToBBHToBB_Par-M-120",
    "SUSYGluGluToBBHToBB_Par-M-350",
    "SUSYGluGluToBBHToBB_Par-M-600",
    "TTHTo2C_Par-MH-125",
    "TTHto2B_Par-MH-125",
    "VBFHTo2C_Par-MH-125",
    "VBFHto2B_Par-MH-125",
    "VBFHto2Tau_Par-MH-125",
    "WtoTauto3Mu",
    "ggH-suep-decay",
    "smj-case-A",
    "haa-4b-ma15",
)


def parse_args() -> argparse.Namespace:
    """Parse one read-only export mode and its explicit campaign root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("command", choices=("snapshot", "finalize"))
    return parser.parse_args()


def _sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    """Load one required JSON object."""
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    """Atomically write deterministic JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    """Atomically write rows with an explicit stable CSV schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    temporary.replace(path)


def _json_cell(value: Any) -> str:
    """Encode a nested value for one deterministic CSV cell."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _relative(path: Path, root: Path) -> str:
    """Prefer a campaign-relative path and retain external provenance paths."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _require_inside(path: Path, roots: Iterable[Path]) -> Path:
    """Resolve a path and require it to remain inside a trusted campaign root."""
    resolved = path.resolve(strict=True)
    if not any(resolved.is_relative_to(root.resolve(strict=True)) for root in roots):
        raise RuntimeError(f"Refusing to read campaign pickle outside trusted roots: {resolved}")
    return resolved


def _design(root: Path) -> dict[str, Any]:
    """Load and authenticate the immutable six-study design."""
    design = _read_json(root / "design.json")
    if design.get("schema_version") != 2 or design.get("shared_trial_pool") is not True:
        raise RuntimeError("Expected a shared-pool schema-2 campaign design.")
    if tuple(design.get("models", ())) != MODEL_ORDER:
        raise RuntimeError("Campaign model order differs from the six-study contract.")
    if len(design.get("studies", [])) != len(MODEL_ORDER):
        raise RuntimeError("Campaign design does not contain exactly six studies.")
    expected_scores = {model: list(scores) for model, scores in SCORES_BY_MODEL.items()}
    if (
        int(design.get("target_trials_per_study", -1)) != TARGET_TRIALS
        or int(design.get("search_epochs", -1)) != 50
        or int(design.get("retrain_epochs", -1)) != 200
        or design.get("scores_by_model") != expected_scores
        or tuple(design.get("cap_strategies", ())) != CAP_STRATEGIES
    ):
        raise RuntimeError("Campaign budgets, scores, or CAP strategies differ from the contract.")
    for study in design["studies"]:
        expected = []
        for score in SCORES_BY_MODEL[study["model"]]:
            expected.extend(
                (f"{score}__cap__{strategy}", ["maximize", "minimize"])
                for strategy in CAP_STRATEGIES
            )
            expected.extend(
                (
                    (f"{score}__wasserstein", ["minimize", "minimize"]),
                    (f"{score}__drift", ["minimize", "minimize"]),
                )
            )
        observed = [
            (objective["id"], objective["directions"]) for objective in study["objectives"]
        ]
        if observed != expected or [
            index for objective in study["objectives"] for index in objective["value_indices"]
        ] != list(range(2 * len(expected))):
            raise RuntimeError(f"Objective contract differs for {study['model']}.")
    preflight = Path(design["preflight"])
    if _sha256(preflight) != design["preflight_sha256"]:
        raise RuntimeError("Pairing preflight hash differs from design.json.")
    pairing = _read_json(preflight)
    expected_tables = {
        (stage, strategy)
        for stage in ("validate", "test")
        for strategy in (
            "flat_physical",
            "physics_summary",
            "typed_sliced_wasserstein",
            "jetclr",
        )
    }
    observed_tables = {
        (str(row.get("stage")), str(row.get("strategy"))) for row in pairing.get("tables", [])
    }
    if (
        pairing.get("dataset_1") != "ZB_run396102"
        or pairing.get("dataset_2") != "ZB_run398183"
        or int(pairing.get("events_per_source", -1)) != 163840
        or observed_tables != expected_tables
        or any(
            row.get("authenticated") is not True or int(row.get("pairs", -1)) != 163840
            for row in pairing.get("tables", [])
        )
    ):
        raise RuntimeError("Pairing preflight content violates the E/G table contract.")
    return design


def _load_study(root: Path, study: dict[str, Any]):
    """Load one existing Optuna study without creating missing state."""
    database = root / "studies" / f"{study['model']}.db"
    if not database.is_file():
        return None
    import optuna

    return optuna.load_study(study_name=study["study_name"], storage=f"sqlite:///{database}")


def _usable(trial: Any, expected_values: int) -> bool:
    """Return whether a trial is complete with a full finite objective vector."""
    return (
        trial.state.name == "COMPLETE"
        and trial.values is not None
        and len(trial.values) == expected_values
        and all(math.isfinite(float(value)) for value in trial.values)
    )


def _trial_exports(
    root: Path, design: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build study-status and all-trial portable rows."""
    trial_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    target = int(design["target_trials_per_study"])
    max_objectives = max(2 * len(study["objectives"]) for study in design["studies"])
    for study in design["studies"]:
        expected = 2 * len(study["objectives"])
        loaded = _load_study(root, study)
        trials = [] if loaded is None else list(loaded.get_trials(deepcopy=False))
        usable_numbers = [int(trial.number) for trial in trials if _usable(trial, expected)][
            :target
        ]
        selected = set(usable_numbers)
        states = Counter(trial.state.name.lower() for trial in trials)
        durations = [
            float(trial.duration.total_seconds()) for trial in trials if trial.duration is not None
        ]
        status_rows.append(
            {
                "model": study["model"],
                "study_name": study["study_name"],
                "logical_objectives": len(study["objectives"]),
                "target": target,
                "total": len(trials),
                "waiting": states["waiting"],
                "running": states["running"],
                "complete": states["complete"],
                "pruned": states["pruned"],
                "fail": states["fail"],
                "usable_complete": sum(_usable(trial, expected) for trial in trials),
                "selected_for_freeze": len(selected),
                "duration_count": len(durations),
                "duration_total_seconds": sum(durations),
                "duration_median_seconds": statistics.median(durations) if durations else "",
            }
        )
        for trial in sorted(trials, key=lambda item: int(item.number)):
            values = None if trial.values is None else [float(value) for value in trial.values]
            row = {
                "model": study["model"],
                "study_name": study["study_name"],
                "trial_number": int(trial.number),
                "state": trial.state.name,
                "usable": _usable(trial, expected),
                "selected_for_freeze": int(trial.number) in selected,
                "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else "",
                "datetime_complete": (
                    trial.datetime_complete.isoformat() if trial.datetime_complete else ""
                ),
                "duration_seconds": trial.duration.total_seconds() if trial.duration else "",
                "params_json": _json_cell(trial.params),
                "values_json": _json_cell(values),
                "user_attrs_json": _json_cell(trial.user_attrs),
                "system_attrs_json": _json_cell(trial.system_attrs),
            }
            for index in range(max_objectives):
                row[f"objective_{index:03d}"] = (
                    "" if values is None or index >= len(values) else values[index]
                )
            trial_rows.append(row)
    return status_rows, trial_rows


def _binding_index(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Load the unique explicit Optuna-to-Hydra binding for every search run."""
    bindings: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted((root / "sweeps").glob("*/*/trial_bindings.json")):
        payload = _read_json(path)
        for row in payload.get("bindings", []):
            key = (str(row["model"]), int(row["trial_number"]))
            if key in bindings:
                raise RuntimeError(f"Duplicate search binding for {key}.")
            row = {**row, "binding_artifact": str(path.resolve())}
            artifact = row.get("optimized_metric_artifact")
            if artifact:
                artifact_path = Path(artifact)
                if _sha256(artifact_path) != row["optimized_metric_artifact_sha256"]:
                    raise RuntimeError(f"Changed optimized-metric artifact: {artifact_path}")
            bindings[key] = row
    return bindings


def _validate_bindings(
    root: Path, design: dict[str, Any], bindings: dict[tuple[str, int], dict[str, Any]]
) -> None:
    """Cross-check every explicit binding against Optuna and run provenance."""
    for study in design["studies"]:
        loaded = _load_study(root, study)
        trials = (
            {} if loaded is None else {int(row.number): row for row in loaded.get_trials(False)}
        )
        for (model, trial_number), binding in bindings.items():
            if model != study["model"]:
                continue
            if trial_number not in trials:
                raise RuntimeError(
                    f"Binding references missing Optuna trial {(model, trial_number)}."
                )
            trial = trials[trial_number]
            values = None if trial.values is None else [float(value) for value in trial.values]
            if (
                binding["study_name"] != study["study_name"]
                or binding["state"] != trial.state.name
                or binding["params"] != trial.params
                or binding["values"] != values
            ):
                raise RuntimeError(f"Binding differs from Optuna trial {(model, trial_number)}.")
            artifact_name = binding.get("optimized_metric_artifact")
            if artifact_name is None:
                continue
            artifact_path = _require_inside(Path(artifact_name), [root / "sweeps"])
            artifact = _read_json(artifact_path)
            if artifact.get("optimized_metric") != values:
                raise RuntimeError(
                    f"Bound artifact values differ from trial {(model, trial_number)}."
                )
            provenance = artifact.get("run_provenance", {})
            hydra_dir = Path(binding["hydra_run_dir"]).resolve()
            checkpoint_dir = Path(provenance.get("checkpoint_run_dir", ""))
            if (
                int(provenance.get("hydra_job_num", -1)) != int(binding["hydra_job_num"])
                or Path(provenance.get("hydra_output_dir", "")).resolve() != hydra_dir
                or artifact_path.parent != hydra_dir
                or not checkpoint_dir.resolve().is_relative_to(
                    (root / "runtime" / "checkpoints").resolve()
                )
                or not any(logger.get("run_id") for logger in provenance.get("loggers", []))
            ):
                raise RuntimeError(f"Incomplete run provenance for trial {(model, trial_number)}.")


def _checkpoint_rows(
    root: Path,
    design: dict[str, Any],
    *,
    include_retrains: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Export selected checkpoints and embedded MLflow run identities."""
    objectives = {
        (study["model"], objective["id"]): objective
        for study in design["studies"]
        for objective in study["objectives"]
    }
    sources: list[tuple[str, str, int, int | None, Path]] = []
    for (model, trial_number), binding in _binding_index(root).items():
        artifact = binding.get("optimized_metric_artifact")
        if artifact:
            sources.append(("search", model, trial_number, None, Path(artifact)))
    if include_retrains:
        marker_dir = root / "retraining" / "markers"
        for marker_path in sorted(marker_dir.glob("*.json")):
            marker = _read_json(marker_path)
            row = marker["row"]
            sources.append(
                (
                    "retrain",
                    str(row["model"]),
                    int(row["trial_number"]),
                    int(row["retrain_index"]),
                    Path(marker["optimized_metric_artifact"]),
                )
            )
    checkpoint_rows = []
    mlflow_rows = []
    for phase, model, trial_number, retrain_index, path in sources:
        artifact = _read_json(path)
        expected_order = [
            objective["id"]
            for study in design["studies"]
            if study["model"] == model
            for objective in study["objectives"]
        ]
        if (
            artifact.get("schema_version") != 2
            or artifact.get("objective_order") != expected_order
        ):
            raise RuntimeError(f"Objective contract mismatch: {path}")
        provenance = artifact.get("run_provenance", {})
        for objective_id in expected_order:
            selection = artifact["selections"][objective_id]
            pair = selection["optimized_metric"]
            if len(pair) != 2 or not all(math.isfinite(float(value)) for value in pair):
                raise RuntimeError(f"Invalid selected objective pair in {path}: {objective_id}")
            objective = objectives[(model, objective_id)]
            checkpoint_rows.append(
                {
                    "phase": phase,
                    "model": model,
                    "trial_number": trial_number,
                    "retrain_index": "" if retrain_index is None else retrain_index,
                    "logical_objective": objective_id,
                    "score": objective["score"],
                    "metric": objective["metric"],
                    "strategy": objective["strategy"] or "",
                    "optimized_ckpt_name": selection["optimized_ckpt_name"],
                    "q_prime": float(pair[0]),
                    "q_double_prime": float(pair[1]),
                    "q_prime_direction": objective["directions"][0],
                    "q_double_prime_direction": objective["directions"][1],
                    "artifact": _relative(path, root),
                    "artifact_sha256": _sha256(path),
                    "hydra_run_dir": provenance.get("hydra_output_dir", ""),
                    "mlflow_run_id": next(
                        (
                            logger.get("run_id")
                            for logger in provenance.get("loggers", [])
                            if logger.get("run_id")
                        ),
                        "",
                    ),
                }
            )
        for logger in provenance.get("loggers", []):
            if not logger.get("run_id"):
                continue
            mlflow_rows.append(
                {
                    "phase": phase,
                    "model": model,
                    "trial_number": trial_number,
                    "retrain_index": "" if retrain_index is None else retrain_index,
                    "experiment_name": provenance.get("experiment_name", ""),
                    "experiment_id": logger.get("experiment_id", ""),
                    "run_id": logger["run_id"],
                    "run_name": provenance.get("run_name", ""),
                    "hydra_run_dir": provenance.get("hydra_output_dir", ""),
                    "checkpoint_run_dir": provenance.get("checkpoint_run_dir", ""),
                    "artifact": _relative(path, root),
                }
            )
    checkpoint_rows.sort(
        key=lambda row: (
            row["phase"],
            MODEL_ORDER.index(row["model"]),
            int(row["trial_number"]),
            row["logical_objective"],
        )
    )
    return checkpoint_rows, mlflow_rows


def _copy_state_ledgers(root: Path, output: Path) -> None:
    """Copy deterministic operator ledgers or create explicit empty schemas."""
    job_fields = [
        "job_id",
        "phase",
        "model_or_index",
        "submitted_at",
        "predicted_start",
        "predicted_completion",
        "state",
        "elapsed",
        "exit_code",
        "retry",
    ]
    source_jobs = root / "state" / "job_history.csv"
    if source_jobs.is_file():
        shutil.copyfile(source_jobs, output / "job_history.csv")
    else:
        _write_csv(output / "job_history.csv", [], job_fields)
    source_failures = root / "state" / "failures.jsonl"
    destination = output / "failures.jsonl"
    if source_failures.is_file():
        shutil.copyfile(source_failures, destination)
    else:
        destination.write_text("", encoding="utf-8")


def _git_state() -> tuple[str, bool, list[str]]:
    """Return the reporting checkout commit and clean-state evidence."""
    git = shutil.which("git")
    if git is None:
        return "", False, ["git-unavailable"]
    commit = subprocess.run(  # nosec B603
        [git, "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(  # nosec B603
        [git, "status", "--porcelain"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return commit, not dirty, dirty


def _provenance(root: Path, design: dict[str, Any], *, final: bool) -> dict[str, Any]:
    """Build portable code, input, budget, and source-artifact provenance."""
    commit, clean, dirty = _git_state()
    databases = {}
    for model in MODEL_ORDER:
        path = root / "studies" / f"{model}.db"
        databases[model] = _sha256(path) if final and path.is_file() else None
    value = {
        "schema_version": 1,
        "campaign_root": str(root.resolve()),
        "code_commit": design["code_commit"],
        "report_checkout_commit": commit,
        "report_checkout_clean": clean,
        "report_checkout_dirty_entries": dirty,
        "design_sha256": _sha256(root / "design.json"),
        "preflight_sha256": design["preflight_sha256"],
        "uv_lock_sha256": _sha256(REPOSITORY_ROOT / "uv.lock"),
        "target_trials_per_study": design["target_trials_per_study"],
        "search_epochs": design["search_epochs"],
        "retrain_epochs": design["retrain_epochs"],
        "retrain_seed": 123,
        "target_rate_khz": TARGET_RATE_KHZ,
        "base_rate_khz": BASE_RATE_KHZ,
        "study_database_sha256": databases,
    }
    manifest = root / "selection" / "pareto_retrain_manifest.json"
    oracle = root / "results" / "oracle_winners.json"
    value["pareto_manifest_sha256"] = _sha256(manifest) if manifest.is_file() else None
    value["oracle_winners_sha256"] = _sha256(oracle) if oracle.is_file() else None
    return value


def snapshot(root: Path, output: Path) -> dict[str, Any]:
    """Export search-only state without traversing downstream result artifacts."""
    design = _design(root)
    output.mkdir(parents=True, exist_ok=True)
    status_rows, trial_rows = _trial_exports(root, design)
    status_fields = list(status_rows[0])
    trial_fields = (
        list(trial_rows[0])
        if trial_rows
        else [
            "model",
            "study_name",
            "trial_number",
            "state",
            "usable",
            "selected_for_freeze",
            "datetime_start",
            "datetime_complete",
            "duration_seconds",
            "params_json",
            "values_json",
            "user_attrs_json",
            "system_attrs_json",
        ]
    )
    _write_csv(output / "study_status.csv", status_rows, status_fields)
    _write_csv(output / "trials.csv", trial_rows, trial_fields)
    bindings = _binding_index(root)
    _validate_bindings(root, design, bindings)
    checkpoints, mlflow = _checkpoint_rows(root, design, include_retrains=False)
    checkpoint_fields = _checkpoint_fields()
    _write_csv(output / "checkpoint_selections.csv", checkpoints, checkpoint_fields)
    _write_csv(output / "mlflow_runs.csv", mlflow, _mlflow_fields())
    _write_json(output / "provenance.json", _provenance(root, design, final=False))
    latest = {
        "schema_version": 1,
        "code_commit": design["code_commit"],
        "studies": status_rows,
        "ready_to_freeze": all(row["usable_complete"] >= row["target"] for row in status_rows),
    }
    _write_json(root / "state" / "latest_status.json", latest)
    _write_experiment_matrix(root, design, status_rows)
    _copy_state_ledgers(root, output)
    return latest


def _checkpoint_fields() -> list[str]:
    """Return the stable checkpoint-selection CSV columns."""
    return [
        "phase",
        "model",
        "trial_number",
        "retrain_index",
        "logical_objective",
        "score",
        "metric",
        "strategy",
        "optimized_ckpt_name",
        "q_prime",
        "q_double_prime",
        "q_prime_direction",
        "q_double_prime_direction",
        "artifact",
        "artifact_sha256",
        "hydra_run_dir",
        "mlflow_run_id",
    ]


def _mlflow_fields() -> list[str]:
    """Return the stable MLflow run-index CSV columns."""
    return [
        "phase",
        "model",
        "trial_number",
        "retrain_index",
        "experiment_name",
        "experiment_id",
        "run_id",
        "run_name",
        "hydra_run_dir",
        "checkpoint_run_dir",
        "artifact",
    ]


def _write_experiment_matrix(
    root: Path,
    design: dict[str, Any],
    status_rows: list[dict[str, Any]],
    *,
    manifest: dict[str, Any] | None = None,
    oracle: dict[str, Any] | None = None,
) -> None:
    """Synchronize the 56-front phase-completeness matrix."""
    search_complete = {
        row["model"]: row["usable_complete"] >= row["target"] for row in status_rows
    }
    front_ids = (
        set()
        if manifest is None
        else {(row["model"], row["logical_objective"]) for row in manifest["front_memberships"]}
    )
    complete_retrains = (
        set()
        if manifest is None
        else {
            int(row["retrain_index"])
            for row in manifest["retrain_runs"]
            if (
                root / "retraining" / "markers" / f"{int(row['retrain_index']):04d}.json"
            ).is_file()
        }
    )
    winner_ids = (
        set()
        if oracle is None
        else {(row["model"], row["logical_objective"]) for row in oracle["winners"]}
    )
    rows = []
    for study in design["studies"]:
        for objective in study["objectives"]:
            key = (study["model"], objective["id"])
            retrain_indices = (
                []
                if manifest is None
                else [
                    int(row["retrain_index"])
                    for row in manifest["front_memberships"]
                    if (row["model"], row["logical_objective"]) == key
                ]
            )
            rows.append(
                {
                    "model": study["model"],
                    "logical_objective": objective["id"],
                    "score": objective["score"],
                    "metric": objective["metric"],
                    "strategy": objective["strategy"] or "",
                    "q_prime_direction": objective["directions"][0],
                    "q_double_prime_direction": objective["directions"][1],
                    "search_status": "complete" if search_complete[study["model"]] else "pending",
                    "front_status": "frozen" if key in front_ids else "pending",
                    "retrain_status": (
                        "complete"
                        if retrain_indices
                        and all(index in complete_retrains for index in retrain_indices)
                        else "pending"
                    ),
                    "report_status": "complete" if key in winner_ids else "pending",
                }
            )
    _write_csv(root / "state" / "experiment_matrix.csv", rows, list(rows[0]))


def _marker_rows(root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Authenticate every frozen retraining row and completion marker."""
    rows = []
    expected_indices = list(range(len(manifest["retrain_runs"])))
    observed_indices = [int(row["retrain_index"]) for row in manifest["retrain_runs"]]
    if observed_indices != expected_indices:
        raise RuntimeError("Retrain indices are not contiguous and ordered.")
    for expected in manifest["retrain_runs"]:
        index = int(expected["retrain_index"])
        marker_path = root / "retraining" / "markers" / f"{index:04d}.json"
        marker = _read_json(marker_path)
        if marker.get("schema_version") != 2 or marker.get("row") != expected:
            raise RuntimeError(f"Retraining marker does not match frozen row: {marker_path}")
        optimized = Path(marker["optimized_metric_artifact"])
        if _sha256(optimized) != marker["optimized_metric_artifact_sha256"]:
            raise RuntimeError(f"Changed optimized artifact: {optimized}")
        for entry in marker["value_files"]:
            if _sha256(Path(entry["path"])) != entry["sha256"]:
                raise RuntimeError(f"Changed downstream artifact: {entry['path']}")
        rows.append(
            {
                "retrain_index": index,
                "model": expected["model"],
                "trial_number": expected["trial_number"],
                "seed": expected["seed"],
                "epochs": expected["epochs"],
                "status": "complete",
                "slurm_job_id": marker.get("slurm_job_id") or "",
                "slurm_array_job_id": marker.get("slurm_array_job_id") or "",
                "slurm_array_task_id": marker.get("slurm_array_task_id") or "",
                "started_at": marker.get("started_at", ""),
                "completed_at": marker["completed_at"],
                "elapsed_seconds": marker.get("elapsed_seconds", ""),
                "marker": _relative(marker_path, root),
                "marker_sha256": _sha256(marker_path),
                "optimized_artifact": _relative(optimized, root),
                "optimized_artifact_sha256": marker["optimized_metric_artifact_sha256"],
                "checkpoint_root": marker["checkpoint_root"],
                "logical_objectives_json": _json_cell(expected["logical_objectives"]),
            }
        )
    return rows


def _winner_exports(
    root: Path, design: dict[str, Any], manifest: dict[str, Any], oracle: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate and export 56 oracle winners and their downstream efficiencies."""
    expected_ids = {
        (study["model"], objective["id"])
        for study in design["studies"]
        for objective in study["objectives"]
    }
    if len(expected_ids) != EXPECTED_FRONTS:
        raise RuntimeError("Design does not contain exactly 56 logical fronts.")
    winners = oracle.get("winners", [])
    winner_ids = [(row["model"], row["logical_objective"]) for row in winners]
    if len(winners) != EXPECTED_FRONTS or set(winner_ids) != expected_ids:
        raise RuntimeError("Oracle result does not contain one winner for every front.")
    if len(winner_ids) != len(set(winner_ids)):
        raise RuntimeError("Oracle result contains duplicate logical winners.")
    memberships = manifest["front_memberships"]
    membership_keys = {
        (row["model"], row["logical_objective"], int(row["trial_number"])) for row in memberships
    }
    front_sizes = Counter((row["model"], row["logical_objective"]) for row in memberships)
    winner_rows = []
    downstream_rows = []
    primary_rows = []
    objective_lookup = {
        (study["model"], objective["id"]): objective
        for study in design["studies"]
        for objective in study["objectives"]
    }
    for winner in sorted(
        winners,
        key=lambda row: (MODEL_ORDER.index(row["model"]), row["logical_objective"]),
    ):
        key = (winner["model"], winner["logical_objective"], int(winner["trial_number"]))
        if key not in membership_keys:
            raise RuntimeError(f"Winner is not a frozen front member: {key}")
        efficiencies = winner["downstream_efficiencies"]
        if set(efficiencies) != set(DOWNSTREAM_DATASETS):
            raise RuntimeError(f"Winner {key} does not have the exact downstream dataset set.")
        values = [float(value) for _, value in sorted(efficiencies.items())]
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
            raise RuntimeError(f"Winner {key} contains invalid efficiency.")
        lowest = sorted(values)[: max(1, len(values) // 4)]
        objective = objective_lookup[(winner["model"], winner["logical_objective"])]
        pair = [float(value) for value in winner["retrained_objective_values"]]
        marker = _read_json(
            root / "retraining" / "markers" / f"{int(winner['retrain_index']):04d}.json"
        )
        optimized_path = Path(marker["optimized_metric_artifact"])
        winner_rows.append(
            {
                "model": winner["model"],
                "logical_objective": winner["logical_objective"],
                "score": winner["score"],
                "metric": winner["metric"],
                "strategy": winner.get("strategy") or "",
                "trial_number": winner["trial_number"],
                "retrain_index": winner["retrain_index"],
                "optimized_ckpt_name": winner["optimized_ckpt_name"],
                "q_prime": pair[0],
                "q_double_prime": pair[1],
                "mean_efficiency": statistics.fmean(values),
                "median_efficiency": statistics.median(values),
                "worst_efficiency": min(values),
                "bottom_quartile_cvar": statistics.fmean(lowest),
                "front_size": front_sizes[(winner["model"], winner["logical_objective"])],
                "params_json": _json_cell(winner["params"]),
            }
        )
        for anomaly, value in sorted(efficiencies.items()):
            downstream_rows.append(
                {
                    "model": winner["model"],
                    "logical_objective": winner["logical_objective"],
                    "score": winner["score"],
                    "metric": winner["metric"],
                    "strategy": winner.get("strategy") or "",
                    "trial_number": winner["trial_number"],
                    "retrain_index": winner["retrain_index"],
                    "anomaly": anomaly,
                    "efficiency_operational": float(value),
                    "is_winner": True,
                }
            )
        for metric_name, value, direction, role in (
            (objective["metric"], pair[0], objective["directions"][0], "primary"),
            ("native_secondary", pair[1], objective["directions"][1], "secondary"),
        ):
            primary_rows.append(
                {
                    "model": winner["model"],
                    "logical_objective": winner["logical_objective"],
                    "score": winner["score"],
                    "strategy": winner.get("strategy") or "",
                    "trial_number": winner["trial_number"],
                    "retrain_index": winner["retrain_index"],
                    "checkpoint": winner["optimized_ckpt_name"],
                    "metric_name": metric_name,
                    "value": value,
                    "direction": direction,
                    "split": "validate",
                    "role": role,
                    "source_artifact": _relative(optimized_path, root),
                    "source_artifact_sha256": marker["optimized_metric_artifact_sha256"],
                }
            )
    return winner_rows, downstream_rows, primary_rows


def _front_rows(design: dict[str, Any], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten all nonempty mixed-direction front memberships."""
    lookup = {
        (study["model"], objective["id"]): objective
        for study in design["studies"]
        for objective in study["objectives"]
    }
    rows = []
    for membership in manifest["front_memberships"]:
        objective = lookup[(membership["model"], membership["logical_objective"])]
        rows.append(
            {
                **{key: value for key, value in membership.items() if key != "objective_values"},
                "q_prime": membership["objective_values"][0],
                "q_double_prime": membership["objective_values"][1],
                "q_prime_direction": objective["directions"][0],
                "q_double_prime_direction": objective["directions"][1],
            }
        )
    if {row["logical_objective"] for row in rows} != {
        objective["id"] for study in design["studies"] for objective in study["objectives"]
    }:
        raise RuntimeError("One or more logical fronts are empty.")
    return sorted(rows, key=lambda row: int(row["membership_index"]))


def _dominates(left: list[float], right: list[float], directions: list[str]) -> bool:
    """Return mixed-direction Pareto dominance for one objective pair."""
    no_worse = []
    strictly_better = []
    for left_value, right_value, direction in zip(left, right, directions):
        if direction == "maximize":
            no_worse.append(left_value >= right_value)
            strictly_better.append(left_value > right_value)
        else:
            no_worse.append(left_value <= right_value)
            strictly_better.append(left_value < right_value)
    return all(no_worse) and any(strictly_better)


def _validate_manifest_against_studies(
    root: Path, design: dict[str, Any], manifest: dict[str, Any]
) -> None:
    """Reconstruct all fronts from the first usable trials and compare exactly."""
    expected_members: dict[tuple[str, str, int], list[float]] = {}
    expected_retrains: dict[tuple[str, int], dict[str, Any]] = {}
    target = int(design["target_trials_per_study"])
    for study in design["studies"]:
        loaded = _load_study(root, study)
        if loaded is None:
            raise RuntimeError(f"Missing study for frozen manifest: {study['model']}")
        expected_directions = [
            direction for objective in study["objectives"] for direction in objective["directions"]
        ]
        if [direction.name.lower() for direction in loaded.directions] != expected_directions:
            raise RuntimeError(f"Optuna direction mismatch for {study['model']}.")
        trials = [
            trial
            for trial in sorted(loaded.get_trials(False), key=lambda row: int(row.number))
            if _usable(trial, len(expected_directions))
        ][:target]
        if len(trials) != target:
            raise RuntimeError(f"Insufficient usable trials for {study['model']}.")
        for objective in study["objectives"]:
            indices = objective["value_indices"]
            directions = objective["directions"]
            front = []
            for trial in trials:
                values = [float(trial.values[index]) for index in indices]
                if not any(
                    other.number != trial.number
                    and _dominates(
                        [float(other.values[index]) for index in indices],
                        values,
                        directions,
                    )
                    for other in trials
                ):
                    front.append(trial)
            for trial in front:
                key = (study["model"], objective["id"], int(trial.number))
                expected_members[key] = [float(trial.values[index]) for index in indices]
                retrain_key = (study["model"], int(trial.number))
                row = expected_retrains.setdefault(
                    retrain_key,
                    {
                        "model": study["model"],
                        "trial_number": int(trial.number),
                        "params": trial.params,
                        "logical_objectives": [],
                        "seed": 123,
                        "epochs": int(design["retrain_epochs"]),
                    },
                )
                row["logical_objectives"].append(objective["id"])
    actual_members = {}
    objective_lookup = {
        (study["model"], objective["id"]): objective
        for study in design["studies"]
        for objective in study["objectives"]
    }
    for membership in manifest["front_memberships"]:
        key = (
            membership["model"],
            membership["logical_objective"],
            int(membership["trial_number"]),
        )
        if key in actual_members:
            raise RuntimeError(f"Duplicate frozen membership: {key}")
        objective = objective_lookup[(membership["model"], membership["logical_objective"])]
        if (
            membership["score"] != objective["score"]
            or membership["metric"] != objective["metric"]
            or membership.get("strategy") != objective["strategy"]
        ):
            raise RuntimeError(f"Frozen membership metadata differs for {key}.")
        actual_members[key] = [float(value) for value in membership["objective_values"]]
    if actual_members != expected_members:
        raise RuntimeError("Frozen Pareto memberships differ from reconstructed Optuna fronts.")
    expected_rows = []
    for row in sorted(
        expected_retrains.values(),
        key=lambda item: (MODEL_ORDER.index(item["model"]), item["trial_number"]),
    ):
        expected_rows.append(
            {
                **row,
                "logical_objectives": sorted(set(row["logical_objectives"])),
                "retrain_index": len(expected_rows),
            }
        )
    if manifest["retrain_runs"] != expected_rows:
        raise RuntimeError("Frozen retraining rows differ from reconstructed front union.")
    index_by_trial = {
        (row["model"], int(row["trial_number"])): int(row["retrain_index"])
        for row in expected_rows
    }
    if any(
        int(membership["retrain_index"])
        != index_by_trial[(membership["model"], int(membership["trial_number"]))]
        for membership in manifest["front_memberships"]
    ):
        raise RuntimeError("Frozen membership-to-retrain indices differ from the front union.")


def _read_pickle_objects(path: Path, trusted_roots: Iterable[Path]) -> list[Any]:
    """Load all objects from one authenticated trusted-local pickle stream."""
    path = _require_inside(path, trusted_roots)
    values = []
    with path.open("rb") as handle:
        while True:
            try:
                values.append(pickle.load(handle))  # nosec B301 -- authenticated local path
            except EOFError:
                break
    return values


def _summary_value(
    objects: list[Any], index: int, checkpoint: str, *, nested: bool
) -> float | None:
    """Read one normalized checkpoint scalar from a trusted summary payload."""
    if len(objects) <= index or not isinstance(objects[index], dict):
        return None
    values = objects[index].get("operational", {}) if nested else objects[index]
    if not isinstance(values, dict):
        return None
    matches = []
    for name, value in values.items():
        normalized = str(name).split("ds=")[-1].split("__")[0] if "ds=" in str(name) else str(name)
        if normalized == checkpoint:
            matches.append(float(value))
    if not matches:
        return None
    if len(set(matches)) != 1:
        raise RuntimeError(f"Conflicting summary values for checkpoint {checkpoint}: {matches}")
    value = matches[0]
    if not math.isfinite(value):
        raise RuntimeError(f"Nonfinite summary value for checkpoint {checkpoint}.")
    return value


def _append_test_metrics(
    root: Path,
    manifest: dict[str, Any],
    oracle: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    """Append trusted held-out proxy, rank-correlation, and reverse-drift diagnostics."""
    marker_by_index = {
        int(row["retrain_index"]): _read_json(
            root / "retraining" / "markers" / f"{int(row['retrain_index']):04d}.json"
        )
        for row in manifest["retrain_runs"]
    }
    for winner in oracle["winners"]:
        marker = marker_by_index[int(winner["retrain_index"])]
        checkpoint_root = Path(marker["checkpoint_root"])
        if winner["metric"] == "cap":
            callback = f"cap_{winner['score']}_{winner['strategy']}_summary"
            expected_objects = 2
            metrics = (("cap_test", 0, False), ("rankcorr_test", 1, False))
        elif winner["metric"] == "wasserstein":
            callback = f"wasserstein_{winner['score']}_summary"
            expected_objects = 1
            metrics = (("wasserstein_test", 0, False),)
        elif winner["metric"] == "drift":
            callback = f"drift_{winner['score']}_summary"
            expected_objects = 2
            metrics = (("drift_test", 0, True), ("reverse_drift_posthoc", 1, True))
        else:
            raise RuntimeError(f"Unknown logical metric: {winner['metric']}")
        candidates = sorted(checkpoint_root.glob(f"**/plots/test/{callback}/summary.pkl"))
        extracted: dict[str, list[tuple[Path, float]]] = {
            metric_name: [] for metric_name, _, _ in metrics
        }
        for path in candidates:
            objects = _read_pickle_objects(path, [checkpoint_root])
            if len(objects) != expected_objects:
                raise RuntimeError(f"Unexpected pickle payload count in {path}.")
            for metric_name, object_index, nested in metrics:
                value = _summary_value(
                    objects,
                    object_index,
                    winner["optimized_ckpt_name"],
                    nested=nested,
                )
                if value is not None:
                    extracted[metric_name].append((path, value))
        for metric_name, matches in extracted.items():
            unique_values = {value for _, value in matches}
            if len(unique_values) != 1:
                raise RuntimeError(
                    f"Expected one {metric_name} value for {winner['model']} "
                    f"{winner['logical_objective']}, found {matches}."
                )
            path, value = matches[0]
            is_reverse = metric_name == "reverse_drift_posthoc"
            rows.append(
                {
                    "model": winner["model"],
                    "logical_objective": winner["logical_objective"],
                    "score": winner["score"],
                    "strategy": winner.get("strategy") or "",
                    "trial_number": winner["trial_number"],
                    "retrain_index": winner["retrain_index"],
                    "checkpoint": winner["optimized_ckpt_name"],
                    "metric_name": metric_name,
                    "value": value,
                    "direction": "diagnostic" if is_reverse else "report",
                    "split": "test",
                    "role": "posthoc_diagnostic" if is_reverse else "heldout_proxy",
                    "source_artifact": _relative(path, root),
                    "source_artifact_sha256": _sha256(path),
                }
            )


def _write_report(output: Path, design: dict[str, Any], winners: list[dict[str, Any]]) -> None:
    """Write the self-contained human-readable report from portable rows."""
    by_metric: dict[str, list[float]] = {}
    by_model: dict[str, list[float]] = {}
    by_score: dict[str, list[float]] = {}
    for row in winners:
        by_metric.setdefault(row["metric"], []).append(float(row["mean_efficiency"]))
        by_model.setdefault(row["model"], []).append(float(row["mean_efficiency"]))
        by_score.setdefault(row["score"], []).append(float(row["mean_efficiency"]))
    lines = [
        "# Physics background-pairing experiment report",
        "",
        "## Executive summary",
        "",
        "The six shared Optuna studies completed the background-era model-selection campaign. "
        "The machine-readable result is `winners.csv` (56 logical fronts); detailed held-out "
        "efficiencies are in `downstream_efficiencies.csv`.",
        "",
        "## Provenance and reproducibility",
        "",
        f"- Code commit: `{design['code_commit']}`",
        f"- Search: {design['target_trials_per_study']} usable trials per model, "
        f"{design['search_epochs']} epochs each",
        f"- Retraining: {design['retrain_epochs']} epochs, seed 123",
        "- Validation sources: ZB_run396102 (2025E) and ZB_run398183 (2025G)",
        "- Fronts are signal-agnostic; each reported member is a downstream-label oracle winner.",
        "",
        "All source hashes and absolute authoritative scratch locations are recorded in "
        "`provenance.json`; downloadable files are authenticated by `manifest.sha256`.",
        "",
        "## Data",
        "",
        "Training and ordinary normal validation/test data follow the paper configuration. Model "
        "selection uses held-out ZB_run396102 and ZB_run398183 events only, with 163,840 events "
        "per source and split. CAP uses flat-physical, physics-summary, typed-sliced-Wasserstein, "
        "JetCLR, and empirical-CDF pairings. The 20 downstream anomalies enter only after the "
        "front manifest is frozen.",
        "",
        "## Models and anomaly scores",
        "",
        "The models are AE, VAE, DSAE, DSVAE, DeepSVDD, and RealNVP. AE uses masked "
        "reconstruction MSE and VAE uses raw latent KL as native controls; both additionally use "
        "masked residual OAS/Mahalanobis. OAS excludes structural padding through its active "
        "observed-feature subspace. The other four models use only their native paper score.",
        "",
        "## Protocol",
        "",
        "Each model has one shared multi-objective study. Every 50-epoch trial evaluates all of "
        "its CAP, W1, and forward E→G drift views. Selection compares last, primary-EMA, and "
        "stable-native checkpoints with the fixed `1.25Q' + Q''` relative-change rule. The 56 "
        "two-dimensional fronts were frozen before downstream access; unique members were "
        "retrained for 200 epochs with seed 123.",
        "",
        "## Execution and completeness",
        "",
        "`study_status.csv` proves the six study budgets; `trials.csv` identifies the first 600 "
        "usable trials per model; `front_memberships.csv` and `retrains.csv` prove front and "
        "retraining completeness. `job_history.csv` and `failures.jsonl` record operational "
        "history without silently dropping failures.",
        "",
        "## Main results",
        "",
        "| Metric | Winner rows | Mean of winner mean efficiencies |",
        "|---|---:|---:|",
    ]
    for metric, values in sorted(by_metric.items()):
        lines.append(f"| {metric} | {len(values)} | {statistics.fmean(values):.6g} |")
    lines.extend(
        [
            "",
            "The complete 56-row result is `winners.csv`; all 1,120 winner/anomaly values are in "
            "`downstream_efficiencies.csv`.",
            "",
            "## Pairing comparison",
            "",
            "The four mapping CAP strategies and CDF control can be compared within each "
            "model/score using `winners.csv`. Raw CAP magnitudes across different mappings are not "
            "treated as interchangeable evidence; downstream mean, median, worst, and "
            "bottom-quartile CVaR columns are the comparison endpoints.",
            "",
            "## Model-selection metric comparison",
            "",
            "CAP variants, W1, and forward drift winner summaries are shown above and are fully "
            "enumerated in `primary_metrics.csv`. W1 is pairing-invariant because it uses only "
            "the two score marginals.",
            "",
            "## AE/VAE native versus OAS",
            "",
            f"The score groups contain {len(by_score.get('native', []))} native and "
            f"{len(by_score.get('residual_oas', []))} residual-OAS winner rows. Exact paired "
            "per-anomaly comparisons are reproducible from `downstream_efficiencies.csv`.",
            "",
            "## Model comparison and robustness",
            "",
            "| Model | Winner rows | Mean of winner mean efficiencies |",
            "|---|---:|---:|",
        ]
    )
    for model in MODEL_ORDER:
        values = by_model[model]
        lines.append(f"| {model} | {len(values)} | {statistics.fmean(values):.6g} |")
    lines.extend(
        [
            "",
            "Robustness columns in `winners.csv` summarize variation across anomaly datasets, not "
            "training-seed uncertainty; retraining uses one fixed seed.",
            "",
            "## Checkpoint and proxy diagnostics",
            "",
            "`checkpoint_selections.csv` records every selected candidate and Q′/Q″ pair. "
            "`primary_metrics.csv` adds held-out CAP, rank correlation, W1, forward drift, and "
            "post-hoc reverse G→E drift for the reported checkpoints.",
            "",
            "## Limitations and interpretation guardrails",
            "",
            "The six-study sampler couples strategies through a high-dimensional M-TPE history. "
            "The final winner within each frozen front uses the 20 anomaly labels and is therefore "
            "an oracle reporting choice, not deployable signal-free model selection. W1 uses only "
            "marginals and cannot measure pairing quality. Reverse G→E drift is post-hoc only.",
            "",
            "The experiment uses one fixed retraining seed and one recorded E/G era pair, so it "
            "does not estimate training-seed uncertainty or establish universality across eras. "
            "OAS conclusions are conditional on the masked active-subspace covariance estimate.",
            "",
            "## Conclusion",
            "",
            "The evidence needed to determine how pairing choice and residual OAS change useful "
            "model selection is contained in the named portable tables. Conclusions should use "
            "the within-model/score comparisons and oracle caveat above rather than raw CAP "
            "magnitude alone.",
            "",
            "## Artifact index",
            "",
            "See `manifest.sha256`, `provenance.json`, `study_status.csv`, `trials.csv`, "
            "`front_memberships.csv`, `retrains.csv`, `checkpoint_selections.csv`, "
            "`primary_metrics.csv`, `downstream_efficiencies.csv`, `winners.csv`, and "
            "`mlflow_runs.csv`.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _write_manifest(output: Path) -> None:
    """Hash every portable file except the hash manifest itself."""
    rows = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = path.relative_to(output).as_posix()
        if relative == "manifest.sha256":
            continue
        rows.append(f"{_sha256(path)}  {relative}")
    temporary = output / f"manifest.sha256.{os.getpid()}.tmp"
    temporary.write_text("\n".join(rows) + "\n", encoding="utf-8")
    temporary.replace(output / "manifest.sha256")


def finalize(root: Path, output: Path) -> None:
    """Validate the frozen campaign and emit the complete portable package."""
    design = _design(root)
    snapshot(root, output)
    status_rows, _ = _trial_exports(root, design)
    target = int(design["target_trials_per_study"])
    if any(row["usable_complete"] < target for row in status_rows):
        raise RuntimeError("Cannot finalize before all six studies reach their usable target.")
    bindings = _binding_index(root)
    selected = {
        (row["model"], int(row["trial_number"]))
        for row in csv.DictReader((output / "trials.csv").open(encoding="utf-8"))
        if row["selected_for_freeze"] == "True"
    }
    incomplete_bindings = [
        key
        for key in sorted(selected)
        if key not in bindings or not bindings[key].get("optimized_metric_artifact")
    ]
    if incomplete_bindings:
        raise RuntimeError(
            "Selected Optuna trials lack complete hashed Hydra artifacts: "
            f"{incomplete_bindings[:10]}"
        )
    manifest_path = root / "selection" / "pareto_retrain_manifest.json"
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema_version") != 2
        or manifest.get("code_commit") != design["code_commit"]
        or manifest.get("selection_uses_downstream_anomalies") is not False
    ):
        raise RuntimeError("Frozen Pareto manifest violates the campaign contract.")
    _validate_manifest_against_studies(root, design, manifest)
    oracle_path = root / "results" / "oracle_winners.json"
    oracle = _read_json(oracle_path)
    if (
        oracle.get("schema_version") != 2
        or oracle.get("oracle_uses_downstream_anomalies") is not True
    ):
        raise RuntimeError("Oracle winner artifact violates the campaign contract.")
    front_rows = _front_rows(design, manifest)
    retrain_rows = _marker_rows(root, manifest)
    winner_rows, downstream_rows, primary_rows = _winner_exports(root, design, manifest, oracle)
    _append_test_metrics(root, manifest, oracle, primary_rows)
    checkpoints, mlflow = _checkpoint_rows(root, design, include_retrains=True)
    _write_csv(output / "front_memberships.csv", front_rows, list(front_rows[0]))
    _write_csv(output / "retrains.csv", retrain_rows, list(retrain_rows[0]))
    _write_csv(output / "checkpoint_selections.csv", checkpoints, _checkpoint_fields())
    _write_csv(output / "mlflow_runs.csv", mlflow, _mlflow_fields())
    _write_csv(output / "winners.csv", winner_rows, list(winner_rows[0]))
    _write_csv(output / "downstream_efficiencies.csv", downstream_rows, list(downstream_rows[0]))
    _write_csv(output / "primary_metrics.csv", primary_rows, list(primary_rows[0]))
    _write_json(output / "provenance.json", _provenance(root, design, final=True))
    _write_experiment_matrix(
        root,
        design,
        status_rows,
        manifest=manifest,
        oracle=oracle,
    )
    for source, name in (
        (REPOSITORY_ROOT / "research/physics_background_pairing/PLAN.md", "PLAN.md"),
        (Path(design["preflight"]), "preflight.json"),
        (root / "design.json", "design.json"),
        (manifest_path, "pareto_retrain_manifest.json"),
        (oracle_path, "oracle_winners.json"),
    ):
        destination = output / name
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
    _write_report(output, design, winner_rows)
    (output / "README.md").write_text(
        "# Portable physics background-pairing results\n\n"
        "Start with `REPORT.md`. Verify downloads with `sha256sum -c manifest.sha256`. "
        "The manifest intentionally excludes itself. Large databases, checkpoints, raw data, "
        "and the full MLflow store remain in the authoritative scratch campaign recorded in "
        "`provenance.json`.\n",
        encoding="utf-8",
    )
    _write_manifest(output)


def main() -> None:
    """Dispatch a search-only snapshot or fail-closed final export."""
    args = parse_args()
    root = args.root.expanduser().resolve()
    output = (args.output or root / "results").expanduser().resolve()
    if args.command == "snapshot":
        print(json.dumps(snapshot(root, output), indent=2, sort_keys=True))
    else:
        finalize(root, output)
        print(output / "REPORT.md")


if __name__ == "__main__":
    main()
