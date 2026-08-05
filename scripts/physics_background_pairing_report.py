#!/usr/bin/env python3
"""Export and validate the physics background-pairing campaign."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import pickle  # nosec B403 -- restricted to authenticated campaign checkpoint roots
import shutil
import statistics
import subprocess  # nosec B404 -- fixed read-only git commands
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlparse

from src.evaluation.physics_background_pairing_analysis import (
    build_physics_background_pairing_analysis,
)
from src.evaluation.physics_background_pairing_artifact_audit import (
    audit_physics_background_pairing_artifacts,
)

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
EXACT_RETRAIN_RNG_STATE_KEY = "physics-background-pairing-exact-retrain-rng-v1"
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
    parser.add_argument(
        "--execution-repository",
        type=Path,
        help="Override the immutable execution worktree resolved from the campaign convention.",
    )
    parser.add_argument(
        "--repository-report",
        type=Path,
        help="Optionally copy the published REPORT.md outside the portable package.",
    )
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


def _write_table_group(directory: Path, tables: dict[str, list[dict[str, Any]]]) -> None:
    """Write nonempty, schema-consistent deterministic row tables."""
    for name in sorted(tables):
        rows = tables[name]
        if not rows:
            raise RuntimeError(f"Portable table {name!r} is unexpectedly empty.")
        fields = list(rows[0])
        if any(set(row) != set(fields) for row in rows):
            raise RuntimeError(f"Portable table {name!r} has inconsistent row schemas.")
        encoded = [
            {
                key: _json_cell(value) if isinstance(value, (dict, list, tuple)) else value
                for key, value in row.items()
            }
            for row in rows
        ]
        _write_csv(directory / f"{name}.csv", encoded, fields)


def _json_cell(value: Any) -> str:
    """Encode a nested value for one deterministic CSV cell."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_sha256(value: Any) -> str:
    """Hash one JSON-compatible value using the campaign's canonical encoding."""
    return hashlib.sha256(_json_cell(value).encode("utf-8")).hexdigest()


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


def _capture_quiescent_studies(root: Path, design: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Freeze read-only study evidence and reject an incomplete or live final state."""
    target = int(design["target_trials_per_study"])
    captured: dict[str, dict[str, Any]] = {}
    for study in design["studies"]:
        model = str(study["model"])
        database = root / "studies" / f"{model}.db"
        if not database.is_file():
            raise FileNotFoundError(database)
        loaded = _load_study(root, study)
        if loaded is None:  # pragma: no cover - guarded by the file check
            raise RuntimeError(f"Missing Optuna study for {model}.")
        trials = list(loaded.get_trials(deepcopy=False))
        states = Counter(trial.state.name for trial in trials)
        expected_values = 2 * len(study["objectives"])
        usable = sum(_usable(trial, expected_values) for trial in trials)
        if states["RUNNING"] or states["WAITING"]:
            raise RuntimeError(
                f"Study {model} is not quiescent: RUNNING={states['RUNNING']}, "
                f"WAITING={states['WAITING']}."
            )
        if usable != target:
            raise RuntimeError(
                f"Study {model} has {usable} usable trials; finalization requires exactly {target}."
            )
        stat = database.stat()
        captured[model] = {
            "path": str(database.resolve()),
            "sha256": _sha256(database),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "usable": usable,
            "total": len(trials),
            "states": dict(sorted(states.items())),
        }
    return captured


def _assert_studies_unchanged(captured: dict[str, dict[str, Any]]) -> None:
    """Prove that no study database changed while the portable package was built."""
    for model, evidence in captured.items():
        database = Path(evidence["path"])
        if not database.is_file():
            raise RuntimeError(f"Study database disappeared during reporting: {model}.")
        stat = database.stat()
        observed = (_sha256(database), stat.st_size, stat.st_mtime_ns)
        expected = (evidence["sha256"], evidence["size"], evidence["mtime_ns"])
        if observed != expected:
            raise RuntimeError(f"Study database changed during reporting: {model}.")


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
        indexed_loggers = [
            logger for logger in provenance.get("loggers", []) if logger.get("run_id")
        ]
        if len(indexed_loggers) != 1:
            raise RuntimeError(f"Expected exactly one indexed MLflow run in {path}.")
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
        for logger in indexed_loggers:
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


def _primary_checkpoint_candidate(objective: dict[str, Any]) -> str:
    """Return the exact primary-EMA checkpoint-family path for one objective."""
    metric = objective["metric"]
    score = objective["score"]
    if metric == "cap":
        primary = f"cap_{score}_{objective['strategy']}_ema_" "ZB_run396102_vs_ZB_run398183"
    elif metric == "wasserstein":
        primary = f"w1dist_{score}_ema_ZB_run396102_vs_ZB_run398183"
    elif metric == "drift":
        primary = f"{score}_operational_drift_ema"
    else:
        raise RuntimeError(f"Unknown logical metric: {metric}")
    direction = "max" if objective["directions"][0] == "maximize" else "min"
    return f"summary/{primary}/{direction}"


def _checkpoint_family(checkpoint: str, objective: dict[str, Any]) -> str:
    """Classify one exact evaluator checkpoint ID for its logical objective."""
    allowed = {
        "last": "last",
        "single/ascore_operational/stable": "stable_native",
        _primary_checkpoint_candidate(objective): "primary_ema",
    }
    if checkpoint not in allowed:
        raise RuntimeError(
            f"Checkpoint {checkpoint!r} is not an exact candidate for {objective['id']}."
        )
    return allowed[checkpoint]


def _checkpoint_candidate_rows(
    root: Path,
    design: dict[str, Any],
    bindings: dict[tuple[str, int], dict[str, Any]],
    *,
    final: bool,
) -> list[dict[str, Any]]:
    """Audit every configured search checkpoint slot and record stable availability."""
    import torch

    studies = {study["model"]: study for study in design["studies"]}
    trusted_root = (root / "runtime" / "checkpoints").resolve(strict=True)
    rows = []
    checkpoint_roots: dict[Path, tuple[str, int]] = {}
    for (model, trial_number), binding in sorted(
        bindings.items(), key=lambda item: (MODEL_ORDER.index(item[0][0]), item[0][1])
    ):
        artifact_path = binding.get("optimized_metric_artifact")
        if not artifact_path:
            continue
        artifact = _read_json(Path(artifact_path))
        checkpoint_root = Path(artifact["run_provenance"]["checkpoint_run_dir"]).resolve(
            strict=True
        )
        if not checkpoint_root.is_relative_to(trusted_root):
            raise RuntimeError(
                f"Checkpoint candidate root escapes the campaign: {checkpoint_root}"
            )
        key = (model, trial_number)
        if checkpoint_root in checkpoint_roots:
            raise RuntimeError(
                f"Search trials {checkpoint_roots[checkpoint_root]} and {key} share checkpoint "
                f"root {checkpoint_root}."
            )
        checkpoint_roots[checkpoint_root] = key
        slots = [
            {
                "slot": "last",
                "checkpoint_family_id": "last",
                "logical_objective": "",
                "score": "",
                "metric": "",
                "strategy": "",
                "required": True,
                "paths": [checkpoint_root / "last.ckpt"],
            },
            {
                "slot": "stable_native",
                "checkpoint_family_id": "single/ascore_operational/stable",
                "logical_objective": "",
                "score": "native",
                "metric": "native_secondary",
                "strategy": "",
                "required": False,
                "paths": sorted(
                    (checkpoint_root / "single" / "ascore_operational" / "stable").glob("*.ckpt")
                ),
            },
        ]
        slots.extend(
            {
                "slot": "primary_ema",
                "checkpoint_family_id": _primary_checkpoint_candidate(objective),
                "logical_objective": objective["id"],
                "score": objective["score"],
                "metric": objective["metric"],
                "strategy": objective.get("strategy") or "",
                "required": True,
                "paths": sorted(
                    (checkpoint_root / _primary_checkpoint_candidate(objective)).glob("*.ckpt")
                ),
            }
            for objective in studies[model]["objectives"]
        )
        for slot in slots:
            paths = slot.pop("paths")
            available = len(paths) == 1 and paths[0].is_file()
            if len(paths) > 1 or (slot["required"] and not available):
                raise RuntimeError(
                    f"Invalid {slot['slot']} candidate count for {(model, trial_number)}: "
                    f"{len(paths)}."
                )
            path = paths[0] if available else None
            thresholds = {}
            if path is not None:
                path = path.resolve(strict=True)
                if not path.is_relative_to(checkpoint_root):
                    raise RuntimeError(f"Checkpoint candidate escapes its run root: {path}")
                payload = torch.load(  # nosec B614 -- authenticated campaign-local checkpoint
                    path, map_location="cpu", weights_only=False
                )
                state = payload.get("state_dict", {})
                expected_thresholds = {"thres_operational"}
                if model in {"ae", "vae"}:
                    expected_thresholds.add("thres_operational__residual_oas")
                observed_thresholds = {
                    str(name) for name in state if str(name).startswith("thres_operational")
                }
                if observed_thresholds != expected_thresholds:
                    raise ValueError(
                        f"Threshold buffer keys differ in {path}: expected "
                        f"{expected_thresholds}, found {observed_thresholds}."
                    )
                for name in sorted(expected_thresholds):
                    value = torch.as_tensor(state[name]).detach().cpu()
                    if value.numel() != 1 or not torch.isfinite(value).item():
                        raise ValueError(
                            f"Threshold buffer {name} is not a finite scalar in {path}."
                        )
                    thresholds[name] = float(value.item())
            rows.append(
                {
                    "model": model,
                    "trial_number": trial_number,
                    **slot,
                    "available": available,
                    "checkpoint": "" if path is None else _relative(path, root),
                    "checkpoint_sha256": "" if path is None else _sha256(path),
                    "thresholds_json": _json_cell(thresholds),
                }
            )
    if final:
        expected = int(design["target_trials_per_study"]) * sum(
            len(study["objectives"]) + 2 for study in design["studies"]
        )
        if len(rows) != expected:
            raise RuntimeError(
                f"Checkpoint-candidate export has {len(rows)} rows, expected {expected}."
            )
    return rows


def _checkpoint_candidate_fields() -> list[str]:
    """Return the stable retained-candidate audit columns."""
    return [
        "model",
        "trial_number",
        "slot",
        "checkpoint_family_id",
        "logical_objective",
        "score",
        "metric",
        "strategy",
        "required",
        "available",
        "checkpoint",
        "checkpoint_sha256",
        "thresholds_json",
    ]


def _checkpoint_availability_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize configured and available retained candidates by model and family."""
    output = []
    for model in MODEL_ORDER:
        for slot in ("last", "primary_ema", "stable_native"):
            selected = [row for row in rows if row["model"] == model and row["slot"] == slot]
            if not selected:
                continue
            output.append(
                {
                    "model": model,
                    "slot": slot,
                    "configured": len(selected),
                    "required": sum(bool(row["required"]) for row in selected),
                    "available": sum(bool(row["available"]) for row in selected),
                    "unavailable": sum(not bool(row["available"]) for row in selected),
                }
            )
    return output


def _annotate_checkpoint_families(
    design: dict[str, Any], rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Attach fail-closed family labels to authenticated selection rows."""
    objectives = {
        (study["model"], objective["id"]): objective
        for study in design["studies"]
        for objective in study["objectives"]
    }
    output = []
    for source in rows:
        row = dict(source)
        objective = objectives[(row["model"], row["logical_objective"])]
        row["checkpoint_family"] = _checkpoint_family(str(row["optimized_ckpt_name"]), objective)
        output.append(row)
    return output


def _selection_metric_keys(design: dict[str, Any], model: str) -> list[str]:
    """Return the exact per-epoch selection histories required for one model."""
    pairing = _read_json(Path(design["preflight"]))
    suffix = f"{pairing['dataset_1']}_vs_{pairing['dataset_2']}"
    keys = []
    for score in SCORES_BY_MODEL[model]:
        for strategy in CAP_STRATEGIES:
            keys.append(f"val/summary/cap_{score}_{strategy}_ema_{suffix}")
            keys.append(f"val/summary/rankcorr_cap_{score}_{strategy}_{suffix}")
        keys.append(f"val/summary/w1dist_{score}_ema_{suffix}")
        keys.append(f"val/summary/{score}_operational_drift_ema")
    keys.append("train/ascore_operational")
    return keys


def _validate_mlflow_runs(
    root: Path,
    design: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    final: bool,
) -> list[dict[str, Any]]:
    """Authenticate indexed runs and their complete file-backed epoch histories."""
    from mlflow.tracking import MlflowClient

    store = (root / "runtime" / "logs" / "mlflow" / "mlruns").resolve()
    if not store.is_dir():
        raise FileNotFoundError(f"Canonical MLflow FileStore is missing: {store}")
    client = MlflowClient(tracking_uri=store.as_uri())
    seen: set[tuple[str, str]] = set()
    validated = []
    for source in rows:
        row = dict(source)
        experiment_id = str(row["experiment_id"])
        run_id = str(row["run_id"])
        identity = (experiment_id, run_id)
        if identity in seen:
            raise RuntimeError(f"Duplicate indexed MLflow run: {identity}.")
        seen.add(identity)
        try:
            run = client.get_run(run_id)
            experiment = client.get_experiment(experiment_id)
        except Exception as error:
            raise RuntimeError(f"Missing indexed MLflow run: {identity}.") from error
        if experiment is None or experiment.name != row["experiment_name"]:
            raise RuntimeError(f"MLflow experiment differs for run {run_id}.")
        if str(run.info.experiment_id) != experiment_id:
            raise RuntimeError(f"MLflow experiment ID differs for run {run_id}.")
        if run.data.tags.get("mlflow.source.git.commit") != design["code_commit"]:
            raise RuntimeError(f"MLflow execution commit differs for run {run_id}.")
        if run.data.tags.get("mlflow.runName") != row["run_name"]:
            raise RuntimeError(f"MLflow run name differs for run {run_id}.")
        if run.info.lifecycle_stage != "active":
            raise RuntimeError(f"MLflow run is not active: {run_id}.")

        parsed = urlparse(run.info.artifact_uri)
        if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
            raise RuntimeError(f"MLflow artifact URI is not a local file URI: {run_id}.")
        artifact_path = Path(unquote(parsed.path)).resolve()
        run_store = (store / experiment_id / run_id).resolve()
        if not (run_store / "meta.yaml").is_file() or not artifact_path.is_relative_to(run_store):
            raise RuntimeError(f"MLflow artifact URI escapes its canonical run: {run_id}.")

        expected_epochs = int(
            design["search_epochs"] if row["phase"] == "search" else design["retrain_epochs"]
        )
        epoch_history = sorted(
            client.get_metric_history(run_id, "epoch_idx"), key=lambda x: x.step
        )
        observed_epochs = len(epoch_history)
        allowed_statuses = {"FINISHED"} if final else {"RUNNING", "FINISHED"}
        if run.info.status not in allowed_statuses:
            raise RuntimeError(f"MLflow run has invalid status {run.info.status}: {run_id}.")
        if run.info.status == "FINISHED" or final:
            if observed_epochs != expected_epochs:
                raise RuntimeError(
                    f"MLflow epoch history for {run_id} has {observed_epochs}, "
                    f"expected {expected_epochs}."
                )
        elif not 0 < observed_epochs <= expected_epochs:
            raise RuntimeError(f"MLflow running epoch history is invalid: {run_id}.")
        epoch_steps = [int(metric.step) for metric in epoch_history]
        epoch_values = [float(metric.value) for metric in epoch_history]
        if (
            len(set(epoch_steps)) != observed_epochs
            or epoch_steps != sorted(epoch_steps)
            or epoch_values != [float(index) for index in range(observed_epochs)]
        ):
            raise RuntimeError(f"MLflow epoch/global-step history is not contiguous: {run_id}.")

        required = _selection_metric_keys(design, str(row["model"]))
        for key in required:
            history = sorted(client.get_metric_history(run_id, key), key=lambda x: x.step)
            if (
                [int(metric.step) for metric in history] != epoch_steps
                or len(history) != observed_epochs
                or not all(math.isfinite(float(metric.value)) for metric in history)
            ):
                raise RuntimeError(f"Invalid MLflow selection history {key!r} for run {run_id}.")
        row.update(
            {
                "tracking_uri": store.as_uri(),
                "status": run.info.status,
                "lifecycle_stage": run.info.lifecycle_stage,
                "artifact_uri": run.info.artifact_uri,
                "expected_epochs": expected_epochs,
                "observed_epochs": observed_epochs,
                "required_history_count": len(required),
                "histories_validated": True,
            }
        )
        validated.append(row)
    return validated


def _validated_state_ledgers(
    root: Path, design: dict[str, Any], *, final: bool
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Load operator ledgers and authenticate every recorded final-job log."""
    source_jobs = root / "state" / "job_history.csv"
    source_failures = root / "state" / "failures.jsonl"
    if not final and not source_jobs.is_file():
        return [], []
    if not source_jobs.is_file():
        raise FileNotFoundError(source_jobs)
    with source_jobs.open(encoding="utf-8", newline="") as handle:
        jobs = list(csv.DictReader(handle))
    if final and not jobs:
        raise RuntimeError("Final job_history.csv must contain at least one data row.")
    required = {"job_id", "state", "commit", "log_path", "log_sha256"}
    if final and (not jobs or not required.issubset(jobs[0])):
        raise RuntimeError(f"job_history.csv lacks required columns: {sorted(required)}.")
    nonterminal = {"", "PENDING", "RUNNING", "WAITING", "CONFIGURING", "COMPLETING"}
    for index, job in enumerate(jobs):
        if final and str(job.get("state", "")).upper() in nonterminal:
            raise RuntimeError(f"Job ledger row {index} is not terminal.")
        if final and job.get("commit") != design["code_commit"]:
            raise RuntimeError(f"Job ledger execution commit differs at row {index}.")
        log_name = job.get("log_path", "")
        expected_hash = job.get("log_sha256", "")
        if not final and not expected_hash:
            continue
        if not log_name or not expected_hash:
            raise RuntimeError(f"Job ledger row {index} lacks a recorded Slurm log/hash.")
        log_path = Path(log_name)
        if not log_path.is_file():
            raise FileNotFoundError(f"Recorded Slurm log is missing: {log_path}")
        if _sha256(log_path) != expected_hash:
            raise RuntimeError(f"Recorded Slurm log hash differs: {log_path}")

    if final and not source_failures.is_file():
        raise FileNotFoundError(source_failures)
    failures = []
    if source_failures.is_file():
        for line_number, line in enumerate(
            source_failures.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Malformed failures.jsonl record at line {line_number}."
                ) from error
            if not isinstance(value, dict):
                raise RuntimeError(f"failures.jsonl line {line_number} is not an object.")
            failures.append(value)
    return jobs, failures


def _copy_state_ledgers(
    root: Path, output: Path, design: dict[str, Any], *, final: bool
) -> list[dict[str, str]]:
    """Copy validated deterministic operator ledgers into the portable package."""
    jobs, _ = _validated_state_ledgers(root, design, final=final)
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
    return jobs


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


def _execution_repository_evidence(
    root: Path,
    design: dict[str, Any],
    explicit: Path | None = None,
) -> dict[str, Any]:
    """Authenticate the immutable worktree that produced the execution campaign."""
    repository = (
        explicit.expanduser()
        if explicit is not None
        else root.parent.parent / "worktrees" / root.name
    ).resolve(strict=True)
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required to authenticate the execution worktree.")

    def git_output(*arguments: str) -> str:
        return subprocess.run(  # nosec B603
            [git, *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    top_level = Path(git_output("rev-parse", "--show-toplevel")).resolve(strict=True)
    commit = git_output("rev-parse", "HEAD")
    dirty = git_output("status", "--porcelain").splitlines()
    uv_lock = repository / "uv.lock"
    if top_level != repository:
        raise RuntimeError("Execution repository path is not its Git worktree root.")
    if commit != design["code_commit"]:
        raise RuntimeError("Execution worktree HEAD differs from design.code_commit.")
    if dirty:
        raise RuntimeError(f"Execution worktree is dirty: {dirty[:10]}")
    if not uv_lock.is_file():
        raise FileNotFoundError(uv_lock)
    return {
        "execution_repository_root": str(repository),
        "execution_repository_commit": commit,
        "execution_repository_clean": True,
        "execution_uv_lock_sha256": _sha256(uv_lock),
    }


def _require_clean_reporting_checkout() -> tuple[str, bool, list[str]]:
    """Require reporting to originate from one committed, reproducible checkout."""
    state = _git_state()
    if not state[0] or not state[1]:
        raise RuntimeError(
            "Final reporting requires a clean checkout; dirty entries: " f"{state[2][:10]}"
        )
    return state


def _provenance(
    root: Path,
    design: dict[str, Any],
    *,
    final: bool,
    report_state: tuple[str, bool, list[str]] | None = None,
    study_evidence: dict[str, dict[str, Any]] | None = None,
    execution_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build portable code, input, budget, and source-artifact provenance."""
    commit, clean, dirty = report_state or _git_state()
    databases = {}
    for model in MODEL_ORDER:
        path = root / "studies" / f"{model}.db"
        databases[model] = (
            study_evidence[model]["sha256"]
            if final and study_evidence is not None
            else (_sha256(path) if final and path.is_file() else None)
        )
    value = {
        "schema_version": 2,
        "campaign_root": str(root.resolve()),
        "code_commit": design["code_commit"],
        "execution_code_commit": design["code_commit"],
        "report_code_commit": commit,
        "report_checkout_commit": commit,
        "report_checkout_clean": clean,
        "report_checkout_dirty_entries": dirty,
        "report_repository_root": str(REPOSITORY_ROOT.resolve()),
        "design_sha256": _sha256(root / "design.json"),
        "preflight_sha256": design["preflight_sha256"],
        "report_script_sha256": _sha256(Path(__file__).resolve()),
        "report_plan_sha256": _sha256(
            REPOSITORY_ROOT / "research" / "physics_background_pairing" / "PLAN.md"
        ),
        "report_uv_lock_sha256": _sha256(REPOSITORY_ROOT / "uv.lock"),
        "target_trials_per_study": design["target_trials_per_study"],
        "search_epochs": design["search_epochs"],
        "retrain_epochs": design["retrain_epochs"],
        "retrain_seed": 123,
        "target_rate_khz": TARGET_RATE_KHZ,
        "base_rate_khz": BASE_RATE_KHZ,
        "study_database_sha256": databases,
        "study_evidence": study_evidence if final else None,
    }
    if execution_evidence is None:
        raise RuntimeError("Execution-worktree evidence is required for provenance.")
    value.update(execution_evidence)
    manifest = root / "selection" / "pareto_retrain_manifest.json"
    oracle = root / "results" / "oracle_winners.json"
    value["pareto_manifest_sha256"] = _sha256(manifest) if manifest.is_file() else None
    value["oracle_winners_sha256"] = _sha256(oracle) if oracle.is_file() else None
    return value


def _build_snapshot(
    root: Path,
    output: Path,
    *,
    final: bool,
    update_state: bool,
    validate_mlflow: bool = True,
    report_state: tuple[str, bool, list[str]] | None = None,
    execution_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build search-state files, optionally under strict final-publication rules."""
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
    selected = {
        (str(row["model"]), int(row["trial_number"]))
        for row in trial_rows
        if bool(row["selected_for_freeze"])
    }
    if final:
        target = int(design["target_trials_per_study"])
        selected_counts = Counter(model for model, _ in selected)
        if any(selected_counts[model] != target for model in MODEL_ORDER):
            raise RuntimeError(
                f"Final selected-trial counts differ from {target}: {dict(selected_counts)}."
            )
        selected_bindings = {key: bindings[key] for key in selected if key in bindings}
        if len(selected_bindings) != len(selected):
            missing = sorted(selected - selected_bindings.keys())
            raise RuntimeError(f"Selected trials lack explicit run bindings: {missing[:10]}")
        candidate_rows = _checkpoint_candidate_rows(root, design, selected_bindings, final=True)
        availability_rows = _checkpoint_availability_rows(candidate_rows)
        _write_csv(
            output / "checkpoint_candidates.csv",
            candidate_rows,
            _checkpoint_candidate_fields(),
        )
        _write_csv(
            output / "checkpoint_candidate_availability.csv",
            availability_rows,
            list(availability_rows[0]),
        )
    checkpoints, mlflow = _checkpoint_rows(root, design, include_retrains=False)
    if final:
        checkpoints = [
            row for row in checkpoints if (row["model"], int(row["trial_number"])) in selected
        ]
        mlflow = [row for row in mlflow if (row["model"], int(row["trial_number"])) in selected]
    checkpoints = _annotate_checkpoint_families(design, checkpoints)
    if validate_mlflow:
        mlflow = _validate_mlflow_runs(root, design, mlflow, final=final)
    checkpoint_fields = _checkpoint_fields()
    _write_csv(output / "checkpoint_selections.csv", checkpoints, checkpoint_fields)
    _write_csv(output / "mlflow_runs.csv", mlflow, _mlflow_fields())
    _write_json(
        output / "provenance.json",
        _provenance(
            root,
            design,
            final=False,
            report_state=report_state,
            execution_evidence=execution_evidence,
        ),
    )
    latest = {
        "schema_version": 1,
        "code_commit": design["code_commit"],
        "studies": status_rows,
        "ready_to_freeze": all(row["usable_complete"] >= row["target"] for row in status_rows),
    }
    if update_state:
        _write_json(root / "state" / "latest_status.json", latest)
        _write_experiment_matrix(root, design, status_rows)
    _copy_state_ledgers(root, output, design, final=final)
    return latest


def snapshot(root: Path, output: Path, execution_repository: Path | None = None) -> dict[str, Any]:
    """Export a search-only snapshot while allowing valid active-run prefixes."""
    root = root.resolve()
    design = _design(root)
    execution_evidence = _execution_repository_evidence(root, design, execution_repository)
    return _build_snapshot(
        root,
        output,
        final=False,
        update_state=True,
        execution_evidence=execution_evidence,
    )


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
        "checkpoint_family",
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
        "tracking_uri",
        "status",
        "lifecycle_stage",
        "artifact_uri",
        "expected_epochs",
        "observed_epochs",
        "required_history_count",
        "histories_validated",
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


def _confined_file(path: Path, parent: Path, label: str) -> Path:
    """Resolve one required file without allowing it to escape its evidence tree."""
    resolved = path.resolve(strict=True)
    expected_parent = parent.resolve(strict=True)
    if not resolved.is_file() or not resolved.is_relative_to(expected_parent):
        raise RuntimeError(f"{label} is outside its authenticated directory: {resolved}")
    return resolved


def _restart_state_evidence(
    root: Path,
    manifest: dict[str, Any],
    expected: dict[str, Any],
    marker: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Authenticate one retrain's complete restart history independently of execution code."""
    index = int(expected["retrain_index"])
    state_dir = root / "retraining" / "restart_state" / f"{index:04d}"
    state_path = _confined_file(
        Path(str(marker.get("restart_state_artifact", ""))),
        state_dir,
        "Retrain restart state",
    )
    if state_path != (state_dir / "state.json").resolve():
        raise RuntimeError(f"Retrain restart state has the wrong path: {state_path}")
    state_sha256 = _sha256(state_path)
    if state_sha256 != marker.get("restart_state_artifact_sha256"):
        raise RuntimeError(f"Changed retrain restart state: {state_path}")
    state = _read_json(state_path)
    manifest_path = (root / "selection" / "pareto_retrain_manifest.json").resolve(strict=True)
    run_name = f"{expected['model']}__trial{int(expected['trial_number']):04d}"
    experiment_name = f"physics_bgpair_retrain_{expected['model']}"
    expected_identity = {
        "code_commit": manifest["code_commit"],
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "retrain_index": index,
        "row": expected,
        "row_sha256": _json_sha256(expected),
        "experiment_name": experiment_name,
        "run_name": run_name,
        "hydra_dir": str((root / "retraining" / "hydra" / f"{index:04d}").resolve()),
        "trainer_default_root_dir": str((state_dir / "lightning").resolve()),
        "mlflow_tracking_uri": (
            f"file:{(root / 'runtime' / 'logs' / 'mlflow' / 'mlruns').resolve()}"
        ),
    }
    completed_at = marker.get("completed_at")
    mlflow_run_id = marker.get("mlflow_run_id")
    if (
        state.get("schema_version") != 1
        or state.get("identity") != expected_identity
        or not isinstance(mlflow_run_id, str)
        or not mlflow_run_id
        or state.get("mlflow_run_id") != mlflow_run_id
        or not isinstance(completed_at, str)
        or not completed_at
        or state.get("completed_at") != completed_at
        or marker.get("experiment_name") != experiment_name
        or marker.get("run_name") != run_name
    ):
        raise RuntimeError(f"Retrain restart identity differs: {state_path}")

    boundaries = [40, 80, 120, 160, 200]
    segments = state.get("segments")
    attempts = state.get("attempts")
    if (
        not isinstance(segments, list)
        or [row.get("epoch_boundary") for row in segments] != boundaries
    ):
        raise RuntimeError(f"Retrain segment boundaries are incomplete: {state_path}")
    if not isinstance(attempts, list) or not attempts:
        raise RuntimeError(f"Retrain attempt history is missing: {state_path}")
    if [row.get("attempt_number") for row in attempts] != list(range(1, len(attempts) + 1)):
        raise RuntimeError(f"Retrain attempt numbers are not contiguous: {state_path}")
    for attempt in attempts:
        phase = attempt.get("phase")
        boundary = attempt.get("target_epoch_boundary")
        if phase not in {"training", "evaluation", "finalization"}:
            raise RuntimeError(f"Retrain attempt has an invalid phase: {state_path}")
        if phase == "training" and boundary not in boundaries:
            raise RuntimeError(f"Retrain training attempt has an invalid boundary: {state_path}")
        if phase in {"evaluation", "finalization"} and boundary != 200:
            raise RuntimeError(f"Retrain terminal attempt has an invalid boundary: {state_path}")

    segment_root = state_dir / "segments"
    evidence_rows: list[dict[str, Any]] = []
    for segment in segments:
        boundary = int(segment["epoch_boundary"])
        attempt_number = int(segment.get("attempt_number", -1))
        if attempt_number < 1 or attempt_number > len(attempts):
            raise RuntimeError(f"Retrain segment has no owning attempt: {state_path}")
        owner = attempts[attempt_number - 1]
        if (
            owner.get("phase") != "training"
            or owner.get("target_epoch_boundary") != boundary
            or not owner.get("completed_at")
        ):
            raise RuntimeError(f"Retrain segment owner is incomplete: {state_path}")
        checkpoint = _confined_file(
            Path(str(segment.get("continuation_checkpoint", ""))),
            segment_root,
            "Retrain continuation checkpoint",
        )
        config = _confined_file(
            Path(str(segment.get("resolved_config", ""))),
            segment_root,
            "Retrain segment config",
        )
        checkpoint_sha256 = _sha256(checkpoint)
        config_sha256 = _sha256(config)
        if checkpoint_sha256 != segment.get(
            "continuation_checkpoint_sha256"
        ) or config_sha256 != segment.get("resolved_config_sha256"):
            raise RuntimeError(f"Changed retrain segment artifact: {state_path}")
        import torch

        payload = torch.load(  # nosec B614 -- confined, hash-authenticated campaign artifact
            checkpoint, map_location="cpu", weights_only=False
        )
        rng_state = payload.get("callbacks", {}).get(EXACT_RETRAIN_RNG_STATE_KEY)
        from src.callbacks.exact_retrain_rng import ExactRetrainRNGState

        try:
            ExactRetrainRNGState.validate_state(
                rng_state,
                require_datamodule_shuffler=True,
                expected_cuda_states=1,
            )
        except RuntimeError as error:
            raise RuntimeError(
                f"Retrain checkpoint lacks exact RNG state: {checkpoint}"
            ) from error
        completed_epochs = int(payload.get("epoch", -1)) + 1
        try:
            progress = payload["loops"]["fit_loop"]["epoch_progress"]["total"]
            loop_counts = [int(progress[name]) for name in ("ready", "started", "processed")]
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"Retrain checkpoint lacks completed-epoch loop state: {checkpoint}"
            ) from error
        if completed_epochs != boundary or loop_counts != [completed_epochs] * 3:
            raise RuntimeError(
                f"Retrain checkpoint is not the declared completed-epoch boundary: {checkpoint}"
            )
        evidence_rows.append(
            {
                "retrain_index": index,
                "model": expected["model"],
                "trial_number": expected["trial_number"],
                "epoch_boundary": boundary,
                "attempt_number": attempt_number,
                "attempt_started_at": owner.get("started_at", ""),
                "attempt_completed_at": owner["completed_at"],
                "slurm_job_id": owner.get("slurm_job_id") or "",
                "slurm_array_job_id": owner.get("slurm_array_job_id") or "",
                "slurm_array_task_id": owner.get("slurm_array_task_id") or "",
                "continuation_checkpoint": _relative(checkpoint, root),
                "continuation_checkpoint_sha256": checkpoint_sha256,
                "resolved_config": _relative(config, root),
                "resolved_config_sha256": config_sha256,
            }
        )
    terminal = attempts[-1]
    if terminal.get("phase") not in {"evaluation", "finalization"} or not terminal.get(
        "completed_at"
    ):
        raise RuntimeError(f"Retrain terminal attempt is incomplete: {state_path}")
    if not any(
        attempt.get("phase") == "evaluation" and attempt.get("completed_at")
        for attempt in attempts
    ):
        raise RuntimeError(f"Retrain evaluation attempt is missing: {state_path}")
    restart_count = len(attempts) - 1
    if marker.get("restart_count") != restart_count:
        raise RuntimeError(f"Retrain restart count differs: {state_path}")
    return (
        {
            "restart_count": restart_count,
            "restart_state_artifact": _relative(state_path, root),
            "restart_state_artifact_sha256": state_sha256,
            "mlflow_run_id": mlflow_run_id,
            "attempt_count": len(attempts),
            "terminal_phase": terminal["phase"],
            "segment_boundaries_json": _json_cell(boundaries),
        },
        evidence_rows,
    )


def _marker_rows(
    root: Path,
    manifest: dict[str, Any],
    restart_summaries: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
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
        if restart_summaries is None:
            restart_summary, _ = _restart_state_evidence(root, manifest, expected, marker)
        else:
            restart_summary = restart_summaries[index]
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
                **restart_summary,
            }
        )
    return rows


def _restart_evidence_bundle(
    root: Path, manifest: dict[str, Any]
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    """Authenticate restart state once and return summaries plus segment rows."""
    summaries: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for expected in manifest["retrain_runs"]:
        index = int(expected["retrain_index"])
        marker = _read_json(root / "retraining" / "markers" / f"{index:04d}.json")
        summary, evidence = _restart_state_evidence(root, manifest, expected, marker)
        summaries[index] = summary
        rows.extend(evidence)
    return summaries, rows


def _checkpoint_identity(name: str) -> str:
    """Normalize evaluator checkpoint labels exactly as campaign aggregation does."""
    parts = str(name).split("ds=")
    return parts[1].split("__")[0] if len(parts) > 1 else parts[0]


def _marker_efficiencies(marker: dict[str, Any], checkpoint: str, score: str) -> dict[str, float]:
    """Reconstruct one candidate's authenticated 20-anomaly efficiency vector."""
    efficiencies: dict[str, float] = {}
    score_folder = f"/eff_{score}/"
    for entry in marker["value_files"]:
        path = Path(entry["path"])
        if score_folder not in path.as_posix():
            continue
        if _sha256(path) != entry["sha256"]:
            raise RuntimeError(f"Changed downstream artifact: {path}")
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if (
                    _checkpoint_identity(row["checkpoint"]) != checkpoint
                    or row["metric"] != "efficiency_operational"
                ):
                    continue
                anomaly = row["intervention"]
                if anomaly in efficiencies:
                    raise RuntimeError(
                        f"Duplicate downstream efficiency for {score}, {checkpoint}, {anomaly}."
                    )
                efficiencies[anomaly] = float(row["value"])
    if set(efficiencies) != set(DOWNSTREAM_DATASETS) or not all(
        math.isfinite(value) and 0.0 <= value <= 1.0 for value in efficiencies.values()
    ):
        raise RuntimeError(
            f"Expected exactly 20 finite downstream efficiencies for {score}, {checkpoint}."
        )
    return efficiencies


def _deterministic_oracle_winner(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Select maximum mean efficiency, breaking exact ties by membership index."""
    if not candidates:
        raise RuntimeError("Cannot select an oracle winner from an empty front.")
    return min(
        candidates,
        key=lambda row: (-float(row["mean_downstream_efficiency"]), int(row["membership_index"])),
    )


def _validate_oracle_candidates(
    root: Path, manifest: dict[str, Any], oracle: dict[str, Any]
) -> None:
    """Rebuild every oracle candidate and prove all declared winners are correct."""
    retrains = {int(row["retrain_index"]): row for row in manifest["retrain_runs"]}
    expected_memberships = {
        int(row["membership_index"]): row for row in manifest["front_memberships"]
    }
    candidates = oracle.get("all_front_memberships", [])
    candidate_by_index: dict[int, dict[str, Any]] = {}
    for candidate in candidates:
        index = int(candidate["membership_index"])
        if index in candidate_by_index:
            raise RuntimeError(f"Duplicate oracle front membership index: {index}.")
        candidate_by_index[index] = candidate
    if set(candidate_by_index) != set(expected_memberships):
        raise RuntimeError("Oracle candidates do not exactly cover the frozen front memberships.")

    verified_by_front: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for index, membership in expected_memberships.items():
        candidate = candidate_by_index[index]
        if any(candidate.get(key) != value for key, value in membership.items()):
            raise RuntimeError(f"Oracle candidate differs from frozen membership {index}.")
        retrain_index = int(membership["retrain_index"])
        retrain = retrains[retrain_index]
        if (
            candidate.get("params") != retrain["params"]
            or candidate["model"] != retrain["model"]
            or int(candidate["trial_number"]) != int(retrain["trial_number"])
        ):
            raise RuntimeError(f"Oracle candidate differs from retrain row {retrain_index}.")
        marker = _read_json(root / "retraining" / "markers" / f"{retrain_index:04d}.json")
        if marker.get("row") != retrain:
            raise RuntimeError(f"Retrain marker differs for oracle candidate {index}.")
        optimized_path = Path(marker["optimized_metric_artifact"])
        if _sha256(optimized_path) != marker["optimized_metric_artifact_sha256"]:
            raise RuntimeError(f"Changed optimized artifact: {optimized_path}")
        optimized = _read_json(optimized_path)
        selection = optimized["selections"][membership["logical_objective"]]
        pair = [float(value) for value in selection["optimized_metric"]]
        if (
            candidate.get("optimized_ckpt_name") != selection["optimized_ckpt_name"]
            or [float(value) for value in candidate.get("retrained_objective_values", [])] != pair
            or len(pair) != 2
            or not all(math.isfinite(value) for value in pair)
        ):
            raise RuntimeError(
                f"Oracle candidate selection differs from retrain artifact {index}."
            )
        efficiencies = _marker_efficiencies(
            marker, selection["optimized_ckpt_name"], membership["score"]
        )
        if candidate.get("downstream_efficiencies") != efficiencies:
            raise RuntimeError(f"Oracle candidate efficiencies differ from marker {index}.")
        mean = sum(efficiencies.values()) / len(efficiencies)
        if not math.isclose(
            float(candidate.get("mean_downstream_efficiency", math.nan)),
            mean,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise RuntimeError(f"Oracle candidate mean is incorrect for membership {index}.")
        verified_by_front.setdefault(
            (membership["model"], membership["logical_objective"]), []
        ).append(candidate)

    declared = {(row["model"], row["logical_objective"]): row for row in oracle.get("winners", [])}
    if len(declared) != len(oracle.get("winners", [])):
        raise RuntimeError("Oracle result contains duplicate logical winners.")
    if set(declared) != set(verified_by_front):
        raise RuntimeError("Oracle winners do not exactly cover the frozen logical fronts.")
    for front, front_candidates in verified_by_front.items():
        expected = _deterministic_oracle_winner(front_candidates)
        if declared[front] != expected:
            raise RuntimeError(
                f"Declared oracle winner is not the deterministic downstream argmax for {front}."
            )


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
    _validate_oracle_candidates(root, manifest, oracle)
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


def _gzip_copy(source: Path, destination: Path) -> None:
    """Copy one text log into a deterministic gzip member."""
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    with source.open("rb") as input_handle, temporary.open("wb") as raw_output:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_output,
            mtime=0,
        ) as output_handle:
            shutil.copyfileobj(input_handle, output_handle)
    temporary.replace(destination)


def _copy_run_evidence(
    root: Path,
    output: Path,
    winners: list[dict[str, Any]],
    jobs: list[dict[str, str]],
) -> None:
    """Package winner-resolved Hydra configs and compact training/Slurm logs."""
    bindings = _binding_index(root)
    config_rows = []
    run_log_rows = []
    search_runs = sorted(
        {(row["model"], int(row["trial_number"])) for row in winners},
        key=lambda item: (MODEL_ORDER.index(item[0]), item[1]),
    )
    retrain_runs = sorted({int(row["retrain_index"]) for row in winners})

    def copy_run(
        phase: str,
        model: str,
        trial_number: int,
        retrain_index: int | None,
        run_dir: Path,
    ) -> None:
        identity = (
            f"{model}__trial{trial_number:04d}"
            if retrain_index is None
            else f"{retrain_index:04d}__{model}__trial{trial_number:04d}"
        )
        destination = output / "configs" / phase / identity
        for name in ("config.yaml", "overrides.yaml"):
            source = run_dir / ".hydra" / name
            if not source.is_file():
                raise FileNotFoundError(f"Missing resolved Hydra evidence: {source}")
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            config_rows.append(
                {
                    "phase": phase,
                    "model": model,
                    "trial_number": trial_number,
                    "retrain_index": "" if retrain_index is None else retrain_index,
                    "kind": name,
                    "source": str(source.resolve()),
                    "source_sha256": _sha256(source),
                    "portable_path": target.relative_to(output).as_posix(),
                    "portable_sha256": _sha256(target),
                }
            )
        source_log = run_dir / "train.log"
        target_log = output / "logs" / "runs" / phase / f"{identity}.train.log.gz"
        _gzip_copy(source_log, target_log)
        run_log_rows.append(
            {
                "phase": phase,
                "model": model,
                "trial_number": trial_number,
                "retrain_index": "" if retrain_index is None else retrain_index,
                "source": str(source_log.resolve()),
                "source_sha256": _sha256(source_log),
                "portable_path": target_log.relative_to(output).as_posix(),
                "portable_sha256": _sha256(target_log),
            }
        )

    for model, trial_number in search_runs:
        binding = bindings[(model, trial_number)]
        copy_run(
            "search",
            model,
            trial_number,
            None,
            Path(binding["hydra_run_dir"]),
        )
    manifest = _read_json(root / "selection" / "pareto_retrain_manifest.json")
    retrain_by_index = {int(row["retrain_index"]): row for row in manifest["retrain_runs"]}
    for retrain_index in retrain_runs:
        row = retrain_by_index[retrain_index]
        marker = _read_json(root / "retraining" / "markers" / f"{retrain_index:04d}.json")
        copy_run(
            "retrain",
            row["model"],
            int(row["trial_number"]),
            retrain_index,
            Path(marker["optimized_metric_artifact"]).parent,
        )
    _write_csv(
        output / "configs" / "index.csv",
        config_rows,
        [
            "phase",
            "model",
            "trial_number",
            "retrain_index",
            "kind",
            "source",
            "source_sha256",
            "portable_path",
            "portable_sha256",
        ],
    )
    _write_csv(
        output / "logs" / "runs" / "index.csv",
        run_log_rows,
        [
            "phase",
            "model",
            "trial_number",
            "retrain_index",
            "source",
            "source_sha256",
            "portable_path",
            "portable_sha256",
        ],
    )

    slurm_rows = []
    for index, job in enumerate(jobs):
        source = Path(job["log_path"])
        job_id = job.get("job_id") or f"row{index:04d}"
        target = output / "logs" / "slurm" / f"{index:04d}__{job_id}__{source.name}.gz"
        _gzip_copy(source, target)
        slurm_rows.append(
            {
                "job_id": job_id,
                "source": str(source.resolve()),
                "source_sha256": _sha256(source),
                "portable_path": target.relative_to(output).as_posix(),
                "portable_sha256": _sha256(target),
            }
        )
    if len(slurm_rows) != len(jobs):  # pragma: no cover - structural guard
        raise RuntimeError("Not every recorded Slurm log was copied.")
    _write_csv(
        output / "logs" / "slurm" / "index.csv",
        slurm_rows,
        ["job_id", "source", "source_sha256", "portable_path", "portable_sha256"],
    )


def _write_figures(
    output: Path,
    winners: list[dict[str, Any]],
    downstream: list[dict[str, Any]],
) -> None:
    """Render the compact descriptive figures referenced by the final report."""
    cache = Path(tempfile.mkdtemp(prefix="adatl1-physics-bgpair-mpl-"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 160, "font.size": 9})

    metric_order = ("cap", "wasserstein", "drift")
    metric_means = [
        statistics.fmean(
            float(row["mean_efficiency"]) for row in winners if row["metric"] == metric
        )
        for metric in metric_order
    ]
    figure, axis = plt.subplots(figsize=(6.2, 3.8))
    axis.bar(metric_order, metric_means, color=("#4477AA", "#EE6677", "#228833"))
    axis.set_ylabel("Mean downstream efficiency")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Reported Pareto winner performance by selection metric")
    figure.tight_layout()
    figure.savefig(figure_dir / "metric_summary.png")
    plt.close(figure)

    pairing_means = [
        statistics.fmean(
            float(row["mean_efficiency"])
            for row in winners
            if row["metric"] == "cap" and row["strategy"] == strategy
        )
        for strategy in CAP_STRATEGIES
    ]
    figure, axis = plt.subplots(figsize=(8.2, 4.2))
    axis.bar(range(len(CAP_STRATEGIES)), pairing_means, color="#4477AA")
    axis.set_xticks(range(len(CAP_STRATEGIES)), CAP_STRATEGIES, rotation=25, ha="right")
    axis.set_ylabel("Mean downstream efficiency")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("CAP pairing comparison across reported winners")
    figure.tight_layout()
    figure.savefig(figure_dir / "cap_pairing_summary.png")
    plt.close(figure)

    score_models = ("ae", "vae")
    score_names = ("native", "residual_oas")
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    width = 0.36
    for offset, score in enumerate(score_names):
        values = [
            statistics.fmean(
                float(row["mean_efficiency"])
                for row in winners
                if row["model"] == model and row["score"] == score
            )
            for model in score_models
        ]
        positions = [index + (offset - 0.5) * width for index in range(len(score_models))]
        axis.bar(positions, values, width=width, label=score)
    axis.set_xticks(range(len(score_models)), [name.upper() for name in score_models])
    axis.set_ylabel("Mean downstream efficiency")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Native versus masked residual-OAS selection")
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_dir / "ae_vae_score_summary.png")
    plt.close(figure)

    anomaly_index = {name: index for index, name in enumerate(DOWNSTREAM_DATASETS)}
    winner_index = {
        (row["model"], row["logical_objective"]): index for index, row in enumerate(winners)
    }
    matrix = [[math.nan] * len(DOWNSTREAM_DATASETS) for _ in winners]
    for row in downstream:
        row_index = winner_index[(row["model"], row["logical_objective"])]
        column_index = anomaly_index[row["anomaly"]]
        matrix[row_index][column_index] = float(row["efficiency_operational"])
    if any(not math.isfinite(value) for values in matrix for value in values):
        raise RuntimeError("Winner/anomaly heatmap is incomplete.")
    figure, axis = plt.subplots(figsize=(13.5, 11.5))
    image = axis.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    axis.set_xticks(range(len(DOWNSTREAM_DATASETS)), DOWNSTREAM_DATASETS, rotation=75, ha="right")
    axis.set_yticks(
        range(len(winners)),
        [f"{row['model']} · {row['logical_objective']}" for row in winners],
        fontsize=5.5,
    )
    axis.set_title("Operational efficiency of all 56 reported winners")
    figure.colorbar(image, ax=axis, label="Efficiency")
    figure.tight_layout()
    figure.savefig(figure_dir / "winner_anomaly_heatmap.png")
    plt.close(figure)


def _write_report(output: Path, design: dict[str, Any], winners: list[dict[str, Any]]) -> None:
    """Write the self-contained human-readable report from portable rows."""
    by_metric: dict[str, list[float]] = {}
    by_model: dict[str, list[float]] = {}
    by_score: dict[str, list[float]] = {}
    for row in winners:
        by_metric.setdefault(row["metric"], []).append(float(row["mean_efficiency"]))
        by_model.setdefault(row["model"], []).append(float(row["mean_efficiency"]))
        by_score.setdefault(row["score"], []).append(float(row["mean_efficiency"]))
    with (output / "checkpoint_candidate_availability.csv").open(encoding="utf-8") as handle:
        candidate_availability = list(csv.DictReader(handle))
    configured_candidates = sum(int(row["configured"]) for row in candidate_availability)
    available_candidates = sum(int(row["available"]) for row in candidate_availability)
    stable_rows = [row for row in candidate_availability if row["slot"] == "stable_native"]
    stable_configured = sum(int(row["configured"]) for row in stable_rows)
    stable_available = sum(int(row["available"]) for row in stable_rows)
    with (output / "analysis" / "checkpoint_family_summary.csv").open(encoding="utf-8") as handle:
        family_rows = list(csv.DictReader(handle))
    family_counts = Counter(
        {
            (population, family): sum(
                int(row["count"])
                for row in family_rows
                if row["population"] == population and row["family"] == family
            )
            for population in ("search", "retrain_candidate", "winner")
            for family in ("last", "primary_ema", "stable_native")
        }
    )
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
        "![Selection metric summary](figures/metric_summary.png)",
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
            "![CAP pairing summary](figures/cap_pairing_summary.png)",
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
            "![AE/VAE native versus residual OAS](figures/ae_vae_score_summary.png)",
            "",
            f"The score groups contain {len(by_score.get('native', []))} native and "
            f"{len(by_score.get('residual_oas', []))} residual-OAS winner rows. Exact paired "
            "per-anomaly comparisons are reproducible from `downstream_efficiencies.csv`.",
            "",
            "## Model comparison and robustness",
            "",
            "![Winner-by-anomaly efficiency heatmap](figures/winner_anomaly_heatmap.png)",
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
            f"All {configured_candidates} configured physical search-candidate slots are indexed "
            f"in `checkpoint_candidates.csv`; {available_candidates} exist. Stable-native "
            f"checkpoints were available for {stable_available}/{stable_configured} trials; "
            "absence is permitted by the reference callback, while last and primary-EMA "
            "candidates are mandatory. `checkpoint_selections.csv` records every selected "
            "candidate, its family, and Q′/Q″ pair. `primary_metrics.csv` adds held-out CAP, "
            "rank correlation, W1, forward drift, and post-hoc reverse G→E drift for the "
            "reported checkpoints.",
            "",
            "| Population | Last | Primary EMA | Stable native |",
            "|---|---:|---:|---:|",
        ]
    )
    for population, label in (
        ("search", "Search selections"),
        ("retrain_candidate", "Retrained front candidates"),
        ("winner", "Reported winners"),
    ):
        lines.append(
            f"| {label} | {family_counts[(population, 'last')]} | "
            f"{family_counts[(population, 'primary_ema')]} | "
            f"{family_counts[(population, 'stable_native')]} |"
        )
    lines.extend(
        [
            "",
            "The model/score/metric/strategy breakdown is in "
            "`analysis/checkpoint_family_summary.csv`.",
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
            "`checkpoint_candidates.csv`, `checkpoint_candidate_availability.csv`, "
            "`primary_metrics.csv`, `downstream_efficiencies.csv`, `winners.csv`, and "
            "`mlflow_runs.csv`. Winner-resolved configurations and compressed execution logs "
            "are indexed under `configs/` and `logs/`.",
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


def _verify_manifest(output: Path) -> None:
    """Require the portable manifest to cover exactly every other package file."""
    manifest = output / "manifest.sha256"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    expected = {
        path.relative_to(output).as_posix(): _sha256(path)
        for path in output.rglob("*")
        if path.is_file() and path != manifest
    }
    observed = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator or relative in observed:
            raise RuntimeError("Portable manifest is malformed or contains duplicates.")
        observed[relative] = digest
    if observed != expected:
        raise RuntimeError("Portable manifest does not exactly authenticate the package.")


def _tree_digests(root: Path) -> dict[str, str]:
    """Return deterministic content identities for idempotent publication."""
    if not root.is_dir():
        return {}
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in root.rglob("*")
        if path.is_file()
    }


def _publish_staged(stage: Path, output: Path) -> None:
    """Recoverably replace one canonical directory after a complete staged build."""
    if _tree_digests(stage) == _tree_digests(output) and output.is_dir():
        shutil.rmtree(stage)
        return
    backup = output.with_name(f".{output.name}.backup-{os.getpid()}")
    if backup.exists():
        raise RuntimeError(f"Refusing to overwrite stale publication backup: {backup}")
    moved_old = False
    try:
        if output.exists():
            output.rename(backup)
            moved_old = True
        stage.rename(output)
    except Exception:
        if moved_old and not output.exists() and backup.exists():
            backup.rename(output)
        raise
    if moved_old:
        shutil.rmtree(backup)


def _copy_repository_report(output: Path, destination: Path) -> None:
    """Copy the published human report outside the manifested portable tree."""
    destination = destination.expanduser().resolve()
    if destination.is_relative_to(output.resolve()):
        raise RuntimeError("The optional repository report must remain outside portable output.")
    source = output / "REPORT.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.read_bytes() == source.read_bytes():
        return
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def _build_final(
    root: Path,
    output: Path,
    *,
    report_state: tuple[str, bool, list[str]],
    study_evidence: dict[str, dict[str, Any]],
    execution_evidence: dict[str, Any],
) -> dict[str, Any]:
    """Build and validate a complete final package in a private staging directory."""
    design = _design(root)
    latest = _build_snapshot(
        root,
        output,
        final=True,
        update_state=False,
        validate_mlflow=False,
        report_state=report_state,
        execution_evidence=execution_evidence,
    )
    status_rows, _ = _trial_exports(root, design)
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
    restart_summaries, restart_evidence_rows = _restart_evidence_bundle(root, manifest)
    retrain_rows = _marker_rows(root, manifest, restart_summaries)
    winner_rows, downstream_rows, primary_rows = _winner_exports(root, design, manifest, oracle)
    _append_test_metrics(root, manifest, oracle, primary_rows)
    checkpoints, mlflow = _checkpoint_rows(root, design, include_retrains=True)
    checkpoints = [
        row
        for row in checkpoints
        if row["phase"] != "search" or (row["model"], int(row["trial_number"])) in selected
    ]
    mlflow = [
        row
        for row in mlflow
        if row["phase"] != "search" or (row["model"], int(row["trial_number"])) in selected
    ]
    mlflow = _validate_mlflow_runs(root, design, mlflow, final=True)
    annotated_checkpoints = _annotate_checkpoint_families(design, checkpoints)
    candidate_rows = list(
        csv.DictReader((output / "checkpoint_candidates.csv").open(encoding="utf-8"))
    )
    available_candidates = {
        (row["model"], int(row["trial_number"]), row["checkpoint_family_id"])
        for row in candidate_rows
        if row["available"] == "True"
    }
    for row in annotated_checkpoints:
        if row["phase"] != "search":
            continue
        key = (
            row["model"],
            int(row["trial_number"]),
            row["optimized_ckpt_name"],
        )
        if key not in available_candidates:
            raise RuntimeError(f"Selected search checkpoint is not retained: {key}.")
    markers = []
    for row in manifest["retrain_runs"]:
        marker = _read_json(
            root / "retraining" / "markers" / f"{int(row['retrain_index']):04d}.json"
        )
        markers.append({**marker, "retrain_index": int(row["retrain_index"])})
    artifact_audit = audit_physics_background_pairing_artifacts(
        campaign_root=root,
        design=design,
        manifest=manifest,
        markers=markers,
        checkpoint_selections=[row for row in annotated_checkpoints if row["phase"] == "retrain"],
    )
    audited_families = {
        (row["model"], int(row["retrain_index"]), row["logical_objective"]): (
            row["checkpoint_family_id"],
            row["checkpoint_family"],
        )
        for row in artifact_audit["checkpoint_families"]
    }
    for row in annotated_checkpoints:
        if row["phase"] != "retrain":
            continue
        key = (row["model"], int(row["retrain_index"]), row["logical_objective"])
        expected = (row["optimized_ckpt_name"], row["checkpoint_family"])
        if audited_families.get(key) != expected:
            raise RuntimeError(f"Artifact-audit checkpoint family differs for {key}.")
    analysis = build_physics_background_pairing_analysis(
        design,
        manifest,
        oracle,
        annotated_checkpoints,
        primary_rows,
    )
    analysis_tables = {name: rows for name, rows in analysis.items() if name != "analysis_summary"}
    _write_csv(output / "front_memberships.csv", front_rows, list(front_rows[0]))
    _write_csv(output / "retrains.csv", retrain_rows, list(retrain_rows[0]))
    _write_csv(
        output / "retrain_restart_evidence.csv",
        restart_evidence_rows,
        list(restart_evidence_rows[0]),
    )
    _write_csv(
        output / "checkpoint_selections.csv",
        annotated_checkpoints,
        _checkpoint_fields(),
    )
    _write_csv(output / "mlflow_runs.csv", mlflow, _mlflow_fields())
    _write_csv(output / "winners.csv", winner_rows, list(winner_rows[0]))
    _write_csv(output / "downstream_efficiencies.csv", downstream_rows, list(downstream_rows[0]))
    _write_csv(output / "primary_metrics.csv", primary_rows, list(primary_rows[0]))
    _write_table_group(output / "analysis", analysis_tables)
    _write_json(output / "analysis" / "analysis_summary.json", analysis["analysis_summary"])
    _write_table_group(output / "artifact_audit", artifact_audit)
    jobs, _ = _validated_state_ledgers(root, design, final=True)
    _copy_run_evidence(root, output, winner_rows, jobs)
    _write_figures(output, winner_rows, downstream_rows)
    _write_json(
        output / "provenance.json",
        _provenance(
            root,
            design,
            final=True,
            report_state=report_state,
            study_evidence=study_evidence,
            execution_evidence=execution_evidence,
        ),
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
        "Machine-readable scientific comparisons and checkpoint/config validation are under "
        "`analysis/` and `artifact_audit/`. "
        "The manifest intentionally excludes itself. Large databases, checkpoints, raw data, "
        "and the full MLflow store remain in the authoritative scratch campaign recorded in "
        "`provenance.json`.\n",
        encoding="utf-8",
    )
    _write_manifest(output)
    _verify_manifest(output)
    return {
        "design": design,
        "latest": latest,
        "status_rows": status_rows,
        "manifest": manifest,
        "oracle": oracle,
    }


def finalize(
    root: Path,
    output: Path,
    repository_report: Path | None = None,
    execution_repository: Path | None = None,
) -> None:
    """Fail closed, then transactionally publish the final portable package."""
    root = root.resolve()
    output = output.resolve()
    report_state = _require_clean_reporting_checkout()
    design = _design(root)
    execution_evidence = _execution_repository_evidence(root, design, execution_repository)
    study_evidence = _capture_quiescent_studies(root, design)
    ledger_paths = (root / "state" / "job_history.csv", root / "state" / "failures.jsonl")
    _validated_state_ledgers(root, design, final=True)
    ledger_hashes = {str(path): _sha256(path) for path in ledger_paths}
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        context = _build_final(
            root,
            stage,
            report_state=report_state,
            study_evidence=study_evidence,
            execution_evidence=execution_evidence,
        )
        _assert_studies_unchanged(study_evidence)
        _validated_state_ledgers(root, design, final=True)
        if any(_sha256(Path(path)) != digest for path, digest in ledger_hashes.items()):
            raise RuntimeError("Operator ledgers changed during final reporting.")
        _verify_manifest(stage)
        _publish_staged(stage, output)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    _write_json(root / "state" / "latest_status.json", context["latest"])
    _write_experiment_matrix(
        root,
        context["design"],
        context["status_rows"],
        manifest=context["manifest"],
        oracle=context["oracle"],
    )
    if repository_report is not None:
        _copy_repository_report(output, repository_report)


def main() -> None:
    """Dispatch a search-only snapshot or fail-closed final export."""
    args = parse_args()
    root = args.root.expanduser().resolve()
    output = (args.output or root / "results").expanduser().resolve()
    if args.command == "snapshot":
        print(
            json.dumps(
                snapshot(root, output, execution_repository=args.execution_repository),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        finalize(
            root,
            output,
            repository_report=args.repository_report,
            execution_repository=args.execution_repository,
        )
        print(output / "REPORT.md")


if __name__ == "__main__":
    main()
