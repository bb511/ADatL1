#!/usr/bin/env python3
"""Manage the six-study physics background-pairing campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess  # nosec B404 -- executes frozen repository/Slurm commands
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path(
    "/iopsstor/scratch/cscs/vjimenez/adatl1/campaigns/" "physics_background_shared_20260804"
)
MODELS = ("ae", "vae", "dsae", "dsvae", "svdd", "realnvp")
SCORES_BY_MODEL = {
    "ae": ("native", "residual_oas"),
    "vae": ("native", "residual_oas"),
    "dsae": ("native",),
    "dsvae": ("native",),
    "svdd": ("native",),
    "realnvp": ("native",),
}
CAP_STRATEGIES = (
    "flat_physical",
    "physics_summary",
    "typed_sliced_wasserstein",
    "jetclr",
    "cdf",
)
TARGET_TRIALS = 600
SEARCH_EPOCHS = 50
RETRAIN_EPOCHS = 200
DOWNSTREAM_DATASETS = 20
DOWNSTREAM_DATASET_NAMES = {
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
}


def parse_args() -> argparse.Namespace:
    """Parse one campaign-management command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init")
    subparsers.add_parser("status")
    subparsers.add_parser("refresh-bindings")
    pilot = subparsers.add_parser("pilot")
    pilot.add_argument("--model", choices=MODELS, default="ae")
    packed_pilot = subparsers.add_parser("packed-pilot")
    packed_pilot.add_argument("--model", choices=MODELS, default="ae")
    sweep = subparsers.add_parser("sweep")
    sweep.add_argument("--model", choices=MODELS, required=True)
    sweep.add_argument("--trials", type=int, default=48)
    sweep.add_argument("--jobs", type=int, default=48)
    sweep.add_argument("--timeout-min", type=int, default=720)
    submit = subparsers.add_parser("submit-next")
    submit.add_argument("--trials", type=int, default=48)
    submit.add_argument("--jobs", type=int, default=48)
    submit.add_argument("--timeout-min", type=int, default=720)
    subparsers.add_parser("freeze")
    retrain = subparsers.add_parser("retrain")
    retrain.add_argument("--index", type=int, required=True)
    subparsers.add_parser("aggregate")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_checkpoint_path(checkpoint_root: Path, family_id: str) -> Path:
    """Resolve one evaluator checkpoint-family ID inside its authenticated run root."""
    if family_id == "last":
        path = checkpoint_root / "last.ckpt"
    else:
        family = (checkpoint_root / family_id).resolve()
        if not family.is_relative_to(checkpoint_root.resolve()) or not family.is_dir():
            raise RuntimeError(f"Checkpoint family escapes or is missing: {family_id}")
        candidates = sorted(family.glob("*.ckpt"))
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected one checkpoint for {family_id}, found {len(candidates)}."
            )
        path = candidates[0]
    path = path.resolve()
    if not path.is_relative_to(checkpoint_root.resolve()) or not path.is_file():
        raise RuntimeError(f"Selected checkpoint is missing or outside its run: {path}")
    return path


def _audit_selected_threshold_checkpoints(artifact_path: Path, model: str) -> list[dict[str, Any]]:
    """Prove selected checkpoints restore finite, score-specific thresholds."""
    import torch

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    checkpoint_root = Path(artifact["run_provenance"]["checkpoint_run_dir"]).resolve()
    if not checkpoint_root.is_dir():
        raise RuntimeError(f"Checkpoint run directory is missing: {checkpoint_root}")
    expected = {"thres_operational"}
    if model in {"ae", "vae"}:
        expected.add("thres_operational__residual_oas")
    family_ids = sorted(
        {str(row["optimized_ckpt_name"]) for row in artifact["selections"].values()}
    )
    rows = []
    for family_id in family_ids:
        checkpoint = _selected_checkpoint_path(checkpoint_root, family_id)
        payload = torch.load(  # nosec B614 -- authenticated campaign-local checkpoint
            checkpoint, map_location="cpu", weights_only=False
        )
        state = payload.get("state_dict", {})
        observed = {str(key) for key in state if str(key).startswith("thres_operational")}
        if observed != expected:
            raise RuntimeError(
                f"Threshold buffers differ in {checkpoint}: expected {expected}, "
                f"found {observed}."
            )
        values = {}
        for name in sorted(expected):
            value = torch.as_tensor(state[name]).detach().cpu()
            if value.numel() != 1 or not torch.isfinite(value).item():
                raise RuntimeError(f"Threshold buffer {name} is not finite scalar: {checkpoint}")
            values[name] = float(value.item())
        rows.append(
            {
                "checkpoint_family_id": family_id,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "thresholds": values,
            }
        )
    return rows


def _git_commit() -> str:
    """Return the repository's current commit."""
    git = shutil.which("git")
    if git is None:
        raise FileNotFoundError("git is required for campaign provenance.")
    return subprocess.run(  # nosec B603 -- fixed git command
        [git, "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_checkout() -> None:
    """Refuse to pin or execute campaign state from a dirty worktree."""
    git = shutil.which("git")
    if git is None:
        raise FileNotFoundError("git is required for campaign provenance.")
    dirty = subprocess.run(  # nosec B603 -- fixed git command
        [git, "status", "--porcelain", "--untracked-files=all"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("Campaign execution requires a clean commit-pinned worktree.")


def _write_json(path: Path, value: Any, *, create: bool = False) -> None:
    """Write a JSON artifact atomically, optionally refusing replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if create and path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if create and path.exists():
        temporary.unlink()
        raise FileExistsError(path)
    temporary.replace(path)


def _jsonable(value: Any) -> Any:
    """Convert tuples and scalar-like Optuna values to JSON-compatible values."""
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "item"):
        return value.item()
    return value


def _optuna():
    """Import Optuna only for commands that inspect persistent studies."""
    import optuna

    return optuna


def _logical_objectives(model: str) -> list[dict[str, Any]]:
    """Return ordered two-objective views embedded in one model study."""
    if model not in MODELS:
        raise ValueError(f"Unknown model: {model}")
    objectives: list[dict[str, Any]] = []
    for score in SCORES_BY_MODEL[model]:
        for strategy in CAP_STRATEGIES:
            objectives.append(
                {
                    "id": f"{score}__cap__{strategy}",
                    "score": score,
                    "metric": "cap",
                    "strategy": strategy,
                    "directions": ["maximize", "minimize"],
                }
            )
        objectives.extend(
            [
                {
                    "id": f"{score}__wasserstein",
                    "score": score,
                    "metric": "wasserstein",
                    "strategy": None,
                    "directions": ["minimize", "minimize"],
                },
                {
                    "id": f"{score}__drift",
                    "score": score,
                    "metric": "drift",
                    "strategy": None,
                    "directions": ["minimize", "minimize"],
                },
            ]
        )
    for index, objective in enumerate(objectives):
        objective["value_indices"] = [2 * index, 2 * index + 1]
    return objectives


def _studies() -> list[dict[str, Any]]:
    """Construct the six shared model studies."""
    return [
        {
            "id": model,
            "model": model,
            "study_name": f"physics_{model}_background_all",
            "objectives": _logical_objectives(model),
        }
        for model in MODELS
    ]


def initialize(root: Path) -> Path:
    """Freeze campaign design, code, and authenticated input provenance."""
    root = root.expanduser().resolve()
    design_path = root / "design.json"
    if design_path.exists():
        return design_path
    _require_clean_checkout()
    preflight = REPOSITORY_ROOT / "research/physics_background_pairing/preflight.json"
    design = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": _git_commit(),
        "preflight": str(preflight),
        "preflight_sha256": _sha256(preflight),
        "target_trials_per_study": TARGET_TRIALS,
        "search_epochs": SEARCH_EPOCHS,
        "retrain_epochs": RETRAIN_EPOCHS,
        "shared_trial_pool": True,
        "models": list(MODELS),
        "scores_by_model": {model: list(scores) for model, scores in SCORES_BY_MODEL.items()},
        "cap_strategies": list(CAP_STRATEGIES),
        "studies": _studies(),
    }
    _write_json(design_path, design, create=True)
    return design_path


def _load_design(root: Path) -> dict[str, Any]:
    """Load and authenticate the frozen campaign design."""
    _require_clean_checkout()
    design_path = initialize(root)
    design = json.loads(design_path.read_text(encoding="utf-8"))
    if design.get("schema_version") != 2:
        raise RuntimeError("Campaign design is not the six-study schema.")
    if _git_commit() != design["code_commit"]:
        raise RuntimeError("Current commit differs from the frozen campaign design.")
    if _sha256(Path(design["preflight"])) != design["preflight_sha256"]:
        raise RuntimeError("Pair-table preflight changed after campaign initialization.")
    return design


def _study(design: dict[str, Any], model: str) -> dict[str, Any]:
    """Resolve one unique shared study by model identifier."""
    matches = [study for study in design["studies"] if study["model"] == model]
    if len(matches) != 1:
        raise ValueError(f"Unknown or ambiguous campaign model: {model}")
    return matches[0]


def _study_path(root: Path, model: str) -> Path:
    """Return one model's persistent Optuna SQLite path."""
    return root / "studies" / f"{model}.db"


def _study_trials(root: Path, study: dict[str, Any]) -> list[Any]:
    """Load existing trials without creating a database or study."""
    database = _study_path(root, study["model"])
    if not database.is_file():
        return []
    loaded = _optuna().load_study(study_name=study["study_name"], storage=f"sqlite:///{database}")
    return list(loaded.get_trials(deepcopy=False))


def _trial_counts(root: Path, study: dict[str, Any]) -> dict[str, int]:
    """Count persistent trials by Optuna state without creating a study."""
    optuna = _optuna()
    database = _study_path(root, study["model"])
    empty = {state.name.lower(): 0 for state in optuna.trial.TrialState}
    empty["usable_complete"] = 0
    if not database.is_file():
        return empty
    for trial in _study_trials(root, study):
        empty[trial.state.name.lower()] += 1
        if trial.state == optuna.trial.TrialState.COMPLETE and _usable_trial(
            trial, expected_values=2 * len(study["objectives"])
        ):
            empty["usable_complete"] += 1
    return empty


def _write_chunk_bindings(
    root: Path,
    study: dict[str, Any],
    sweep_dir: Path,
    prior_trial_numbers: set[int],
    *,
    all_trials: list[Any] | None = None,
) -> Path:
    """Bind chunk-local Hydra jobs to persistent Optuna trial numbers."""
    new_trials = sorted(
        (
            trial
            for trial in (_study_trials(root, study) if all_trials is None else all_trials)
            if int(trial.number) not in prior_trial_numbers
        ),
        key=lambda trial: int(trial.number),
    )
    run_dirs = (
        sorted(
            (path for path in sweep_dir.iterdir() if path.is_dir() and path.name.isdigit()),
            key=lambda path: int(path.name),
        )
        if sweep_dir.is_dir()
        else []
    )
    run_by_job = {int(path.name): path for path in run_dirs}
    if run_dirs and [int(path.name) for path in run_dirs] != list(range(len(run_dirs))):
        raise RuntimeError(f"Hydra job directories are not contiguous in {sweep_dir}.")
    if len(new_trials) < len(run_dirs):
        raise RuntimeError(f"Fewer Optuna trials than Hydra runs in {sweep_dir}.")
    new_trials = new_trials[: len(run_dirs)]
    rows = []
    for local_index, trial in enumerate(new_trials):
        run_dir = run_by_job.get(local_index)
        optimized = None if run_dir is None else run_dir / "optimized_metric.json"
        has_optimized = optimized is not None and optimized.is_file()
        rows.append(
            {
                "model": study["model"],
                "study_name": study["study_name"],
                "trial_number": int(trial.number),
                "hydra_job_num": local_index,
                "hydra_run_dir": None if run_dir is None else str(run_dir.resolve()),
                "optimized_metric_artifact": (str(optimized.resolve()) if has_optimized else None),
                "optimized_metric_artifact_sha256": (
                    _sha256(optimized) if has_optimized else None
                ),
                "state": trial.state.name,
                "params": _jsonable(trial.params),
                "values": None if trial.values is None else _jsonable(trial.values),
            }
        )
    output = sweep_dir / "trial_bindings.json"
    _write_json(
        output,
        {
            "schema_version": 1,
            "model": study["model"],
            "study_name": study["study_name"],
            "prior_trial_numbers": sorted(prior_trial_numbers),
            "bindings": rows,
        },
    )
    return output


def refresh_bindings(root: Path) -> None:
    """Reconcile chunk bindings after a classified partial-failure repair."""
    design = _load_design(root)
    for study in design["studies"]:
        for chunk_path in sorted((root / "sweeps" / study["model"]).glob("*/chunk.json")):
            chunk = json.loads(chunk_path.read_text(encoding="utf-8"))
            _write_chunk_bindings(
                root,
                study,
                chunk_path.parent,
                {int(number) for number in chunk["prior_trial_numbers"]},
            )


def _usable_trial(trial: Any, *, expected_values: int) -> bool:
    """Return whether a completed trial has the full finite objective vector."""
    return (
        trial.values is not None
        and len(trial.values) == expected_values
        and all(math.isfinite(float(value)) for value in trial.values)
    )


def status(root: Path) -> list[dict[str, Any]]:
    """Return and print completion state for all six studies."""
    design = _load_design(root)
    rows = []
    for study in design["studies"]:
        counts = _trial_counts(root, study)
        rows.append(
            {
                "model": study["model"],
                "study_name": study["study_name"],
                "logical_objectives": len(study["objectives"]),
                **counts,
                "target": TARGET_TRIALS,
            }
        )
    print(json.dumps(rows, indent=2, sort_keys=True))
    return rows


def _dominates(left: Iterable[float], right: Iterable[float], directions: Iterable[str]) -> bool:
    """Return whether left Pareto-dominates right under mixed directions."""
    comparisons = []
    for left_value, right_value, direction in zip(left, right, directions):
        if direction == "maximize":
            comparisons.append((left_value >= right_value, left_value > right_value))
        elif direction == "minimize":
            comparisons.append((left_value <= right_value, left_value < right_value))
        else:
            raise ValueError(f"Unknown optimization direction: {direction}")
    return all(no_worse for no_worse, _ in comparisons) and any(
        better for _, better in comparisons
    )


def _pareto_front(
    trials: list[Any],
    indices: list[int],
    directions: list[str],
) -> list[Any]:
    """Build one logical two-objective front from the shared trial pool."""
    candidates = []
    for trial in trials:
        if trial.values is None or len(trial.values) <= max(indices):
            raise RuntimeError(f"Trial {trial.number} has an incomplete objective vector.")
        candidates.append(trial)
    front = []
    for trial in candidates:
        values = [trial.values[index] for index in indices]
        dominated = any(
            other.number != trial.number
            and _dominates([other.values[index] for index in indices], values, directions)
            for other in candidates
        )
        if not dominated:
            front.append(trial)
    return sorted(front, key=lambda item: item.number)


def freeze(root: Path) -> Path:
    """Freeze all strategy-specific fronts before downstream anomaly evaluation."""
    optuna = _optuna()
    design = _load_design(root)
    manifest_path = root / "selection" / "pareto_retrain_manifest.json"
    if manifest_path.exists():
        return manifest_path

    memberships: list[dict[str, Any]] = []
    retrains: dict[tuple[str, int], dict[str, Any]] = {}
    for study in design["studies"]:
        counts = _trial_counts(root, study)
        if counts["usable_complete"] < TARGET_TRIALS:
            raise RuntimeError(
                f"Cannot freeze {study['model']}: {counts['usable_complete']}/"
                f"{TARGET_TRIALS} trials have complete finite objective vectors."
            )
        database = _study_path(root, study["model"])
        loaded = optuna.load_study(study_name=study["study_name"], storage=f"sqlite:///{database}")
        expected_directions = [
            direction for objective in study["objectives"] for direction in objective["directions"]
        ]
        observed_directions = [direction.name.lower() for direction in loaded.directions]
        if observed_directions != expected_directions:
            raise RuntimeError(
                f"Study {study['study_name']} direction order differs from its design."
            )
        completed = sorted(
            (
                trial
                for trial in loaded.get_trials(deepcopy=False)
                if trial.state == optuna.trial.TrialState.COMPLETE
                and _usable_trial(trial, expected_values=len(expected_directions))
            ),
            key=lambda item: item.number,
        )[:TARGET_TRIALS]
        for objective in study["objectives"]:
            front = _pareto_front(
                completed,
                objective["value_indices"],
                objective["directions"],
            )
            for trial in front:
                key = (study["model"], int(trial.number))
                if key not in retrains:
                    retrains[key] = {
                        "model": study["model"],
                        "trial_number": int(trial.number),
                        "params": _jsonable(trial.params),
                        "logical_objectives": [],
                        "seed": 123,
                        "epochs": RETRAIN_EPOCHS,
                    }
                retrains[key]["logical_objectives"].append(objective["id"])
                memberships.append(
                    {
                        "membership_index": len(memberships),
                        "model": study["model"],
                        "logical_objective": objective["id"],
                        "score": objective["score"],
                        "metric": objective["metric"],
                        "strategy": objective["strategy"],
                        "trial_number": int(trial.number),
                        "objective_values": [
                            float(trial.values[index]) for index in objective["value_indices"]
                        ],
                    }
                )

    retrain_rows = []
    for row in sorted(
        retrains.values(),
        key=lambda item: (MODELS.index(item["model"]), item["trial_number"]),
    ):
        row["retrain_index"] = len(retrain_rows)
        row["logical_objectives"] = sorted(set(row["logical_objectives"]))
        retrain_rows.append(row)
    index_by_trial = {
        (row["model"], row["trial_number"]): row["retrain_index"] for row in retrain_rows
    }
    for membership in memberships:
        membership["retrain_index"] = index_by_trial[
            (membership["model"], membership["trial_number"])
        ]

    payload = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": design["code_commit"],
        "selection_uses_downstream_anomalies": False,
        "front_memberships": memberships,
        "retrain_runs": retrain_rows,
    }
    _write_json(manifest_path, payload, create=True)
    return manifest_path


def _hydra_value(value: Any) -> str:
    """Serialize an Optuna parameter as one Hydra override value."""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, separators=(",", ":"))
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _retrain_suite(model: str) -> str:
    """Select downstream callbacks for the model's available anomaly scores."""
    suffix = "native_oas" if "residual_oas" in SCORES_BY_MODEL[model] else "native"
    return f"physics_background_{suffix}"


def _validate_retrain_marker(marker: Path, expected_row: dict[str, Any]) -> None:
    """Fail closed when resuming an existing retraining marker."""
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2 or payload.get("row") != expected_row:
        raise RuntimeError(f"Retraining marker does not match its frozen row: {marker}")
    optimized = Path(payload["optimized_metric_artifact"])
    if _sha256(optimized) != payload["optimized_metric_artifact_sha256"]:
        raise RuntimeError(f"Retraining optimized artifact changed: {optimized}")
    for entry in payload["value_files"]:
        path = Path(entry["path"])
        if _sha256(path) != entry["sha256"]:
            raise RuntimeError(f"Retraining value artifact changed: {path}")


def retrain(root: Path, index: int) -> Path:
    """Retrain one unique shared-front trial for 200 epochs and evaluate anomalies."""
    _load_design(root)
    manifest_path = root / "selection" / "pareto_retrain_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Freeze the Pareto fronts before retraining.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest["retrain_runs"]
    if index < 0 or index >= len(rows):
        raise IndexError(f"Retrain index {index} is outside [0,{len(rows)}).")
    row = rows[index]
    marker = root / "retraining" / "markers" / f"{index:04d}.json"
    if marker.is_file():
        _validate_retrain_marker(marker, row)
        return marker

    run_name = f"{row['model']}__trial{int(row['trial_number']):04d}"
    experiment_name = f"physics_bgpair_retrain_{row['model']}"
    hydra_dir = root / "retraining" / "hydra" / f"{index:04d}"
    command = [
        sys.executable,
        "src/train.py",
        f"experiment=physics/{row['model']}_background_all",
        f"+retrain_suite={_retrain_suite(row['model'])}",
        f"experiment_name={experiment_name}",
        f"run_name={run_name}",
        f"hydra.run.dir={hydra_dir}",
        "trainer=gpu",
        "trainer.devices=[0]",
        f"trainer.min_epochs={RETRAIN_EPOCHS}",
        f"trainer.max_epochs={RETRAIN_EPOCHS}",
        "trainer.num_sanity_val_steps=0",
        "test=true",
        f"seed={int(row['seed'])}",
    ]
    command.extend(f"{key}={_hydra_value(value)}" for key, value in row["params"].items())
    started_at = datetime.now(timezone.utc).isoformat()
    started_clock = time.monotonic()
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)  # nosec B603
    elapsed_seconds = time.monotonic() - started_clock

    optimized = hydra_dir / "optimized_metric.json"
    if not optimized.is_file():
        raise RuntimeError(f"Retraining did not emit {optimized}.")
    artifact = json.loads(optimized.read_text(encoding="utf-8"))
    if artifact.get("schema_version") != 2:
        raise RuntimeError("Retraining emitted a legacy optimized-metric artifact.")
    expected_objectives = [objective["id"] for objective in _logical_objectives(row["model"])]
    if artifact.get("objective_order") != expected_objectives:
        raise RuntimeError("Retraining emitted an unexpected optimized-objective order.")
    checkpoint_root = Path(os.environ.get("CHECKPOINT_DIR", REPOSITORY_ROOT / "checkpoints"))
    checkpoint_root = checkpoint_root / experiment_name / run_name
    value_files = sorted(checkpoint_root.glob("**/plots/test/**/values.csv"))
    expected_scores = SCORES_BY_MODEL[row["model"]]
    for score in expected_scores:
        if not any(f"/eff_{score}/" in path.as_posix() for path in value_files):
            raise RuntimeError(f"Retraining emitted no downstream values for {score}.")
    _write_json(
        marker,
        {
            "schema_version": 2,
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed_seconds,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "row": row,
            "command": command,
            "optimized_metric_artifact": str(optimized),
            "optimized_metric_artifact_sha256": _sha256(optimized),
            "checkpoint_root": str(checkpoint_root),
            "value_files": [{"path": str(path), "sha256": _sha256(path)} for path in value_files],
        },
        create=True,
    )
    return marker


def _checkpoint_identity(name: str) -> str:
    """Apply the evaluator's dataset-aware checkpoint-name normalization."""
    parts = name.split("ds=")
    return parts[1].split("__")[0] if len(parts) > 1 else parts[0]


def _read_downstream_efficiencies(
    marker: dict[str, Any], checkpoint: str, score: str
) -> dict[str, float]:
    """Read one score's 20 held-out efficiencies for the selected checkpoint."""
    efficiencies = {}
    score_folder = f"/eff_{score}/"
    for entry in marker["value_files"]:
        path = Path(entry["path"])
        if score_folder not in path.as_posix():
            continue
        if _sha256(path) != entry["sha256"]:
            raise RuntimeError(f"Efficiency artifact changed: {path}")
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if _checkpoint_identity(row["checkpoint"]) != checkpoint:
                    continue
                if row["metric"] != "efficiency_operational":
                    continue
                efficiencies[row["intervention"]] = float(row["value"])
    if set(efficiencies) != DOWNSTREAM_DATASET_NAMES:
        raise RuntimeError(
            f"Expected the exact {DOWNSTREAM_DATASETS}-dataset anomaly set for {score} "
            f"and {checkpoint}; found {sorted(efficiencies)}."
        )
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in efficiencies.values()):
        raise RuntimeError(f"Invalid downstream efficiency for {score} and {checkpoint}.")
    return efficiencies


def aggregate(root: Path) -> Path:
    """Select the downstream-oracle winner within every logical Pareto front."""
    design = _load_design(root)
    manifest_path = root / "selection" / "pareto_retrain_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    markers: dict[int, dict[str, Any]] = {}
    optimized_artifacts: dict[int, dict[str, Any]] = {}
    retrains_by_index: dict[int, dict[str, Any]] = {}
    for row in manifest["retrain_runs"]:
        index = int(row["retrain_index"])
        marker_path = root / "retraining" / "markers" / f"{index:04d}.json"
        if not marker_path.is_file():
            raise RuntimeError(f"Missing retraining marker {marker_path}.")
        _validate_retrain_marker(marker_path, row)
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        optimized_path = Path(marker["optimized_metric_artifact"])
        if _sha256(optimized_path) != marker["optimized_metric_artifact_sha256"]:
            raise RuntimeError(f"Optimized-metric artifact changed: {optimized_path}")
        optimized = json.loads(optimized_path.read_text(encoding="utf-8"))
        if optimized.get("schema_version") != 2:
            raise RuntimeError(f"Expected schema-2 optimized metrics: {optimized_path}")
        markers[index] = marker
        optimized_artifacts[index] = optimized
        retrains_by_index[index] = row

    results = []
    for membership in manifest["front_memberships"]:
        index = int(membership["retrain_index"])
        selection = optimized_artifacts[index]["selections"][membership["logical_objective"]]
        checkpoint = selection["optimized_ckpt_name"]
        efficiencies = _read_downstream_efficiencies(
            markers[index], checkpoint, membership["score"]
        )
        results.append(
            {
                **membership,
                "params": retrains_by_index[index]["params"],
                "optimized_ckpt_name": checkpoint,
                "retrained_objective_values": selection["optimized_metric"],
                "mean_downstream_efficiency": (sum(efficiencies.values()) / len(efficiencies)),
                "downstream_efficiencies": efficiencies,
            }
        )

    winners = []
    for study in design["studies"]:
        for objective in study["objectives"]:
            candidates = [
                result
                for result in results
                if result["model"] == study["model"]
                and result["logical_objective"] == objective["id"]
            ]
            if not candidates:
                raise RuntimeError(f"No candidates for {study['model']} {objective['id']}.")
            winners.append(max(candidates, key=lambda item: item["mean_downstream_efficiency"]))
    output = root / "results" / "oracle_winners.json"
    _write_json(
        output,
        {
            "schema_version": 2,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "oracle_uses_downstream_anomalies": True,
            "winners": winners,
            "all_front_memberships": results,
        },
    )
    return output


def _base_train_command(root: Path, model: str) -> list[str]:
    """Build invariant Hydra overrides for one shared model study."""
    study_name = f"physics_{model}_background_all"
    return [
        sys.executable,
        "src/train.py",
        "-m",
        f"experiment=physics/{model}_background_all",
        f"hparams_search={model}_shared_optuna",
        f"experiment_name=physics_bgpair_search_{model}",
        "trainer=gpu",
        "trainer.devices=[0]",
        f"trainer.min_epochs={SEARCH_EPOCHS}",
        f"trainer.max_epochs={SEARCH_EPOCHS}",
        "trainer.num_sanity_val_steps=0",
        "test=false",
        f"hydra.sweeper.storage='sqlite:///{_study_path(root, model)}?timeout=600'",
        f"hydra.sweeper.study_name={study_name}",
    ]


def sweep(root: Path, model: str, trials: int, jobs: int, timeout_min: int = 720) -> None:
    """Launch one resumable shared-study chunk inside a packed Slurm allocation."""
    design = _load_design(root)
    study = _study(design, model)
    if trials <= 0 or jobs <= 0 or timeout_min <= 0:
        raise ValueError("trials, jobs, and timeout-min must be positive.")
    counts = _trial_counts(root, study)
    if counts["running"] or counts["waiting"]:
        raise RuntimeError(
            f"Refusing to overlap {model} chunks: {counts['running']} trials are running "
            f"and {counts['waiting']} are waiting."
        )
    remaining = TARGET_TRIALS - counts["usable_complete"]
    if remaining <= 0:
        print(f"{model} already has {TARGET_TRIALS} usable completed trials.")
        return
    trials = min(int(trials), remaining)
    jobs = min(int(jobs), trials)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sweep_dir = root / "sweeps" / model / stamp
    database = _study_path(root, model)
    database.parent.mkdir(parents=True, exist_ok=True)
    prior_trial_numbers = {int(trial.number) for trial in _study_trials(root, study)}
    command = _base_train_command(root, model) + [
        "hydra/launcher=packed_srun_clariden",
        f"hydra.sweeper.n_trials={trials}",
        f"hydra.sweeper.n_jobs={jobs}",
        f"hydra.sweep.dir={sweep_dir}",
        "hydra.launcher.cpus_per_task=16",
        "hydra.launcher.mem_gb=96",
    ]
    chunk_path = sweep_dir / "chunk.json"
    chunk = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "model": model,
        "study_name": study["study_name"],
        "requested_trials": trials,
        "parallel_jobs": jobs,
        "timeout_min": int(timeout_min),
        "prior_trial_numbers": sorted(prior_trial_numbers),
        "command": command,
    }
    _write_json(chunk_path, chunk, create=True)
    failure = None
    started_clock = time.monotonic()
    try:
        subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)  # nosec B603
    except subprocess.CalledProcessError as error:
        failure = error
    finally:
        _write_chunk_bindings(root, study, sweep_dir, prior_trial_numbers)
        chunk.update(
            {
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.monotonic() - started_clock,
                "controller_status": "failed" if failure is not None else "complete",
            }
        )
        _write_json(chunk_path, chunk)
    if failure is not None:
        raise failure


def pilot(root: Path, model: str) -> None:
    """Run one one-epoch pilot of the model's complete selection suite."""
    design = _load_design(root)
    output = root / "pilots" / model
    command = [
        sys.executable,
        "src/train.py",
        f"experiment=physics/{model}_background_all",
        f"experiment_name=physics_bgpair_pilot_{model}",
        f"run_name={design['code_commit'][:12]}",
        f"hydra.run.dir={output}",
        "trainer=gpu",
        "trainer.devices=[0]",
        "trainer.min_epochs=1",
        "trainer.max_epochs=1",
        "trainer.num_sanity_val_steps=0",
        "+trainer.limit_train_batches=2",
        "physics_selection.cap_metric_config.n_epochs=1",
        "test=false",
    ]
    started_at = datetime.now(timezone.utc).isoformat()
    started_clock = time.monotonic()
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)  # nosec B603
    elapsed_seconds = time.monotonic() - started_clock
    optimized = output / "optimized_metric.json"
    if not optimized.is_file():
        raise RuntimeError(f"Pilot emitted no {optimized}.")
    artifact = json.loads(optimized.read_text(encoding="utf-8"))
    expected = [objective["id"] for objective in _logical_objectives(model)]
    if artifact.get("schema_version") != 2 or artifact.get("objective_order") != expected:
        raise RuntimeError("Pilot optimized-metric artifact has the wrong objective contract.")
    selections = artifact.get("selections", {})
    if set(selections) != set(expected) or any(
        selections[name].get("optimized_ckpt_name") is None
        or selections[name].get("optimized_metric") is None
        for name in expected
    ):
        raise RuntimeError("Pilot did not select a complete checkpoint/objective pair per view.")
    threshold_audit = _audit_selected_threshold_checkpoints(optimized, model)
    _write_json(
        output / "success.json",
        {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "started_at": started_at,
            "elapsed_seconds": elapsed_seconds,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "model": model,
            "logical_objectives": expected,
            "threshold_checkpoint_audit": threshold_audit,
            "code_commit": design["code_commit"],
            "command": command,
        },
    )


def packed_pilot(root: Path, model: str) -> None:
    """Exercise four real Optuna trials in one four-GPU debug allocation."""
    design = _load_design(root)
    study = _study(design, model)
    output = root / "packed_pilots" / model
    database = output / "study.db"
    output.mkdir(parents=True, exist_ok=True)
    study_name = f"physics_{model}_packed_launcher_pilot_{design['code_commit'][:12]}"
    command = [
        sys.executable,
        "src/train.py",
        "-m",
        f"experiment=physics/{model}_background_all",
        f"hparams_search={model}_shared_optuna",
        "hydra/launcher=packed_srun_clariden",
        f"experiment_name=physics_bgpair_packed_pilot_{model}",
        "trainer=gpu",
        "trainer.devices=[0]",
        "trainer.min_epochs=1",
        "trainer.max_epochs=1",
        "trainer.num_sanity_val_steps=0",
        "+trainer.limit_train_batches=2",
        "physics_selection.cap_metric_config.n_epochs=1",
        "test=false",
        f"hydra.sweeper.storage='sqlite:///{database}?timeout=600'",
        f"hydra.sweeper.study_name={study_name}",
        "hydra.sweeper.n_trials=4",
        "hydra.sweeper.n_jobs=4",
        f"hydra.sweep.dir={output}",
    ]
    started_at = datetime.now(timezone.utc).isoformat()
    started_clock = time.monotonic()
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)  # nosec B603
    elapsed_seconds = time.monotonic() - started_clock
    loaded = _optuna().load_study(study_name=study_name, storage=f"sqlite:///{database}")
    trials = list(loaded.get_trials(deepcopy=False))
    expected_values = 2 * len(study["objectives"])
    if len(trials) != 4 or any(
        trial.state != _optuna().trial.TrialState.COMPLETE
        or not _usable_trial(trial, expected_values=expected_values)
        for trial in trials
    ):
        raise RuntimeError("Packed pilot did not produce four finite completed trials.")
    pilot_study = {**study, "study_name": study_name}
    bindings = _write_chunk_bindings(
        root,
        pilot_study,
        output,
        set(),
        all_trials=trials,
    )
    binding_rows = json.loads(bindings.read_text(encoding="utf-8"))["bindings"]
    if [row["trial_number"] for row in binding_rows] != [0, 1, 2, 3] or any(
        row["hydra_job_num"] != row["trial_number"] or row["optimized_metric_artifact"] is None
        for row in binding_rows
    ):
        raise RuntimeError("Packed pilot trial-to-Hydra bindings are incomplete or reordered.")
    threshold_audit = [
        {
            "trial_number": int(row["trial_number"]),
            "checkpoints": _audit_selected_threshold_checkpoints(
                Path(row["optimized_metric_artifact"]), model
            ),
        }
        for row in binding_rows
    ]
    _write_json(
        output / "success.json",
        {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "started_at": started_at,
            "elapsed_seconds": elapsed_seconds,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "model": model,
            "trials": 4,
            "values_per_trial": expected_values,
            "threshold_checkpoint_audit": threshold_audit,
            "code_commit": design["code_commit"],
            "command": command,
        },
    )


def submit_next(root: Path, trials: int, jobs: int, timeout_min: int = 720) -> None:
    """Submit a controller for the first incomplete model study."""
    design = _load_design(root)
    incomplete = [
        study
        for study in design["studies"]
        if _trial_counts(root, study)["usable_complete"] < TARGET_TRIALS
    ]
    if not incomplete:
        print("All six shared studies are complete.")
        return
    sbatch = shutil.which("sbatch")
    if sbatch is None:
        raise FileNotFoundError("sbatch is required to submit campaign controllers.")
    study = incomplete[0]
    script = REPOSITORY_ROOT / "scripts/clariden/physics_background_pairing_sweep.sbatch"
    result = subprocess.run(  # nosec B603
        [
            sbatch,
            "--parsable",
            f"--nodes={math.ceil(min(trials, jobs) / 4)}",
            (
                "--export=ALL,"
                f"STUDY_MODEL={study['model']},CHUNK_TRIALS={trials},"
                f"CHUNK_JOBS={jobs},TIMEOUT_MIN={timeout_min}"
            ),
            str(script),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    print(json.dumps({"model": study["model"], "job_id": result.stdout.strip()}))


def main() -> None:
    """Dispatch campaign initialization, status, search, or selection."""
    args = parse_args()
    root = args.root.expanduser().resolve()
    if args.command == "init":
        print(initialize(root))
    elif args.command == "status":
        status(root)
    elif args.command == "refresh-bindings":
        refresh_bindings(root)
    elif args.command == "pilot":
        pilot(root, args.model)
    elif args.command == "packed-pilot":
        packed_pilot(root, args.model)
    elif args.command == "sweep":
        sweep(root, args.model, args.trials, args.jobs, args.timeout_min)
    elif args.command == "submit-next":
        submit_next(root, args.trials, args.jobs, args.timeout_min)
    elif args.command == "freeze":
        print(freeze(root))
    elif args.command == "retrain":
        print(retrain(root, args.index))
    elif args.command == "aggregate":
        print(aggregate(root))


if __name__ == "__main__":
    main()
