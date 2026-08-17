#!/usr/bin/env python3
"""Manage the six-study physics background-pairing campaign."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess  # nosec B404 -- executes frozen repository/Slurm commands
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

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
MAX_RETRAIN_ARRAY_TASKS = 1001
DESIGN_SCHEMA_VERSION = 2
SCREENING_SAMPLER = {
    "name": "nsgaii",
    "_target_": "optuna.samplers.NSGAIISampler",
    "seed": 123,
    "population_size": 48,
    "mutation_prob": None,
    "crossover_prob": 0.9,
    "swapping_prob": 0.5,
    "constraints_func": None,
}
SCREENING_SAMPLER_SEED_POLICY = {
    "name": "base_seed_plus_model_offset_plus_starting_trial",
    "formula": "base_seed + model_offsets[model] + starting_optuna_trial_number",
    "base_seed": 123,
    "model_offsets": {model: index * 100000 for index, model in enumerate(MODELS)},
    "population_size": 48,
    "generation_size": 48,
    "final_partial_population": 24,
    "target_trials": TARGET_TRIALS,
    "require_pristine_study": True,
    "canary_studies_are_separate": True,
}
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
    production_canary = subparsers.add_parser("production-canary")
    production_canary.add_argument("--model", choices=MODELS, required=True)
    production_canary.add_argument("--attempt", type=int, required=True)
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
    resolve_retrain = subparsers.add_parser("resolve-retrain-index")
    resolve_retrain.add_argument("--task-map", type=Path, required=True)
    resolve_retrain.add_argument("--task-map-sha256", required=True)
    resolve_retrain.add_argument("--local-task-id", type=int, required=True)
    retrain = subparsers.add_parser("retrain")
    retrain.add_argument("--index", type=int, required=True)
    retrain.add_argument("--task-map", type=Path)
    retrain.add_argument("--task-map-sha256")
    subparsers.add_parser("aggregate")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    """Hash one JSON-compatible value with deterministic serialization."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _inside(path: Path, root: Path, label: str, *, require_file: bool = False) -> Path:
    """Resolve a path and require confinement to an authenticated campaign subtree."""
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root.resolve(strict=True)):
        raise RuntimeError(f"{label} escapes its authenticated root: {path}")
    if require_file and not resolved.is_file():
        raise RuntimeError(f"{label} is not a file: {resolved}")
    return resolved


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


def _load_retrain_task_map(root: Path, task_map: Path, expected_sha256: str) -> dict[str, Any]:
    """Authenticate one content-addressed Slurm-local retrain task map."""
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise RuntimeError(f"Invalid retrain task-map SHA-256: {expected_sha256!r}.")
    expected_path = (root / "state" / "retrain_task_maps" / f"{expected_sha256}.json").resolve()
    task_map = task_map.expanduser().resolve()
    if task_map != expected_path or not task_map.is_file() or _sha256(task_map) != expected_sha256:
        raise RuntimeError(f"Retrain task map is missing, misplaced, or changed: {task_map}")
    payload = json.loads(task_map.read_text(encoding="utf-8"))
    design_path = root / "design.json"
    design = json.loads(design_path.read_text(encoding="utf-8"))
    if design.get("schema_version") != DESIGN_SCHEMA_VERSION:
        raise RuntimeError(f"Retrain task map references an invalid campaign: {design_path}")
    manifest_path = (root / "selection" / "pareto_retrain_manifest.json").resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    if (
        payload.get("schema_version") != 1
        or payload.get("code_commit") != design["code_commit"]
        or payload.get("campaign_root") != str(root.resolve())
        or payload.get("manifest") != str(manifest_path)
        or payload.get("manifest_sha256") != _sha256(manifest_path)
        or payload.get("batch_size") != len(entries)
        or not 1 <= len(entries) <= MAX_RETRAIN_ARRAY_TASKS
        or [entry.get("local_task_id") for entry in entries] != list(range(len(entries)))
    ):
        raise RuntimeError(f"Retrain task map does not match the frozen campaign: {task_map}")
    rows = manifest.get("retrain_runs", [])
    actual_indices = [entry.get("retrain_index") for entry in entries]
    if actual_indices != sorted(set(actual_indices)):
        raise RuntimeError(f"Retrain task map has non-exact actual indices: {task_map}")
    for entry in entries:
        index = int(entry["retrain_index"])
        if (
            not 0 <= index < len(rows)
            or rows[index].get("retrain_index") != index
            or entry.get("model") != rows[index].get("model")
            or entry.get("row_sha256") != _json_sha256(rows[index])
        ):
            raise RuntimeError(
                f"Retrain task-map row differs from the frozen manifest: {task_map}"
            )
    return payload


def resolve_retrain_task_index(
    root: Path, task_map: Path, task_map_sha256: str, local_task_id: int
) -> int:
    """Map one bounded Slurm-local task ID to its exact frozen retrain index."""
    payload = _load_retrain_task_map(root, task_map, task_map_sha256)
    entries = payload["entries"]
    if not 0 <= local_task_id < len(entries):
        raise RuntimeError(f"Local retrain task {local_task_id} is absent from its task map.")
    entry = entries[local_task_id]
    if int(entry["local_task_id"]) != local_task_id:
        raise RuntimeError(f"Local retrain task {local_task_id} is not exact in its task map.")
    return int(entry["retrain_index"])


def _retrain_scheduler_identity(
    root: Path,
    index: int,
    task_map: Path | None,
    task_map_sha256: str | None,
) -> dict[str, Any]:
    """Authenticate mapped Slurm identity, or a legacy direct-index invocation."""
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    map_environment = os.environ.get("PHYSICS_BGPAIR_RETRAIN_TASK_MAP")
    sha_environment = os.environ.get("PHYSICS_BGPAIR_RETRAIN_TASK_MAP_SHA256")
    index_environment = os.environ.get("PHYSICS_BGPAIR_RETRAIN_INDEX")
    local_environment = os.environ.get("PHYSICS_BGPAIR_RETRAIN_LOCAL_TASK_ID")
    if task_map is None and task_map_sha256 is None:
        if any(
            value is not None
            for value in (map_environment, sha_environment, index_environment, local_environment)
        ):
            raise RuntimeError("A retrain task-map environment requires explicit map arguments.")
        if task_id is not None and int(task_id) != index:
            raise RuntimeError(
                f"Legacy Slurm task {task_id} cannot execute retrain index {index}."
            )
        return {
            "mode": "legacy_direct" if task_id is not None else "manual",
            "local_task_id": None if task_id is None else int(task_id),
            "retrain_index": index,
        }
    if task_map is None or task_map_sha256 is None or task_id is None:
        raise RuntimeError("Mapped retraining requires a map, SHA-256, and Slurm array task.")
    task_map = task_map.expanduser().resolve()
    local_task_id = int(task_id)
    if (
        map_environment != str(task_map)
        or sha_environment != task_map_sha256
        or local_environment != str(local_task_id)
        or index_environment != str(index)
    ):
        raise RuntimeError("Mapped retrain arguments differ from their exported Slurm identity.")
    resolved = resolve_retrain_task_index(root, task_map, task_map_sha256, local_task_id)
    if resolved != index:
        raise RuntimeError(
            f"Mapped Slurm task {local_task_id} resolves to {resolved}, not retrain index {index}."
        )
    return {
        "mode": "mapped",
        "task_map": str(task_map),
        "task_map_sha256": task_map_sha256,
        "local_task_id": local_task_id,
        "retrain_index": index,
    }


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
            "hparams_search": f"{model}_shared_nsgaii",
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
        "schema_version": DESIGN_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": _git_commit(),
        "preflight": str(preflight),
        "preflight_sha256": _sha256(preflight),
        "target_trials_per_study": TARGET_TRIALS,
        "search_epochs": SEARCH_EPOCHS,
        "retrain_epochs": RETRAIN_EPOCHS,
        "shared_trial_pool": True,
        "screening_sampler": SCREENING_SAMPLER,
        "screening_sampler_seed_policy": SCREENING_SAMPLER_SEED_POLICY,
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
    if design.get("schema_version") != DESIGN_SCHEMA_VERSION:
        raise RuntimeError("Campaign design is not the six-study schema.")
    if design.get("screening_sampler") is not None:
        if design["screening_sampler"] != SCREENING_SAMPLER:
            raise RuntimeError("Campaign sampler differs from the frozen NSGA-II contract.")
        for study in design.get("studies", []):
            expected = f"{study.get('model')}_shared_nsgaii"
            if study.get("hparams_search") != expected:
                raise RuntimeError("Campaign HPO config differs from the NSGA-II contract.")
        if design.get("screening_sampler_seed_policy") != SCREENING_SAMPLER_SEED_POLICY:
            raise RuntimeError("Campaign sampler seed policy differs from the frozen contract.")
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
    sampler_invocation: dict[str, int] | None = None,
) -> Path:
    """Bind chunk-local Hydra jobs to persistent Optuna trial numbers."""
    hparams_search = study.get("hparams_search", f"{study['model']}_shared_optuna")
    sampler = SCREENING_SAMPLER if hparams_search.endswith("_shared_nsgaii") else None
    if sampler is not None and sampler_invocation is None:
        sampler_invocation = {
            "starting_optuna_trial_number": 0,
            "expected_generation": 0,
            "resolved_seed": SCREENING_SAMPLER["seed"],
        }
    resolved_sampler = (
        None if sampler is None else {**sampler, "seed": sampler_invocation["resolved_seed"]}
    )
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
    rows = []
    for local_index, trial in enumerate(new_trials):
        run_dir = run_by_job.get(local_index)
        optimized = None if run_dir is None else run_dir / "optimized_metric.json"
        has_optimized = optimized is not None and optimized.is_file()
        row = {
            "model": study["model"],
            "study_name": study["study_name"],
            "hparams_search": hparams_search,
            "trial_number": int(trial.number),
            "hydra_job_num": local_index,
            "hydra_run_dir": None if run_dir is None else str(run_dir.resolve()),
            "optimized_metric_artifact": (str(optimized.resolve()) if has_optimized else None),
            "optimized_metric_artifact_sha256": (_sha256(optimized) if has_optimized else None),
            "state": trial.state.name,
            "params": _jsonable(trial.params),
            "values": None if trial.values is None else _jsonable(trial.values),
        }
        if sampler is not None:
            generation = getattr(trial, "system_attrs", {}).get("nsga2:generation")
            if isinstance(generation, int) and generation >= 0:
                if generation != sampler_invocation["expected_generation"]:
                    raise RuntimeError(
                        f"NSGA-II trial {trial.number} belongs to generation {generation}, not "
                        f"{sampler_invocation['expected_generation']}."
                    )
                row["nsga2_trial_phase"] = "sampled"
                row["nsga2_generation"] = generation
            elif (
                trial.state.name == "FAIL"
                and not trial.params
                and trial.values is None
                and "nsga2:generation" not in getattr(trial, "system_attrs", {})
            ):
                row["nsga2_trial_phase"] = "pre_sampling_failure"
                row["nsga2_generation"] = None
                row["pre_sampling_failure_evidence"] = "optuna_terminal_state"
            else:
                raise RuntimeError(
                    f"NSGA-II trial {trial.number} has an unauthenticated sampling state."
                )
            row["screening_sampler"] = resolved_sampler
            row["resolved_sampler_seed"] = sampler_invocation["resolved_seed"]
        rows.append(row)
    output = sweep_dir / "trial_bindings.json"
    _write_json(
        output,
        {
            "schema_version": 2 if sampler is not None else 1,
            "model": study["model"],
            "study_name": study["study_name"],
            "hparams_search": hparams_search,
            "screening_sampler_base": sampler,
            "screening_sampler_base_sha256": (None if sampler is None else _json_sha256(sampler)),
            "screening_sampler": resolved_sampler,
            "screening_sampler_sha256": (
                None if resolved_sampler is None else _json_sha256(resolved_sampler)
            ),
            "screening_sampler_seed_policy": (
                None if sampler is None else SCREENING_SAMPLER_SEED_POLICY
            ),
            "screening_sampler_seed_policy_sha256": (
                None if sampler is None else _json_sha256(SCREENING_SAMPLER_SEED_POLICY)
            ),
            "sampler_invocation": sampler_invocation,
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
                sampler_invocation=chunk.get("sampler_invocation"),
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


RETRAIN_SEGMENT_EPOCHS = 40
EXACT_RETRAIN_RNG_STATE_KEY = "physics-background-pairing-exact-retrain-rng-v1"


def _retrain_paths(root: Path, index: int) -> dict[str, Path]:
    """Return isolated persistent paths for one resumable retrain task."""
    hydra_dir = root / "retraining" / "hydra" / f"{index:04d}"
    state_dir = root / "retraining" / "restart_state" / f"{index:04d}"
    return {
        "hydra": hydra_dir,
        "trainer": state_dir / "lightning",
        "state": state_dir / "state.json",
        "lock": state_dir / "state.lock",
        "segments": state_dir / "segments",
        "hpc_archive": state_dir / "hpc_archive",
    }


def _retrain_identity(
    root: Path,
    design: dict[str, Any],
    manifest_path: Path,
    index: int,
    row: dict[str, Any],
) -> dict[str, Any]:
    """Build the immutable identity shared by every allocation of one retrain."""
    # Historical Pareto manifests contain one seed per (model, trial) and retain
    # their original run name.  Confirmation manifests provide an explicit
    # content-authenticated run ID because the same trial is trained under
    # multiple seeds; omitting the seed would make those runs share checkpoints.
    run_name = str(row.get("run_id", f"{row['model']}__trial{int(row['trial_number']):04d}"))
    if not run_name or "/" in run_name or run_name in {".", ".."}:
        raise RuntimeError(f"Invalid frozen retrain run ID: {run_name!r}.")
    paths = _retrain_paths(root, index)
    tracking_dir = root / "runtime" / "logs" / "mlflow" / "mlruns"
    return {
        "code_commit": design["code_commit"],
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "retrain_index": index,
        "row": row,
        "row_sha256": _json_sha256(row),
        "experiment_name": f"physics_bgpair_retrain_{row['model']}",
        "run_name": run_name,
        "hydra_dir": str(paths["hydra"].resolve()),
        "trainer_default_root_dir": str(paths["trainer"].resolve()),
        "mlflow_tracking_uri": f"file:{tracking_dir.resolve()}",
    }


def _mlflow_run_for_retrain(identity: dict[str, Any], existing_run_id: str | None) -> str:
    """Create or authenticate the one MLflow run owned by a resumable retrain."""
    from mlflow.tracking import MlflowClient

    tracking_uri = str(identity["mlflow_tracking_uri"])
    tracking_dir = Path(tracking_uri.removeprefix("file:"))
    tracking_dir.mkdir(parents=True, exist_ok=True)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_name = str(identity["experiment_name"])
    experiment_lock = tracking_dir.parent / ".physics_bgpair_mlflow_experiments.lock"
    experiment_lock.parent.mkdir(parents=True, exist_ok=True)
    with experiment_lock.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = client.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
    authenticated_experiment = client.get_experiment(experiment_id)
    if authenticated_experiment is None or authenticated_experiment.name != experiment_name:
        raise RuntimeError("Retrain MLflow experiment identity changed after creation.")
    identity_sha = _json_sha256(identity)
    tag_name = "physics_bgpair_retrain_identity_sha256"
    matches = client.search_runs(
        [experiment_id],
        filter_string=f"tags.{tag_name} = '{identity_sha}'",
        max_results=2,
    )
    if len(matches) > 1:
        raise RuntimeError(f"Multiple MLflow runs claim retrain identity {identity_sha}.")
    discovered = None if not matches else matches[0].info.run_id
    if existing_run_id is not None and discovered != existing_run_id:
        raise RuntimeError("Persisted retrain MLflow run differs from tagged MLflow state.")
    if discovered is None:
        if existing_run_id is not None:
            raise RuntimeError("Persisted retrain MLflow run is missing from its experiment.")
        expected_tags = {
            tag_name: identity_sha,
            "mlflow.runName": str(identity["run_name"]),
            "mlflow.source.git.commit": str(identity["code_commit"]),
            "physics_bgpair_code_commit": str(identity["code_commit"]),
            "physics_bgpair_retrain_index": str(identity["retrain_index"]),
        }
        if "seed" in identity.get("row", {}):
            expected_tags["physics_bgpair_seed"] = str(identity["row"]["seed"])
        run = client.create_run(
            experiment_id,
            tags=expected_tags,
        )
        discovered = run.info.run_id
    else:
        expected_tags = {
            tag_name: identity_sha,
            "mlflow.runName": str(identity["run_name"]),
            "mlflow.source.git.commit": str(identity["code_commit"]),
            "physics_bgpair_code_commit": str(identity["code_commit"]),
            "physics_bgpair_retrain_index": str(identity["retrain_index"]),
        }
        if "seed" in identity.get("row", {}):
            expected_tags["physics_bgpair_seed"] = str(identity["row"]["seed"])
    run = client.get_run(discovered)
    if run.info.experiment_id != experiment_id:
        raise RuntimeError("Retrain MLflow run belongs to a different experiment.")
    if any(run.data.tags.get(key) != value for key, value in expected_tags.items()):
        raise RuntimeError("Retrain MLflow run tags differ from its immutable identity.")
    return str(discovered)


def _hpc_checkpoint_rows(trainer_root: Path) -> list[dict[str, Any]]:
    """Authenticate Lightning emergency-resume checkpoints in version order."""
    rows = []
    for path in sorted(trainer_root.glob("hpc_ckpt_*.ckpt")):
        rows.append({"path": str(path.resolve()), "sha256": _sha256(path)})
    return rows


def _validate_exact_rng_checkpoint_payload(payload: dict[str, Any], checkpoint: Path) -> None:
    """Require exact stochastic state at a true completed-epoch boundary."""
    state = payload.get("callbacks", {}).get(EXACT_RETRAIN_RNG_STATE_KEY)
    from src.callbacks.exact_retrain_rng import ExactRetrainRNGState

    try:
        ExactRetrainRNGState.validate_state(
            state,
            require_datamodule_shuffler=True,
            expected_cuda_states=1,
        )
    except RuntimeError as error:
        raise RuntimeError(f"Checkpoint lacks exact retrain RNG state: {checkpoint}") from error
    completed_epochs = int(payload.get("epoch", -1)) + 1
    try:
        progress = payload["loops"]["fit_loop"]["epoch_progress"]["total"]
        observed = [int(progress[name]) for name in ("ready", "started", "processed")]
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"Checkpoint lacks completed-epoch loop state: {checkpoint}") from error
    if completed_epochs <= 0 or observed != [completed_epochs] * 3:
        raise RuntimeError(
            f"Checkpoint is not a completed-epoch boundary: {checkpoint}; "
            f"completed_epochs={completed_epochs}, progress={observed}."
        )


def _validate_exact_rng_checkpoint(checkpoint: Path) -> None:
    """Load and authenticate exact stochastic continuation state."""
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)  # nosec B614
    _validate_exact_rng_checkpoint_payload(payload, checkpoint)


def _live_last_resume(identity: dict[str, Any], checkpoint_root: Path) -> dict[str, Any] | None:
    """Authenticate a live last checkpoint against exact contiguous MLflow epochs."""
    from mlflow.tracking import MlflowClient

    checkpoint = checkpoint_root / "last.ckpt"
    client = MlflowClient(tracking_uri=str(identity["mlflow_tracking_uri"]))
    history = sorted(
        client.get_metric_history(str(identity["mlflow_run_id"]), "epoch_idx"),
        key=lambda metric: int(metric.step),
    )
    if not checkpoint.is_file() and not history:
        return None
    if not checkpoint.is_file() or not history:
        raise RuntimeError("Live last checkpoint and MLflow epoch history are inconsistent.")
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)  # nosec B614
    _validate_exact_rng_checkpoint_payload(payload, checkpoint)
    checkpoint_epoch = int(payload.get("epoch", -1))
    steps = [int(metric.step) for metric in history]
    values = [float(metric.value) for metric in history]
    completed_epochs = checkpoint_epoch + 1
    if (
        completed_epochs <= 0
        or len(history) != completed_epochs
        or len(set(steps)) != len(steps)
        or any(later <= earlier for earlier, later in zip(steps, steps[1:]))
        or values != [float(epoch) for epoch in range(completed_epochs)]
    ):
        raise RuntimeError("Live last checkpoint differs from contiguous MLflow epoch history.")
    return {
        "kind": "live_last",
        "path": str(checkpoint.resolve()),
        "sha256": _sha256(checkpoint),
        "completed_epochs": completed_epochs,
    }


def _atomic_copy(source: Path, target: Path) -> None:
    """Durably publish one immutable segment artifact without a partial final path."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        shutil.copy2(source, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _adopt_exact_orphan_segment(
    root: Path,
    identity: dict[str, Any],
    state: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    """Adopt a crash-between-copy-and-state segment only when every source is exact."""
    completed = [int(row["epoch_boundary"]) for row in state["segments"]]
    boundary = (len(completed) + 1) * RETRAIN_SEGMENT_EPOCHS
    if boundary > RETRAIN_EPOCHS:
        return
    checkpoint = paths["segments"] / f"epoch_{boundary:03d}.ckpt"
    config = paths["segments"] / f"epoch_{boundary:03d}_config.yaml"
    if not checkpoint.exists() and not config.exists():
        return
    checkpoint_root = (
        root
        / "runtime"
        / "checkpoints"
        / str(identity["experiment_name"])
        / str(identity["run_name"])
    )
    live = _live_last_resume(
        {**identity, "mlflow_run_id": state["mlflow_run_id"]}, checkpoint_root
    )
    resolved = Path(str(identity["hydra_dir"])) / ".hydra" / "config.yaml"
    if live is None or int(live["completed_epochs"]) != boundary or not resolved.is_file():
        raise RuntimeError(f"Orphan retrain segment does not match live epoch {boundary} state.")
    live_checkpoint = Path(str(live["path"]))
    if checkpoint.exists() and (not checkpoint.is_file() or live["sha256"] != _sha256(checkpoint)):
        raise RuntimeError(f"Orphan checkpoint differs at epoch {boundary}.")
    if config.exists() and (not config.is_file() or _sha256(resolved) != _sha256(config)):
        raise RuntimeError(f"Orphan config differs at epoch {boundary}.")
    if not checkpoint.exists():
        _atomic_copy(live_checkpoint, checkpoint)
    if not config.exists():
        _atomic_copy(resolved, config)
    candidates = [
        attempt
        for attempt in state["attempts"]
        if attempt.get("phase") == "training"
        and int(attempt.get("target_epoch_boundary", -1)) == boundary
    ]
    if not candidates:
        raise RuntimeError(f"Orphan retrain segment has no owning attempt at epoch {boundary}.")
    attempt = candidates[-1]
    if attempt.get("completed_at") is not None:
        raise RuntimeError(f"Orphan retrain owner is already completed at epoch {boundary}.")
    adopted_at = datetime.now(timezone.utc).isoformat()
    attempt["completed_at"] = adopted_at
    attempt["orphan_adopted_at"] = adopted_at
    state["segments"].append(
        {
            "epoch_boundary": boundary,
            "attempt_number": int(attempt["attempt_number"]),
            "continuation_checkpoint": str(checkpoint.resolve()),
            "continuation_checkpoint_sha256": _sha256(checkpoint),
            "resolved_config": str(config.resolve()),
            "resolved_config_sha256": _sha256(config),
        }
    )


def _prepare_retrain_attempt(
    root: Path,
    identity: dict[str, Any],
    scheduler_identity: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Persist one fail-closed allocation attempt and its exact resume source."""
    index = int(identity["retrain_index"])
    paths = _retrain_paths(root, index)
    paths["state"].parent.mkdir(parents=True, exist_ok=True)
    with paths["lock"].open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        state = None
        if paths["state"].is_file():
            state = json.loads(paths["state"].read_text(encoding="utf-8"))
            if state.get("schema_version") != 1 or state.get("identity") != identity:
                raise RuntimeError(f"Retrain restart state differs: {paths['state']}")
        else:
            checkpoint_root = (
                root
                / "runtime"
                / "checkpoints"
                / str(identity["experiment_name"])
                / str(identity["run_name"])
            )
            if checkpoint_root.is_dir() and any(checkpoint_root.iterdir()):
                raise RuntimeError(
                    "Checkpoint artifacts exist without authenticated retrain restart state: "
                    f"{checkpoint_root}"
                )
            hydra_dir = Path(str(identity["hydra_dir"]))
            if hydra_dir.is_dir() and any(hydra_dir.iterdir()):
                raise RuntimeError(
                    "Hydra artifacts exist without authenticated retrain restart state: "
                    f"{hydra_dir}"
                )
        existing_run_id = None if state is None else state.get("mlflow_run_id")
        run_id = _mlflow_run_for_retrain(identity, existing_run_id)
        if state is None:
            state = {
                "schema_version": 1,
                "identity": identity,
                "mlflow_run_id": run_id,
                "attempts": [],
                "segments": [],
            }
        checkpoint_root = (
            root
            / "runtime"
            / "checkpoints"
            / str(identity["experiment_name"])
            / str(identity["run_name"])
        )
        _adopt_exact_orphan_segment(root, identity, state, paths)
        completed = [int(row["epoch_boundary"]) for row in state["segments"]]
        all_boundaries = list(
            range(RETRAIN_SEGMENT_EPOCHS, RETRAIN_EPOCHS + 1, RETRAIN_SEGMENT_EPOCHS)
        )
        expected_completed = all_boundaries[: len(completed)]
        if completed != expected_completed:
            raise RuntimeError(f"Retrain segment history is not contiguous: {completed}")
        if state.get("completed_at") is not None:
            if completed != all_boundaries:
                raise RuntimeError("Completed retrain state lacks all training boundaries.")
            phase = "finalization"
        else:
            phase = "evaluation" if completed == all_boundaries else "training"
        attempt_number = len(state["attempts"]) + 1
        # Lightning HPC checkpoints can be written between epochs. Restoring one can
        # replay or skip dataloader state, so retain it as scheduler/restart evidence
        # but remove it from the trainer directory before Trainer.fit can discover it.
        hpc_evidence = _archive_hpc_checkpoints(paths, attempt_number)
        if _hpc_checkpoint_rows(paths["trainer"]):
            raise RuntimeError("HPC checkpoints remain discoverable after archival.")
        resume = None
        live = None
        if phase == "training":
            live = _live_last_resume(
                {**identity, "mlflow_run_id": state["mlflow_run_id"]}, checkpoint_root
            )
        if phase == "training" and completed:
            segment = state["segments"][-1]
            checkpoint = _inside(
                Path(segment["continuation_checkpoint"]),
                paths["segments"],
                "continuation checkpoint",
                require_file=True,
            )
            if _sha256(checkpoint) != segment["continuation_checkpoint_sha256"]:
                raise RuntimeError(f"Continuation checkpoint changed: {checkpoint}")
            _validate_exact_rng_checkpoint(checkpoint)
            resume = {
                "kind": "completed_segment",
                "path": str(checkpoint),
                "sha256": segment["continuation_checkpoint_sha256"],
                "completed_epochs": completed[-1],
            }
        next_boundary = (
            RETRAIN_EPOCHS
            if phase in {"evaluation", "finalization"}
            else (len(completed) + 1) * RETRAIN_SEGMENT_EPOCHS
        )
        if live is not None:
            live_epochs = int(live["completed_epochs"])
            committed_epochs = 0 if not completed else completed[-1]
            if live_epochs < committed_epochs:
                raise RuntimeError("Live last checkpoint predates committed retrain progress.")
            if live_epochs > next_boundary:
                raise RuntimeError(
                    "Live last checkpoint exceeds the next retrain segment boundary."
                )
            # Prefer the live checkpoint only when it contains additional completed
            # epochs. At an exact committed boundary, the immutable segment copy is
            # the stronger recovery source.
            if live_epochs > committed_epochs:
                resume = live
        # With neither an authenticated live epoch nor committed boundary, retrying
        # from scratch is exact. Archived HPC evidence is deliberately never a fit
        # resume source.
        attempt = {
            "attempt_number": attempt_number,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "target_epoch_boundary": next_boundary,
            "phase": phase,
            "resume": resume,
            "hpc_evidence_archived_before_fit": hpc_evidence,
            "scheduler_identity": scheduler_identity,
        }
        task_id = attempt["slurm_array_task_id"]
        expected_task_id = (
            index if scheduler_identity is None else scheduler_identity.get("local_task_id")
        )
        if (
            task_id is not None
            and expected_task_id is not None
            and int(task_id) != expected_task_id
        ):
            raise RuntimeError(
                f"Slurm task {task_id} differs from authenticated task {expected_task_id}."
            )
        state["attempts"].append(attempt)
        _write_json(paths["state"], state)
        return state, attempt


def _archive_hpc_checkpoints(paths: dict[str, Path], attempt_number: int) -> list[dict[str, Any]]:
    """Remove HPC files from Lightning discovery and retain content-addressed evidence."""
    if attempt_number <= 0:
        raise ValueError("Retrain attempt numbers are one-based.")
    rows = _hpc_checkpoint_rows(paths["trainer"])
    archive_root = paths["hpc_archive"]
    archive_root.mkdir(parents=True, exist_ok=True)
    for source_row in rows:
        source = Path(source_row["path"])
        digest = str(source_row["sha256"])
        target = archive_root / (
            f"attempt_{attempt_number:03d}__{source.stem}__{digest[:16]}{source.suffix}"
        )
        if target.exists():
            if not target.is_file() or _sha256(target) != digest:
                raise RuntimeError(f"HPC archive target differs: {target}")
            raise RuntimeError(f"HPC checkpoint and its immutable archive both exist: {source}")
        source.replace(target)
    prefix = f"attempt_{attempt_number:03d}__"
    return [
        {"path": str(path.resolve()), "sha256": _sha256(path)}
        for path in sorted(archive_root.glob(f"{prefix}*.ckpt"))
    ]


def _validate_retrain_progress(
    root: Path,
    index: int,
    expected_row: dict[str, Any],
    scheduler_job_id: str | None = None,
) -> Path:
    """Authenticate one successfully committed training segment without a final marker."""
    paths = _retrain_paths(root, index)
    state_path = _inside(
        paths["state"], paths["state"].parent, "retrain restart state", require_file=True
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    design_path = root / "design.json"
    design = json.loads(design_path.read_text(encoding="utf-8"))
    manifest_path = root / "selection" / "pareto_retrain_manifest.json"
    identity = _retrain_identity(root, design, manifest_path, index, expected_row)
    boundaries = [int(segment["epoch_boundary"]) for segment in state.get("segments", [])]
    expected_boundaries = list(
        range(RETRAIN_SEGMENT_EPOCHS, RETRAIN_EPOCHS + 1, RETRAIN_SEGMENT_EPOCHS)
    )[: len(boundaries)]
    attempts = state.get("attempts", [])
    if (
        state.get("schema_version") != 1
        or state.get("identity") != identity
        or state.get("completed_at") is not None
        or not boundaries
        or boundaries != expected_boundaries
        or not attempts
    ):
        raise RuntimeError(f"Retrain segment progress is not exact: {state_path}")
    latest = attempts[-1]
    if (
        latest.get("phase") != "training"
        or latest.get("completed_at") is None
        or int(latest.get("target_epoch_boundary", -1)) != boundaries[-1]
        or int(state["segments"][-1].get("attempt_number", -1))
        != int(latest.get("attempt_number", -2))
    ):
        raise RuntimeError(f"Latest retrain segment attempt is incomplete: {state_path}")
    if scheduler_job_id is not None:
        array_job_id = latest.get("slurm_array_job_id")
        array_task_id = latest.get("slurm_array_task_id")
        observed_job_id = (
            None
            if array_job_id is None or array_task_id is None
            else f"{array_job_id}_{array_task_id}"
        )
        if observed_job_id != scheduler_job_id:
            raise RuntimeError(
                f"Committed retrain segment belongs to {observed_job_id}, not {scheduler_job_id}."
            )
    for attempt in attempts:
        scheduler_identity = attempt.get("scheduler_identity")
        if scheduler_identity is None or scheduler_identity.get("mode") != "mapped":
            continue
        resolved = resolve_retrain_task_index(
            root,
            Path(scheduler_identity["task_map"]),
            str(scheduler_identity["task_map_sha256"]),
            int(scheduler_identity["local_task_id"]),
        )
        if resolved != index or scheduler_identity.get("retrain_index") != index:
            raise RuntimeError(f"Retrain progress has an invalid scheduler map: {state_path}")
    for segment in state["segments"]:
        boundary = int(segment["epoch_boundary"])
        checkpoint = _inside(
            Path(segment["continuation_checkpoint"]),
            paths["segments"],
            "continuation checkpoint",
            require_file=True,
        )
        config_path = _inside(
            Path(segment["resolved_config"]),
            paths["segments"],
            "segment resolved config",
            require_file=True,
        )
        if _sha256(checkpoint) != segment.get("continuation_checkpoint_sha256") or _sha256(
            config_path
        ) != segment.get("resolved_config_sha256"):
            raise RuntimeError(f"Retrain segment artifact changed: {state_path}")
        _validate_exact_rng_checkpoint(checkpoint)
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if (
            int(_config_value(config, "trainer.min_epochs")) != boundary
            or int(_config_value(config, "trainer.max_epochs")) != boundary
            or _config_value(config, "train") is not True
            or _config_value(config, "test") is not False
            or _config_value(config, "skip_evaluation") is not True
            or int(_config_value(config, "seed")) != int(expected_row["seed"])
        ):
            raise RuntimeError(f"Retrain segment config differs at epoch {boundary}.")
        for key, value in expected_row["params"].items():
            if _config_value(config, key) != value:
                raise RuntimeError(f"Retrain segment parameter differs for {key}.")
    return state_path


def _validate_retrain_marker(marker: Path, expected_row: dict[str, Any]) -> None:
    """Fail closed when resuming an existing retraining marker."""
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2 or payload.get("row") != expected_row:
        raise RuntimeError(f"Retraining marker does not match its frozen row: {marker}")
    index = int(expected_row["retrain_index"])
    root = marker.parents[2]
    if marker.name != f"{index:04d}.json":
        raise RuntimeError(f"Retraining marker path differs from index {index}: {marker}")
    optimized = Path(payload["optimized_metric_artifact"])
    if _sha256(optimized) != payload["optimized_metric_artifact_sha256"]:
        raise RuntimeError(f"Retraining optimized artifact changed: {optimized}")
    for entry in payload["value_files"]:
        path = Path(entry["path"])
        if _sha256(path) != entry["sha256"]:
            raise RuntimeError(f"Retraining value artifact changed: {path}")
    state_path = _inside(
        Path(payload["restart_state_artifact"]),
        root / "retraining" / "restart_state" / f"{index:04d}",
        "retrain restart state",
        require_file=True,
    )
    if _sha256(state_path) != payload.get("restart_state_artifact_sha256"):
        raise RuntimeError(f"Retraining restart state changed: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    manifest_path = _inside(
        Path(str(state.get("identity", {}).get("manifest", ""))),
        root / "selection",
        "frozen retrain manifest",
        require_file=True,
    )
    if manifest_path != (root / "selection" / "pareto_retrain_manifest.json").resolve():
        raise RuntimeError(f"Retrain restart state names the wrong manifest: {manifest_path}")
    if (
        state.get("schema_version") != 1
        or state.get("identity", {}).get("row") != expected_row
        or _sha256(manifest_path) != state.get("identity", {}).get("manifest_sha256")
        or state.get("mlflow_run_id") != payload.get("mlflow_run_id")
        or state.get("completed_at") != payload.get("completed_at")
    ):
        raise RuntimeError(f"Retraining restart evidence differs: {state_path}")
    scheduler_identity = payload.get("scheduler_identity")
    if scheduler_identity is not None:
        attempts = state.get("attempts", [])
        if not attempts or scheduler_identity != attempts[-1].get("scheduler_identity"):
            raise RuntimeError(f"Retraining scheduler identity differs: {state_path}")
        for attempt in attempts:
            attempt_identity = attempt.get("scheduler_identity")
            if attempt_identity is None or attempt_identity.get("mode") != "mapped":
                continue
            resolved = resolve_retrain_task_index(
                root,
                Path(attempt_identity["task_map"]),
                str(attempt_identity["task_map_sha256"]),
                int(attempt_identity["local_task_id"]),
            )
            if resolved != index or attempt_identity.get("retrain_index") != index:
                raise RuntimeError(f"Retraining scheduler map resolves incorrectly: {state_path}")
    boundaries = [int(row["epoch_boundary"]) for row in state.get("segments", [])]
    if boundaries != list(
        range(RETRAIN_SEGMENT_EPOCHS, RETRAIN_EPOCHS + 1, RETRAIN_SEGMENT_EPOCHS)
    ):
        raise RuntimeError(f"Retraining segment boundaries are incomplete: {boundaries}")
    for segment in state["segments"]:
        for key in ("continuation_checkpoint", "resolved_config"):
            path = _inside(
                Path(segment[key]), state_path.parent, f"retrain segment {key}", require_file=True
            )
            if _sha256(path) != segment[f"{key}_sha256"]:
                raise RuntimeError(f"Retraining segment evidence changed: {path}")
        _validate_exact_rng_checkpoint(Path(segment["continuation_checkpoint"]))
    audit = _audit_retrain_outputs(root, index, expected_row, payload)
    if "run_id" in expected_row and (
        payload.get("downstream_efficiency_counts") != audit["downstream_efficiency_counts"]
        or payload.get("downstream_auprc_counts") != audit["downstream_auprc_counts"]
    ):
        raise RuntimeError("Confirmation endpoint counts differ from the sealed marker.")


def _config_value(config: dict[str, Any], dotted_key: str) -> Any:
    """Read one exact dotted Optuna parameter from a resolved Hydra config."""
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise RuntimeError(f"Resolved retrain config lacks {dotted_key}.")
        value = value[part]
    return value


def _audit_retrain_outputs(
    root: Path,
    index: int,
    row: dict[str, Any],
    marker: dict[str, Any],
) -> dict[str, Any]:
    """Authenticate the exact 200-epoch config, selections, thresholds, and anomalies."""
    config_path = _inside(
        root / "retraining" / "hydra" / f"{index:04d}" / ".hydra" / "config.yaml",
        root,
        "resolved retrain config",
        require_file=True,
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise RuntimeError(f"Resolved retrain config is not a mapping: {config_path}")
    trainer = config.get("trainer", {})
    if trainer.get("min_epochs") != RETRAIN_EPOCHS or trainer.get("max_epochs") != RETRAIN_EPOCHS:
        raise RuntimeError("Final retrain config is not exactly 200 epochs.")
    if (
        config.get("train") is not False
        or config.get("test") is not True
        or config.get("skip_evaluation") is not False
    ):
        raise RuntimeError("Final retrain phase is not evaluation-only after epoch 200.")
    if int(config.get("seed", -1)) != int(row["seed"]):
        raise RuntimeError("Final retrain seed differs from its frozen row.")
    if int(row.get("epochs", -1)) != RETRAIN_EPOCHS or int(row["retrain_index"]) != index:
        raise RuntimeError("Frozen retrain epoch/index contract differs.")
    for key, expected in row.get("params", {}).items():
        if _config_value(config, str(key)) != expected:
            raise RuntimeError(f"Final retrain parameter differs for {key}.")
    expected_trainer_root = _retrain_paths(root, index)["trainer"].resolve()
    if Path(str(trainer.get("default_root_dir"))).resolve() != expected_trainer_root:
        raise RuntimeError("Lightning default_root_dir is not isolated to this retrain index.")

    optimized = _inside(
        Path(marker["optimized_metric_artifact"]),
        root / "retraining" / "hydra" / f"{index:04d}",
        "optimized metric artifact",
        require_file=True,
    )
    artifact = json.loads(optimized.read_text(encoding="utf-8"))
    expected_objectives = [objective["id"] for objective in _logical_objectives(row["model"])]
    if (
        artifact.get("schema_version") != 2
        or artifact.get("objective_order") != expected_objectives
        or set(artifact.get("selections", {})) != set(expected_objectives)
    ):
        raise RuntimeError("Retraining emitted an unexpected objective/selection contract.")
    provenance = artifact.get("run_provenance", {})
    if (
        provenance.get("experiment_name") != marker["experiment_name"]
        or provenance.get("run_name") != marker["run_name"]
        or Path(str(provenance.get("hydra_output_dir"))).resolve() != optimized.parent.resolve()
    ):
        raise RuntimeError("Retraining optimized artifact has inconsistent run provenance.")
    logger_ids = {
        str(logger.get("run_id"))
        for logger in provenance.get("loggers", [])
        if logger.get("run_id") is not None
    }
    if logger_ids != {str(marker["mlflow_run_id"])}:
        raise RuntimeError("Retraining did not preserve its single MLflow run identity.")

    checkpoint_root = _inside(
        Path(marker["checkpoint_root"]),
        root / "runtime" / "checkpoints",
        "retrain checkpoint root",
    )
    expected_checkpoint_root = (
        root / "runtime" / "checkpoints" / marker["experiment_name"] / marker["run_name"]
    ).resolve()
    if checkpoint_root != expected_checkpoint_root:
        raise RuntimeError("Retraining checkpoint root differs from its deterministic identity.")
    if Path(str(provenance.get("checkpoint_run_dir"))).resolve() != checkpoint_root:
        raise RuntimeError("Retraining artifact checkpoint root differs from its marker.")
    threshold_rows = _audit_selected_threshold_checkpoints(optimized, row["model"])
    if threshold_rows != marker.get("threshold_checkpoint_audit"):
        raise RuntimeError("Retraining checkpoint threshold audit differs from its marker.")
    value_files = marker.get("value_files", [])
    if not value_files:
        raise RuntimeError("Retraining emitted no downstream value files.")
    for entry in value_files:
        _inside(
            Path(entry["path"]), checkpoint_root, "downstream efficiency file", require_file=True
        )
    efficiency_counts = {}
    auprc_counts = {}
    for objective_id, selection in artifact["selections"].items():
        checkpoint = str(selection.get("optimized_ckpt_name", ""))
        values = selection.get("optimized_metric")
        if (
            not checkpoint
            or not isinstance(values, list)
            or len(values) != 2
            or not all(math.isfinite(float(value)) for value in values)
        ):
            raise RuntimeError(f"Invalid retrained selection for {objective_id}.")
        score = objective_id.split("__", maxsplit=1)[0]
        checkpoint_path = _selected_checkpoint_path(checkpoint_root, checkpoint)
        efficiencies = _read_downstream_efficiencies(marker, checkpoint_path, score)
        auprcs = _read_downstream_auprcs(marker, checkpoint_path, score)
        efficiency_counts[f"{objective_id}:{checkpoint}"] = len(efficiencies)
        auprc_counts[f"{objective_id}:{checkpoint}"] = len(auprcs)
    return {
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "threshold_checkpoint_audit": threshold_rows,
        "downstream_efficiency_counts": efficiency_counts,
        "downstream_auprc_counts": auprc_counts,
    }


def _retrain_hyperparameter_logging_override(attempt_number: int) -> str:
    """Log immutable retrain parameters once, before cumulative boundaries change."""
    if attempt_number <= 0:
        raise ValueError("Retrain attempt numbers are one-based.")
    return f"log_hyperparameters={str(attempt_number == 1).lower()}"


def retrain(
    root: Path,
    index: int,
    task_map: Path | None = None,
    task_map_sha256: str | None = None,
) -> Path:
    """Retrain one front trial in exact resumable segments, then evaluate once at epoch 200."""
    design = _load_design(root)
    scheduler_identity = _retrain_scheduler_identity(root, index, task_map, task_map_sha256)
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

    identity = _retrain_identity(root, design, manifest_path, index, row)
    state, attempt = _prepare_retrain_attempt(root, identity, scheduler_identity)
    paths = _retrain_paths(root, index)
    run_name = str(identity["run_name"])
    experiment_name = str(identity["experiment_name"])
    hydra_dir = paths["hydra"]
    boundary = int(attempt["target_epoch_boundary"])
    phase = str(attempt["phase"])
    training_segment = phase == "training"
    finalization_only = phase == "finalization"
    if phase not in {"training", "evaluation", "finalization"}:
        raise RuntimeError(f"Unknown retrain attempt phase: {phase}")
    command = []
    if not finalization_only:
        command = [
            sys.executable,
            "src/train.py",
            f"experiment=physics/{row['model']}_background_all",
            f"+retrain_suite={_retrain_suite(row['model'])}",
            f"experiment_name={experiment_name}",
            f"run_name={run_name}",
            f"hydra.run.dir={hydra_dir}",
            "trainer=gpu",
            f"trainer.default_root_dir={paths['trainer']}",
            "trainer.devices=[0]",
            f"trainer.min_epochs={boundary}",
            f"trainer.max_epochs={boundary}",
            "trainer.num_sanity_val_steps=0",
            f"train={str(training_segment).lower()}",
            f"test={str(not training_segment).lower()}",
            f"skip_evaluation={str(training_segment).lower()}",
            _retrain_hyperparameter_logging_override(int(attempt["attempt_number"])),
            f"seed={int(row['seed'])}",
            f"+logger.mlflow.run_id={state['mlflow_run_id']}",
        ]
        resume = attempt.get("resume")
        if training_segment and resume is not None:
            if resume.get("kind") not in {"live_last", "completed_segment"}:
                raise RuntimeError(f"Unsupported exact retrain resume source: {resume!r}")
            command.append(f"ckpt_path={resume['path']}")
        command.extend(f"{key}={_hydra_value(value)}" for key, value in row["params"].items())
    started_at = datetime.now(timezone.utc).isoformat()
    started_clock = time.monotonic()
    if not finalization_only:
        previous_sigusr1 = signal.getsignal(signal.SIGUSR1)
        signal.signal(signal.SIGUSR1, lambda _signum, _frame: None)
        try:
            subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)  # nosec B603
        finally:
            signal.signal(signal.SIGUSR1, previous_sigusr1)
    elapsed_seconds = time.monotonic() - started_clock

    checkpoint_root = Path(os.environ.get("CHECKPOINT_DIR", REPOSITORY_ROOT / "checkpoints"))
    checkpoint_root = checkpoint_root / experiment_name / run_name
    hpc_evidence = _archive_hpc_checkpoints(paths, int(attempt["attempt_number"]))
    continuation = None
    config_evidence = None
    if training_segment:
        last_checkpoint = _inside(
            checkpoint_root / "last.ckpt",
            root / "runtime" / "checkpoints",
            "retrain segment checkpoint",
            require_file=True,
        )
        paths["segments"].mkdir(parents=True, exist_ok=True)
        continuation = paths["segments"] / f"epoch_{boundary:03d}.ckpt"
        if continuation.exists():
            raise RuntimeError(f"Retrain continuation checkpoint already exists: {continuation}")
        _atomic_copy(last_checkpoint, continuation)
        _validate_exact_rng_checkpoint(continuation)
        resolved_config = _inside(
            hydra_dir / ".hydra" / "config.yaml",
            root,
            "resolved retrain config",
            require_file=True,
        )
        config_evidence = paths["segments"] / f"epoch_{boundary:03d}_config.yaml"
        if config_evidence.exists():
            raise RuntimeError(f"Retrain segment config already exists: {config_evidence}")
        _atomic_copy(resolved_config, config_evidence)
    if training_segment:
        with paths["lock"].open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            state = json.loads(paths["state"].read_text(encoding="utf-8"))
            active = state["attempts"][-1]
            if active != attempt:
                raise RuntimeError("Retrain attempt state changed while training.")
            active.update(
                {
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": elapsed_seconds,
                    "hpc_evidence": hpc_evidence,
                }
            )
            assert continuation is not None and config_evidence is not None
            state["segments"].append(
                {
                    "epoch_boundary": boundary,
                    "attempt_number": int(attempt["attempt_number"]),
                    "continuation_checkpoint": str(continuation.resolve()),
                    "continuation_checkpoint_sha256": _sha256(continuation),
                    "resolved_config": str(config_evidence.resolve()),
                    "resolved_config_sha256": _sha256(config_evidence),
                }
            )
            _write_json(paths["state"], state)
        return paths["state"]

    completion_time = str(state.get("completed_at") or datetime.now(timezone.utc).isoformat())

    optimized = hydra_dir / "optimized_metric.json"
    if not optimized.is_file():
        raise RuntimeError(f"Retraining did not emit {optimized}.")
    artifact = json.loads(optimized.read_text(encoding="utf-8"))
    if artifact.get("schema_version") != 2:
        raise RuntimeError("Retraining emitted a legacy optimized-metric artifact.")
    value_files = sorted(checkpoint_root.glob("**/plots/test/**/values.csv"))
    expected_scores = SCORES_BY_MODEL[row["model"]]
    for score in expected_scores:
        if not any(f"/eff_{score}/" in path.as_posix() for path in value_files):
            raise RuntimeError(f"Retraining emitted no downstream values for {score}.")
        if not any(f"/auprc_{score}/" in path.as_posix() for path in value_files):
            raise RuntimeError(f"Retraining emitted no downstream AUPRC values for {score}.")
    threshold_audit = _audit_selected_threshold_checkpoints(optimized, row["model"])
    payload = {
        "schema_version": 2,
        "started_at": started_at,
        "completed_at": completion_time,
        "elapsed_seconds": elapsed_seconds,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "scheduler_identity": scheduler_identity,
        "retrain_index": index,
        "row": row,
        "command": command,
        "experiment_name": experiment_name,
        "run_name": run_name,
        "mlflow_run_id": state["mlflow_run_id"],
        "restart_count": len(state["attempts"]) - 1,
        "restart_state_artifact": str(paths["state"].resolve()),
        "restart_state_artifact_sha256": _sha256(paths["state"]),
        "optimized_metric_artifact": str(optimized),
        "optimized_metric_artifact_sha256": _sha256(optimized),
        "checkpoint_root": str(checkpoint_root),
        "value_files": [{"path": str(path), "sha256": _sha256(path)} for path in value_files],
        "threshold_checkpoint_audit": threshold_audit,
    }
    audit = _audit_retrain_outputs(root, index, row, payload)

    with paths["lock"].open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        state = json.loads(paths["state"].read_text(encoding="utf-8"))
        active = state["attempts"][-1]
        if active != attempt:
            raise RuntimeError("Retrain attempt state changed while training.")
        active.update(
            {
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": elapsed_seconds,
                "hpc_evidence": hpc_evidence,
            }
        )
        if state.get("completed_at") is None:
            state["completed_at"] = datetime.now(timezone.utc).isoformat()
            completion_time = state["completed_at"]
        _write_json(paths["state"], state)
    payload["completed_at"] = completion_time
    payload["restart_count"] = len(state["attempts"]) - 1
    payload["restart_state_artifact_sha256"] = _sha256(paths["state"])
    payload["resolved_config"] = audit["config"]
    payload["resolved_config_sha256"] = audit["config_sha256"]
    payload["downstream_efficiency_counts"] = audit["downstream_efficiency_counts"]
    payload["downstream_auprc_counts"] = audit["downstream_auprc_counts"]
    _write_json(marker, payload, create=True)
    _validate_retrain_marker(marker, row)
    return marker


def _checkpoint_identity(name: str) -> str:
    """Apply the evaluator's dataset-aware checkpoint-name normalization."""
    parts = name.split("ds=")
    return parts[1].split("__")[0] if len(parts) > 1 else parts[0]


def _read_downstream_values(
    marker: dict[str, Any],
    checkpoint: Path,
    score: str,
    *,
    callback_prefix: str,
    metric: str,
) -> dict[str, float]:
    """Read one exact callback file bound to one physical checkpoint path."""
    checkpoint = checkpoint.resolve(strict=True)
    folder = f"{callback_prefix}_{score}"
    expected_path = (
        checkpoint.parent / "plots" / "test" / checkpoint.stem / folder / "values.csv"
    ).resolve()
    entries = [
        entry for entry in marker["value_files"] if Path(entry["path"]).resolve() == expected_path
    ]
    if len(entries) != 1:
        raise RuntimeError(
            f"Expected one {folder} artifact for checkpoint {checkpoint}; found {len(entries)}."
        )
    entry = entries[0]
    if not expected_path.is_file() or _sha256(expected_path) != entry["sha256"]:
        raise RuntimeError(f"Endpoint artifact changed: {expected_path}")
    values = {}
    with expected_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["checkpoint"] != checkpoint.stem:
                raise RuntimeError(
                    f"Endpoint row checkpoint differs from {checkpoint.stem}: {expected_path}"
                )
            if row["metric"] != metric:
                continue
            if row["intervention"] in values:
                raise RuntimeError(
                    f"Duplicate {metric} for {score}, {checkpoint}, and " f"{row['intervention']}."
                )
            values[row["intervention"]] = float(row["value"])
    if set(values) != DOWNSTREAM_DATASET_NAMES:
        raise RuntimeError(
            f"Expected the exact {DOWNSTREAM_DATASETS}-dataset {metric} set for {score} "
            f"and {checkpoint}; found {sorted(values)}."
        )
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values.values()):
        raise RuntimeError(f"Invalid downstream {metric} for {score} and {checkpoint}.")
    return values


def _read_downstream_efficiencies(
    marker: dict[str, Any], checkpoint: Path, score: str
) -> dict[str, float]:
    """Read one score's 20 held-out efficiencies for the selected checkpoint."""
    return _read_downstream_values(
        marker,
        checkpoint,
        score,
        callback_prefix="eff",
        metric="efficiency_operational",
    )


def _read_downstream_auprcs(
    marker: dict[str, Any], checkpoint: Path, score: str
) -> dict[str, float]:
    """Read one score's exact 20-anomaly AUPRC panel for a selected checkpoint."""
    return _read_downstream_values(
        marker, checkpoint, score, callback_prefix="auprc", metric="auprc"
    )


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
        checkpoint_path = _selected_checkpoint_path(
            Path(markers[index]["checkpoint_root"]), checkpoint
        )
        efficiencies = _read_downstream_efficiencies(
            markers[index], checkpoint_path, membership["score"]
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
            winners.append(_oracle_winner(candidates))
    expected_fronts = {
        (study["model"], objective["id"])
        for study in design["studies"]
        for objective in study["objectives"]
    }
    observed_fronts = {(row["model"], row["logical_objective"]) for row in winners}
    if len(winners) != 56 or observed_fronts != expected_fronts:
        raise RuntimeError(
            f"Oracle aggregation must produce exactly the 56 frozen fronts; found {len(winners)}."
        )
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


def _oracle_winner(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose maximum mean efficiency with a deterministic frozen-order tie break."""
    if not candidates:
        raise RuntimeError("Cannot choose an oracle winner from an empty front.")
    return max(
        candidates,
        key=lambda item: (
            item["mean_downstream_efficiency"],
            -int(item["membership_index"]),
        ),
    )


def _base_train_command(
    root: Path,
    model: str,
    study: dict[str, Any] | None = None,
    *,
    database: Path | None = None,
    study_name: str | None = None,
) -> list[str]:
    """Build invariant Hydra overrides for one shared model study."""
    if study is None:
        study = _study({"studies": _studies()}, model)
    study_name = study["study_name"] if study_name is None else study_name
    database = _study_path(root, model) if database is None else database
    hparams_search = study.get("hparams_search", f"{model}_shared_optuna")
    return [
        sys.executable,
        "src/train.py",
        "-m",
        f"experiment=physics/{model}_background_all",
        f"hparams_search={hparams_search}",
        f"experiment_name=physics_bgpair_search_{model}",
        "trainer=gpu",
        "trainer.devices=[0]",
        f"trainer.min_epochs={SEARCH_EPOCHS}",
        f"trainer.max_epochs={SEARCH_EPOCHS}",
        "trainer.num_sanity_val_steps=0",
        "test=false",
        f"hydra.sweeper.storage='sqlite:///{database}?timeout=600'",
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
    generation = counts["usable_complete"] // SCREENING_SAMPLER_SEED_POLICY["generation_size"]
    completed_in_generation = (
        counts["usable_complete"] % SCREENING_SAMPLER_SEED_POLICY["generation_size"]
    )
    generation_capacity = min(
        SCREENING_SAMPLER_SEED_POLICY["generation_size"],
        TARGET_TRIALS - generation * SCREENING_SAMPLER_SEED_POLICY["generation_size"],
    )
    generation_remaining = generation_capacity - completed_in_generation
    trials = min(int(trials), remaining, generation_remaining)
    jobs = min(int(jobs), trials)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sweep_dir = root / "sweeps" / model / stamp
    database = _study_path(root, model)
    database.parent.mkdir(parents=True, exist_ok=True)
    prior_trial_numbers = {int(trial.number) for trial in _study_trials(root, study)}
    starting_trial_number = max(prior_trial_numbers, default=-1) + 1
    resolved_seed = (
        SCREENING_SAMPLER_SEED_POLICY["base_seed"]
        + SCREENING_SAMPLER_SEED_POLICY["model_offsets"][model]
        + starting_trial_number
    )
    sampler_invocation = {
        "starting_optuna_trial_number": starting_trial_number,
        "expected_generation": generation,
        "resolved_seed": resolved_seed,
    }
    resolved_sampler = {**design["screening_sampler"], "seed": resolved_seed}
    command = _base_train_command(root, model, study) + [
        f"hydra.sweeper.sampler.seed={resolved_seed}",
        "hydra/launcher=packed_srun_clariden",
        f"hydra.sweeper.n_trials={trials}",
        f"hydra.sweeper.n_jobs={jobs}",
        f"hydra.sweep.dir={sweep_dir}",
        "hydra.launcher.cpus_per_task=16",
        "hydra.launcher.mem_gb=96",
    ]
    chunk_path = sweep_dir / "chunk.json"
    chunk = {
        "schema_version": 2 if design.get("screening_sampler") is not None else 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "model": model,
        "study_name": study["study_name"],
        "hparams_search": study.get("hparams_search", f"{model}_shared_optuna"),
        "screening_sampler_base": design.get("screening_sampler"),
        "screening_sampler_base_sha256": (
            None
            if design.get("screening_sampler") is None
            else _json_sha256(design["screening_sampler"])
        ),
        "screening_sampler": resolved_sampler,
        "screening_sampler_sha256": _json_sha256(resolved_sampler),
        "screening_sampler_seed_policy": design.get("screening_sampler_seed_policy"),
        "screening_sampler_seed_policy_sha256": _json_sha256(
            design["screening_sampler_seed_policy"]
        ),
        "sampler_invocation": sampler_invocation,
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
        _write_chunk_bindings(
            root,
            study,
            sweep_dir,
            prior_trial_numbers,
            sampler_invocation=sampler_invocation,
        )
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
            "schema_version": 2,
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


def production_canary(root: Path, model: str, attempt: int) -> None:
    """Run four full 50-epoch trials in an isolated disposable Optuna study."""
    if attempt <= 0:
        raise ValueError("Production-canary attempt must be positive.")
    design = _load_design(root)
    study = _study(design, model)
    output = root / "production_canaries" / model / f"attempt_{attempt:04d}"
    database = output / "study.db"
    if database.exists():
        raise RuntimeError(f"Production canary study already exists: {database}")
    output.mkdir(parents=True, exist_ok=True)
    study_name = (
        f"physics_{model}_production_canary_{design['code_commit'][:12]}_attempt_{attempt:04d}"
    )
    resolved_seed = (
        SCREENING_SAMPLER_SEED_POLICY["base_seed"]
        + SCREENING_SAMPLER_SEED_POLICY["model_offsets"][model]
    )
    invocation = {
        "starting_optuna_trial_number": 0,
        "expected_generation": 0,
        "resolved_seed": resolved_seed,
    }
    canary_study = {**study, "study_name": study_name}
    command = _base_train_command(
        root,
        model,
        canary_study,
        database=database,
        study_name=study_name,
    ) + [
        f"hydra.sweeper.sampler.seed={resolved_seed}",
        "hydra/launcher=packed_srun_clariden",
        "hydra.sweeper.n_trials=4",
        "hydra.sweeper.n_jobs=4",
        f"hydra.sweep.dir={output}",
        "hydra.launcher.cpus_per_task=16",
        "hydra.launcher.mem_gb=96",
    ]
    started_at = datetime.now(timezone.utc).isoformat()
    started_clock = time.monotonic()
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)  # nosec B603
    loaded = _optuna().load_study(study_name=study_name, storage=f"sqlite:///{database}")
    trials = list(loaded.get_trials(deepcopy=False))
    expected_values = 2 * len(study["objectives"])
    if len(trials) != 4 or any(
        trial.state != _optuna().trial.TrialState.COMPLETE
        or not _usable_trial(trial, expected_values=expected_values)
        for trial in trials
    ):
        raise RuntimeError("Production canary did not produce four finite completed trials.")
    bindings = _write_chunk_bindings(
        root,
        canary_study,
        output,
        set(),
        all_trials=trials,
        sampler_invocation=invocation,
    )
    rows = json.loads(bindings.read_text(encoding="utf-8"))["bindings"]
    if len(rows) != 4 or any(row["optimized_metric_artifact"] is None for row in rows):
        raise RuntimeError("Production canary bindings are incomplete.")
    _write_json(
        output / "success.json",
        {
            "schema_version": 2,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "started_at": started_at,
            "elapsed_seconds": time.monotonic() - started_clock,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "model": model,
            "attempt": attempt,
            "study_name": study_name,
            "trials": 4,
            "epochs": SEARCH_EPOCHS,
            "values_per_trial": expected_values,
            "code_commit": design["code_commit"],
            "screening_sampler_base": design["screening_sampler"],
            "screening_sampler_base_sha256": _json_sha256(design["screening_sampler"]),
            "screening_sampler": {**design["screening_sampler"], "seed": resolved_seed},
            "screening_sampler_sha256": _json_sha256(
                {**design["screening_sampler"], "seed": resolved_seed}
            ),
            "screening_sampler_seed_policy": design["screening_sampler_seed_policy"],
            "screening_sampler_seed_policy_sha256": _json_sha256(
                design["screening_sampler_seed_policy"]
            ),
            "sampler_invocation": invocation,
            "bindings": str(bindings.resolve()),
            "bindings_sha256": _sha256(bindings),
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
        f"hparams_search={study.get('hparams_search', f'{model}_shared_optuna')}",
        f"hydra.sweeper.sampler.seed={SCREENING_SAMPLER['seed']}",
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
        sampler_invocation={
            "starting_optuna_trial_number": 0,
            "expected_generation": 0,
            "resolved_seed": SCREENING_SAMPLER["seed"],
        },
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
            "schema_version": 2,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "started_at": started_at,
            "elapsed_seconds": elapsed_seconds,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "model": model,
            "trials": 4,
            "values_per_trial": expected_values,
            "threshold_checkpoint_audit": threshold_audit,
            "code_commit": design["code_commit"],
            "screening_sampler_base": design.get("screening_sampler"),
            "screening_sampler": design.get("screening_sampler"),
            "screening_sampler_seed_policy": design.get("screening_sampler_seed_policy"),
            "sampler_invocation": {
                "starting_optuna_trial_number": 0,
                "expected_generation": 0,
                "resolved_seed": SCREENING_SAMPLER["seed"],
            },
            "command": command,
        },
    )


def submit_next(root: Path, trials: int, jobs: int, timeout_min: int = 720) -> None:
    """Refuse the legacy launcher that bypasses the operational control plane."""
    raise RuntimeError(
        "submit-next is disabled. Use physics_background_pairing_operations.py "
        "submit-search so the canary gate, ledger, and generation boundaries are enforced."
    )


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
    elif args.command == "production-canary":
        production_canary(root, args.model, args.attempt)
    elif args.command == "sweep":
        sweep(root, args.model, args.trials, args.jobs, args.timeout_min)
    elif args.command == "submit-next":
        submit_next(root, args.trials, args.jobs, args.timeout_min)
    elif args.command == "freeze":
        print(freeze(root))
    elif args.command == "resolve-retrain-index":
        _load_design(root)
        print(
            resolve_retrain_task_index(
                root, args.task_map, args.task_map_sha256, args.local_task_id
            )
        )
    elif args.command == "retrain":
        print(retrain(root, args.index, args.task_map, args.task_map_sha256))
    elif args.command == "aggregate":
        print(aggregate(root))


if __name__ == "__main__":
    main()
