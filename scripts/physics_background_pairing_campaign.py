#!/usr/bin/env python3
"""Manage the resumable physics background-pairing search campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess  # nosec B404 -- executes frozen repository/Slurm commands
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import optuna

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path(
    "/iopsstor/scratch/cscs/vjimenez/adatl1/campaigns/" "physics_background_pairing_20260803"
)
MODELS = ("ae", "vae", "dsae", "dsvae", "svdd", "realnvp")
MODEL_SCORES = {
    "ae": ("mse", "residual_oas"),
    "vae": ("kl_raw", "residual_oas"),
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
METRICS = ("wasserstein", "drift")
TARGET_TRIALS = 600
SEARCH_EPOCHS = 50
RETRAIN_EPOCHS = 200


def parse_args() -> argparse.Namespace:
    """Parse one campaign-management command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init")
    subparsers.add_parser("status")
    pilot = subparsers.add_parser("pilot")
    pilot.add_argument("--metric", choices=("cap_mapping", *METRICS), required=True)
    pilot.add_argument("--model", choices=MODELS, default="ae")
    pilot.add_argument("--score", default="residual_oas")
    pilot.add_argument("--strategy", choices=CAP_STRATEGIES, default="physics_summary")
    sweep = subparsers.add_parser("sweep")
    sweep.add_argument("--cell", required=True)
    sweep.add_argument("--trials", type=int, default=48)
    sweep.add_argument("--jobs", type=int, default=48)
    submit = subparsers.add_parser("submit-next")
    submit.add_argument("--trials", type=int, default=48)
    submit.add_argument("--jobs", type=int, default=48)
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
    """Convert tuples and scalar-like Optuna values into JSON-compatible values."""
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "item"):
        return value.item()
    return value


def _cells() -> list[dict[str, str]]:
    """Construct the complete score-stratified primary search matrix."""
    cells = []
    for model in MODELS:
        for score in MODEL_SCORES[model]:
            score_id = f"__{score}" if len(MODEL_SCORES[model]) > 1 else ""
            for strategy in CAP_STRATEGIES:
                cells.append(
                    {
                        "id": f"{model}{score_id}__cap__{strategy}",
                        "model": model,
                        "score": score,
                        "metric": "cap_mapping",
                        "strategy": strategy,
                    }
                )
            for metric in METRICS:
                cells.append(
                    {
                        "id": f"{model}{score_id}__{metric}",
                        "model": model,
                        "score": score,
                        "metric": metric,
                        "strategy": "physics_summary",
                    }
                )
    return cells


def initialize(root: Path) -> Path:
    """Freeze campaign design, code, and authenticated input provenance."""
    root = root.expanduser().resolve()
    design_path = root / "design.json"
    if design_path.exists():
        return design_path
    preflight = REPOSITORY_ROOT / "research/physics_background_pairing/preflight.json"
    design = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": _git_commit(),
        "preflight": str(preflight),
        "preflight_sha256": _sha256(preflight),
        "target_trials_per_cell": TARGET_TRIALS,
        "search_epochs": SEARCH_EPOCHS,
        "retrain_epochs": RETRAIN_EPOCHS,
        "models": list(MODELS),
        "model_scores": {model: list(scores) for model, scores in MODEL_SCORES.items()},
        "cap_strategies": list(CAP_STRATEGIES),
        "cells": _cells(),
    }
    _write_json(design_path, design, create=True)
    return design_path


def _load_design(root: Path) -> dict[str, Any]:
    """Load and authenticate the frozen campaign design."""
    design_path = initialize(root)
    design = json.loads(design_path.read_text(encoding="utf-8"))
    if _git_commit() != design["code_commit"]:
        raise RuntimeError("Current commit differs from the frozen campaign design.")
    if _sha256(Path(design["preflight"])) != design["preflight_sha256"]:
        raise RuntimeError("Pair-table preflight changed after campaign initialization.")
    return design


def _cell(design: dict[str, Any], cell_id: str) -> dict[str, str]:
    """Resolve one unique matrix cell by identifier."""
    matches = [cell for cell in design["cells"] if cell["id"] == cell_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown or ambiguous campaign cell: {cell_id}")
    return matches[0]


def _study_path(root: Path, cell_id: str) -> Path:
    """Return one cell's persistent Optuna SQLite path."""
    return root / "studies" / f"{cell_id}.db"


def _trial_counts(root: Path, cell_id: str) -> dict[str, int]:
    """Count persistent trials by Optuna state without creating a study."""
    database = _study_path(root, cell_id)
    if not database.is_file():
        return {"complete": 0, "fail": 0, "running": 0, "waiting": 0, "pruned": 0}
    study = optuna.load_study(study_name=cell_id, storage=f"sqlite:///{database}")
    counts = {state.name.lower(): 0 for state in optuna.trial.TrialState}
    for trial in study.get_trials(deepcopy=False):
        counts[trial.state.name.lower()] += 1
    return counts


def _fail_stale_running_trials(root: Path, cell_id: str) -> int:
    """Mark orphaned running trials failed before starting a replacement chunk."""
    database = _study_path(root, cell_id)
    if not database.is_file():
        return 0
    study = optuna.load_study(study_name=cell_id, storage=f"sqlite:///{database}")
    stale = [
        trial
        for trial in study.get_trials(deepcopy=False)
        if trial.state == optuna.trial.TrialState.RUNNING
    ]
    for trial in stale:
        study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
    return len(stale)


def status(root: Path) -> list[dict[str, Any]]:
    """Return and print completion state for all primary cells."""
    design = _load_design(root)
    rows = []
    for cell in design["cells"]:
        counts = _trial_counts(root, cell["id"])
        rows.append({**cell, **counts, "target": TARGET_TRIALS})
    print(json.dumps(rows, indent=2, sort_keys=True))
    return rows


def freeze(root: Path) -> Path:
    """Freeze every completed study's full Pareto front before anomaly evaluation."""
    design = _load_design(root)
    manifest_path = root / "selection" / "pareto_retrain_manifest.json"
    if manifest_path.exists():
        return manifest_path
    rows = []
    for cell in design["cells"]:
        counts = _trial_counts(root, cell["id"])
        if counts["complete"] < TARGET_TRIALS:
            raise RuntimeError(
                f"Cannot freeze {cell['id']}: {counts['complete']}/{TARGET_TRIALS} "
                "trials are complete."
            )
        database = _study_path(root, cell["id"])
        study = optuna.load_study(study_name=cell["id"], storage=f"sqlite:///{database}")
        for trial in sorted(study.best_trials, key=lambda item: item.number):
            rows.append(
                {
                    "retrain_index": len(rows),
                    **cell,
                    "trial_number": int(trial.number),
                    "objective_values": _jsonable(trial.values),
                    "params": _jsonable(trial.params),
                    "seed": 123,
                    "epochs": RETRAIN_EPOCHS,
                }
            )
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": design["code_commit"],
        "selection_uses_downstream_anomalies": False,
        "rows": rows,
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


def retrain(root: Path, index: int) -> Path:
    """Retrain one frozen Pareto-front trial for 200 epochs and evaluate anomalies."""
    design = _load_design(root)
    manifest_path = root / "selection" / "pareto_retrain_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Freeze the Pareto fronts before retraining.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest["rows"]
    if index < 0 or index >= len(rows):
        raise IndexError(f"Retrain index {index} is outside [0,{len(rows)}).")
    row = rows[index]
    marker = root / "retraining" / "markers" / f"{index:04d}.json"
    if marker.is_file():
        return marker
    run_name = f"{row['id']}__trial{int(row['trial_number']):04d}"
    hydra_dir = root / "retraining" / "hydra" / f"{index:04d}"
    command = [
        sys.executable,
        "src/train.py",
        f"experiment=physics/{row['model']}_background_pairing",
        f"+selection_metric={row['metric']}_retrain",
        "experiment_name=physics_bgpair_retrain",
        f"run_name={run_name}",
        f"hydra.run.dir={hydra_dir}",
        "logger=none",
        "trainer=gpu",
        "trainer.devices=[0]",
        f"trainer.min_epochs={RETRAIN_EPOCHS}",
        f"trainer.max_epochs={RETRAIN_EPOCHS}",
        "trainer.num_sanity_val_steps=0",
        "test=true",
        f"seed={int(row['seed'])}",
    ]
    command.extend(_pairing_overrides(row))
    command.extend(_score_overrides(row))
    command.extend(f"{key}={_hydra_value(value)}" for key, value in row["params"].items())
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)  # nosec B603
    optimized = hydra_dir / "optimized_metric.json"
    if not optimized.is_file():
        raise RuntimeError(f"Retraining did not emit {optimized}.")
    checkpoint_root = Path(os.environ.get("CHECKPOINT_DIR", REPOSITORY_ROOT / "checkpoints"))
    checkpoint_root = checkpoint_root / "physics_bgpair_retrain" / run_name
    value_files = sorted(checkpoint_root.glob("**/plots/test/**/anomaly_efficiency/values.csv"))
    if not value_files:
        raise RuntimeError("Retraining emitted no held-out anomaly-efficiency values.")
    _write_json(
        marker,
        {
            "schema_version": 1,
            "completed_at": datetime.now(timezone.utc).isoformat(),
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


def aggregate(root: Path) -> Path:
    """Select the paper-style downstream oracle winner within each frozen Pareto set."""
    design = _load_design(root)
    manifest_path = root / "selection" / "pareto_retrain_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = []
    for row in manifest["rows"]:
        index = int(row["retrain_index"])
        marker_path = root / "retraining" / "markers" / f"{index:04d}.json"
        if not marker_path.is_file():
            raise RuntimeError(f"Missing retraining marker {marker_path}.")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        optimized_path = Path(marker["optimized_metric_artifact"])
        if _sha256(optimized_path) != marker["optimized_metric_artifact_sha256"]:
            raise RuntimeError(f"Optimized-metric artifact changed: {optimized_path}")
        optimized = json.loads(optimized_path.read_text(encoding="utf-8"))
        checkpoint = optimized["optimized_ckpt_name"]
        efficiencies = {}
        for entry in marker["value_files"]:
            path = Path(entry["path"])
            if _sha256(path) != entry["sha256"]:
                raise RuntimeError(f"Efficiency artifact changed: {path}")
            with path.open(encoding="utf-8", newline="") as handle:
                for value_row in csv.DictReader(handle):
                    if _checkpoint_identity(value_row["checkpoint"]) != checkpoint:
                        continue
                    if value_row["metric"] != "efficiency_operational":
                        continue
                    efficiencies[value_row["intervention"]] = float(value_row["value"])
        if len(efficiencies) != 20:
            raise RuntimeError(
                f"Expected 20 downstream efficiencies for retrain {index}, "
                f"found {len(efficiencies)}."
            )
        results.append(
            {
                **row,
                "optimized_ckpt_name": checkpoint,
                "mean_downstream_efficiency": sum(efficiencies.values()) / len(efficiencies),
                "downstream_efficiencies": efficiencies,
            }
        )
    winners = []
    for cell in design["cells"]:
        candidates = [result for result in results if result["id"] == cell["id"]]
        winners.append(max(candidates, key=lambda item: item["mean_downstream_efficiency"]))
    output = root / "results" / "oracle_winners.json"
    _write_json(
        output,
        {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "oracle_uses_downstream_anomalies": True,
            "winners": winners,
            "all_retrains": results,
        },
    )
    return output


def _base_train_command(root: Path, cell: dict[str, str]) -> list[str]:
    """Build invariant Hydra overrides for one selection cell."""
    command = [
        sys.executable,
        "src/train.py",
        "-m",
        f"experiment=physics/{cell['model']}_background_pairing",
        f"+selection_metric={cell['metric']}",
        f"hparams_search={cell['model']}_optuna",
        f"experiment_name=physics_bgpair_{cell['id']}",
        "logger=none",
        "trainer=gpu",
        "trainer.devices=[0]",
        f"trainer.min_epochs={SEARCH_EPOCHS}",
        f"trainer.max_epochs={SEARCH_EPOCHS}",
        "trainer.num_sanity_val_steps=0",
        "test=false",
        f"hydra.sweeper.storage=sqlite:///{_study_path(root, cell['id'])}?timeout=600",
        f"hydra.sweeper.study_name={cell['id']}",
    ]
    command.extend(_pairing_overrides(cell))
    command.extend(_score_overrides(cell))
    return command


def _pairing_overrides(cell: dict[str, str]) -> list[str]:
    """Select a precomputed CAP table or the empirical-CDF control."""
    if cell["metric"] == "cap_mapping" and cell["strategy"] == "cdf":
        return ["pairing=physics_cdf"]
    return [f"physics_pairing.strategy={cell['strategy']}"]


def _score_overrides(cell: dict[str, str]) -> list[str]:
    """Route one model's generic anomaly output through the selected score."""
    model, score = cell["model"], cell["score"]
    if score not in MODEL_SCORES[model]:
        raise ValueError(f"Score {score!r} is not configured for model {model!r}.")
    if model == "ae":
        overrides = [f"algorithm.anomaly_score={score}"]
        if score == "mse":
            overrides.append("callbacks.residual_oas_state=null")
        return overrides
    if model == "vae":
        overrides = [f"algorithm.anomaly_score={score}"]
        if score == "kl_raw":
            overrides.append("callbacks.vae_residual_state=null")
        return overrides
    return []


def sweep(root: Path, cell_id: str, trials: int, jobs: int) -> None:
    """Launch one resumable Optuna chunk through Hydra's Slurm launcher."""
    design = _load_design(root)
    cell = _cell(design, cell_id)
    if trials <= 0 or jobs <= 0:
        raise ValueError("trials and jobs must be positive.")
    _fail_stale_running_trials(root, cell_id)
    remaining = TARGET_TRIALS - _trial_counts(root, cell_id)["complete"]
    if remaining <= 0:
        print(f"{cell_id} already has {TARGET_TRIALS} completed trials.")
        return
    trials = min(int(trials), remaining)
    jobs = min(int(jobs), trials)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sweep_dir = root / "sweeps" / cell_id / stamp
    database = _study_path(root, cell_id)
    database.parent.mkdir(parents=True, exist_ok=True)
    command = _base_train_command(root, cell) + [
        "hydra/launcher=submitit_slurm_clariden",
        f"hydra.sweeper.n_trials={trials}",
        f"hydra.sweeper.n_jobs={jobs}",
        f"hydra.sweep.dir={sweep_dir}",
        f"hydra.launcher.array_parallelism={jobs}",
        "hydra.launcher.timeout_min=720",
        "hydra.launcher.cpus_per_task=16",
        "hydra.launcher.mem_gb=96",
    ]
    if cell["metric"] != "cap_mapping":
        command.append("hydra.sweeper.direction=[minimize,minimize]")
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)  # nosec B603


def pilot(root: Path, metric: str, model: str, score: str, strategy: str) -> None:
    """Run one one-epoch pilot for a model, score, and metric/pairing family."""
    design = _load_design(root)
    if score not in MODEL_SCORES[model]:
        raise ValueError(f"Score {score!r} is not configured for model {model!r}.")
    if metric != "cap_mapping" and strategy != "physics_summary":
        raise ValueError("A pairing strategy can only be selected for a CAP pilot.")
    cell = {
        "model": model,
        "score": score,
        "metric": metric,
        "strategy": strategy,
    }
    pilot_id = f"{model}__{score}__{metric}"
    if metric == "cap_mapping":
        pilot_id = f"{pilot_id}__{strategy}"
    output = root / "pilots" / pilot_id
    command = [
        sys.executable,
        "src/train.py",
        f"experiment=physics/{cell['model']}_background_pairing",
        f"+selection_metric={metric}",
        f"experiment_name=physics_bgpair_pilot_{model}_{score}_{metric}",
        f"run_name={design['code_commit'][:12]}",
        f"hydra.run.dir={output}",
        "logger=none",
        "trainer=gpu",
        "trainer.devices=[0]",
        "trainer.min_epochs=1",
        "trainer.max_epochs=1",
        "trainer.num_sanity_val_steps=0",
        "+trainer.limit_train_batches=2",
        "test=false",
    ]
    command.extend(_pairing_overrides(cell))
    command.extend(_score_overrides(cell))
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)  # nosec B603
    _write_json(
        output / "success.json",
        {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "metric": metric,
            "model": model,
            "score": score,
            "strategy": strategy,
            "code_commit": design["code_commit"],
            "command": command,
        },
    )


def submit_next(root: Path, trials: int, jobs: int) -> None:
    """Submit a controller for the first incomplete cell in design order."""
    design = _load_design(root)
    incomplete = [
        cell
        for cell in design["cells"]
        if _trial_counts(root, cell["id"])["complete"] < TARGET_TRIALS
    ]
    if not incomplete:
        print("All primary search cells are complete.")
        return
    sbatch = shutil.which("sbatch")
    if sbatch is None:
        raise FileNotFoundError("sbatch is required to submit campaign controllers.")
    cell = incomplete[0]
    script = REPOSITORY_ROOT / "scripts/clariden/physics_background_pairing_sweep.sbatch"
    result = subprocess.run(  # nosec B603
        [
            sbatch,
            "--parsable",
            f"--export=ALL,CELL_ID={cell['id']},CHUNK_TRIALS={trials},CHUNK_JOBS={jobs}",
            str(script),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    print(json.dumps({"cell": cell["id"], "job_id": result.stdout.strip()}))


def main() -> None:
    """Dispatch campaign initialization, status, pilots, search, or submission."""
    args = parse_args()
    root = args.root.expanduser().resolve()
    if args.command == "init":
        print(initialize(root))
    elif args.command == "status":
        status(root)
    elif args.command == "pilot":
        pilot(root, args.metric, args.model, args.score, args.strategy)
    elif args.command == "sweep":
        sweep(root, args.cell, args.trials, args.jobs)
    elif args.command == "submit-next":
        submit_next(root, args.trials, args.jobs)
    elif args.command == "freeze":
        print(freeze(root))
    elif args.command == "retrain":
        print(retrain(root, args.index))
    elif args.command == "aggregate":
        print(aggregate(root))


if __name__ == "__main__":
    main()
