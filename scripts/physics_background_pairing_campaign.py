#!/usr/bin/env python3
"""Manage the resumable physics background-pairing search campaign."""

from __future__ import annotations

import argparse
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
CAP_STRATEGIES = (
    "flat_physical",
    "physics_summary",
    "typed_sliced_wasserstein",
    "jetclr",
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
    sweep = subparsers.add_parser("sweep")
    sweep.add_argument("--cell", required=True)
    sweep.add_argument("--trials", type=int, default=48)
    sweep.add_argument("--jobs", type=int, default=48)
    submit = subparsers.add_parser("submit-next")
    submit.add_argument("--trials", type=int, default=48)
    submit.add_argument("--jobs", type=int, default=48)
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


def _cells() -> list[dict[str, str]]:
    """Construct the complete 36-cell primary search matrix."""
    cells = []
    for model in MODELS:
        for strategy in CAP_STRATEGIES:
            cells.append(
                {
                    "id": f"{model}__cap__{strategy}",
                    "model": model,
                    "metric": "cap_mapping",
                    "strategy": strategy,
                }
            )
        for metric in METRICS:
            cells.append(
                {
                    "id": f"{model}__{metric}",
                    "model": model,
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


def status(root: Path) -> list[dict[str, Any]]:
    """Return and print completion state for all primary cells."""
    design = _load_design(root)
    rows = []
    for cell in design["cells"]:
        counts = _trial_counts(root, cell["id"])
        rows.append({**cell, **counts, "target": TARGET_TRIALS})
    print(json.dumps(rows, indent=2, sort_keys=True))
    return rows


def _base_train_command(root: Path, cell: dict[str, str]) -> list[str]:
    """Build invariant Hydra overrides for one selection cell."""
    return [
        sys.executable,
        "src/train.py",
        "-m",
        f"experiment=physics/{cell['model']}_background_pairing",
        f"+selection_metric={cell['metric']}",
        f"physics_pairing.strategy={cell['strategy']}",
        f"hparams_search={cell['model']}_optuna",
        f"experiment_name=physics_bgpair_{cell['id']}",
        "logger=none",
        "trainer=gpu",
        "trainer.devices=[0]",
        f"trainer.min_epochs={SEARCH_EPOCHS}",
        f"trainer.max_epochs={SEARCH_EPOCHS}",
        "trainer.num_sanity_val_steps=0",
        "test=false",
        f"hydra.sweeper.storage=sqlite:///{_study_path(root, cell['id'])}",
        f"hydra.sweeper.study_name={cell['id']}",
    ]


def sweep(root: Path, cell_id: str, trials: int, jobs: int) -> None:
    """Launch one resumable Optuna chunk through Hydra's Slurm launcher."""
    design = _load_design(root)
    cell = _cell(design, cell_id)
    if trials <= 0 or jobs <= 0:
        raise ValueError("trials and jobs must be positive.")
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


def pilot(root: Path, metric: str) -> None:
    """Run one one-epoch AE pilot for a model-selection metric family."""
    design = _load_design(root)
    cell = {
        "model": "ae",
        "metric": metric,
        "strategy": "physics_summary",
    }
    output = root / "pilots" / metric
    command = [
        sys.executable,
        "src/train.py",
        f"experiment=physics/{cell['model']}_background_pairing",
        f"+selection_metric={metric}",
        f"physics_pairing.strategy={cell['strategy']}",
        f"experiment_name=physics_bgpair_pilot_{metric}",
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
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)  # nosec B603
    _write_json(
        output / "success.json",
        {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "metric": metric,
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
        pilot(root, args.metric)
    elif args.command == "sweep":
        sweep(root, args.cell, args.trials, args.jobs)
    elif args.command == "submit-next":
        submit_next(root, args.trials, args.jobs)


if __name__ == "__main__":
    main()
