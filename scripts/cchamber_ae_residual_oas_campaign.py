"""Run the confirmatory residual-OAS Causal Chamber AE campaign.

The campaign reuses the frozen 16-candidate panel and reporting seeds from the paper audit, but
trains every trajectory again. All label-free branches consume the same train-normal-fitted
residual Mahalanobis score.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess  # nosec B404
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import cchamber_candidate_rank_audit as rank
import hydra
import pytorch_lightning as pl
import torch

from src.utils.pairing.io import compose_config

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_AUDIT = Path(
    "/iopsstor/scratch/cscs/vjimenez/adatl1/audits/"
    "cchamber_real_20260801_3789655_candidate_rank_3789655/audit.json"
)
EXPERIMENT = "cchamber/ae_residual_oas_confirmatory"
OUTPUT_NAME = "ascore/residual_oas"
EXPECTED_TRAJECTORIES = 48
EXPECTED_EPOCHS = 200


def _sha256(path: Path) -> str:
    """Return the streaming SHA-256 digest for one artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Any:
    """Load one UTF-8 JSON artifact."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    """Atomically write one JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _git_commit() -> str:
    """Return HEAD after requiring a clean campaign-design worktree."""
    git = shutil.which("git")
    if git is None:
        raise FileNotFoundError("git is required for campaign provenance.")
    status = subprocess.run(  # nosec B603
        [git, "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError("Campaign design requires a clean worktree.")
    return subprocess.run(  # nosec B603
        [git, "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def initialize(root: Path) -> Path:
    """Freeze source provenance and the exact 48-trajectory AE panel."""
    root = root.resolve()
    design_path = root / "design.json"
    if design_path.exists():
        return design_path
    source = _load(SOURCE_AUDIT)
    trajectory_path = Path(source["trajectory_manifest"])
    source_trajectories = _load(trajectory_path)
    selected = [row for row in source_trajectories if row["model"] == "ae"]
    if len(selected) != EXPECTED_TRAJECTORIES:
        raise ValueError("Source audit does not contain exactly 48 AE trajectories.")
    trajectories = []
    for index, row in enumerate(selected):
        copied = dict(row)
        copied["source_trajectory_index"] = int(row["trajectory_index"])
        copied["trajectory_index"] = index
        trajectories.append(copied)
    trajectory_output = root / "trajectories.json"
    _write_json(trajectory_output, trajectories)
    design = {
        "schema_version": 1,
        "classification": "confirmatory_residual_oas_ae_rerun",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": _git_commit(),
        "experiment": EXPERIMENT,
        "score_output": OUTPUT_NAME,
        "expected_epochs": EXPECTED_EPOCHS,
        "expected_trajectories": EXPECTED_TRAJECTORIES,
        "strategies": list(rank.STRATEGIES),
        "source_audit": str(SOURCE_AUDIT),
        "source_audit_sha256": _sha256(SOURCE_AUDIT),
        "source_trajectory_manifest": str(trajectory_path),
        "source_trajectory_manifest_sha256": _sha256(trajectory_path),
        "trajectory_manifest": str(trajectory_output),
        "trajectory_manifest_sha256": _sha256(trajectory_output),
        "data_dir": source["data_dir"],
        "interventions": source["interventions"],
        "encoder_validation_pair_table": source["encoder_validation_pair_table"],
        "encoder_validation_pair_table_sha256": source["encoder_validation_pair_table_sha256"],
        "outcomes_inspected_before_checkpoint_freeze": False,
        "primary_strategy": "cap_cdf",
        "primary_score": "train_normal_residual_oas",
    }
    _write_json(design_path, design)
    return design_path


def _design(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load a frozen design and verify its trajectory and code identities."""
    design_path = root.resolve() / "design.json"
    design = _load(design_path)
    if _sha256(Path(design["trajectory_manifest"])) != design["trajectory_manifest_sha256"]:
        raise ValueError("Trajectory manifest changed after design freeze.")
    git = shutil.which("git")
    if git is None:
        raise FileNotFoundError("git is required for campaign provenance.")
    current = subprocess.run(  # nosec B603
        [git, "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current != design["code_commit"]:
        raise RuntimeError("Current commit differs from the frozen campaign commit.")
    return design, _load(Path(design["trajectory_manifest"]))


def _training_command(
    root: Path, design: Mapping[str, Any], trajectory: Mapping[str, Any], attempt: Path
) -> tuple[list[str], dict[str, str]]:
    """Build one exact Hydra training command and its provenance environment."""
    run_name = f"ae_oas_t{int(trajectory['trajectory_index']):03d}_{attempt.name}"
    command = [
        sys.executable,
        "src/train.py",
        f"experiment={EXPERIMENT}",
        "trainer=gpu",
        "trainer.devices=[0]",
        "trainer.deterministic=true",
        f"trainer.min_epochs={EXPECTED_EPOCHS}",
        f"trainer.max_epochs={EXPECTED_EPOCHS}",
        "trainer.num_sanity_val_steps=0",
        f"seed={int(trajectory['reporting_seed'])}",
        "data.seed=314159",
        "data.signal_experiments=[]",
        "train=true",
        "test=false",
        "logger=none",
        "experiment_name=cchamber_ae_residual_oas_confirmatory",
        f"run_name={run_name}",
        f"paths.base_data_dir={design['data_dir']}",
        f"paths.log_dir={root / 'logs'}",
        f"paths.checkpoints_dir={attempt / 'checkpoints'}",
        f"hydra.run.dir={attempt / 'hydra'}",
        "extras.print_config=false",
        *[f"{name}={rank._hydra_value(value)}" for name, value in trajectory["params"].items()],
    ]
    environment = os.environ.copy()
    environment["CCHAMBER_VALID_PAIR_TABLE"] = str(design["encoder_validation_pair_table"])
    environment["CCHAMBER_AUDIT_CHECKPOINT_MANIFEST"] = str(attempt / "checkpoint_branches.json")
    environment["CCHAMBER_AUDIT_TRAJECTORY_FINGERPRINT"] = str(
        attempt / "trajectory_fingerprint.json"
    )
    environment["LOG_DIR"] = str(root / "logs")
    return command, environment


def train(root: Path, trajectory_index: int) -> Path:
    """Train one trajectory and validate its six checkpoint branches."""
    if "SLURM_JOB_ID" not in os.environ or not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA Slurm allocation.")
    root = root.resolve()
    design, trajectories = _design(root)
    trajectory = trajectories[trajectory_index]
    marker = root / "training" / f"{trajectory_index:03d}.json"
    if marker.exists():
        return marker
    attempt = root / "attempts" / "training" / f"{trajectory_index:03d}" / rank._attempt_id()
    command, environment = _training_command(root, design, trajectory, attempt)
    subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)  # nosec B603
    branch_manifest = attempt / "checkpoint_branches.json"
    branches = rank._validate_branch_manifest(branch_manifest, expected_epochs=EXPECTED_EPOCHS)
    value = {
        "schema_version": 1,
        **{key: value for key, value in trajectory.items() if key != "params"},
        "params": trajectory["params"],
        "code_commit": design["code_commit"],
        "branches": branches,
        "branch_manifest": str(branch_manifest),
        "branch_manifest_sha256": _sha256(branch_manifest),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    _write_json(marker, value)
    return marker


def freeze(root: Path) -> Path:
    """Freeze hashes for all 288 checkpoints before intervention evaluation."""
    root = root.resolve()
    design, trajectories = _design(root)
    records = []
    for trajectory in trajectories:
        marker_path = root / "training" / f"{int(trajectory['trajectory_index']):03d}.json"
        marker = _load(marker_path)
        for branch in marker["branches"]:
            checkpoint = Path(branch["checkpoint"])
            records.append(
                {
                    "trajectory_index": int(trajectory["trajectory_index"]),
                    "candidate_id": trajectory["candidate_id"],
                    "reporting_seed": int(trajectory["reporting_seed"]),
                    **branch,
                    "checkpoint_sha256": _sha256(checkpoint),
                }
            )
    if len(records) != EXPECTED_TRAJECTORIES * len(rank.STRATEGIES):
        raise ValueError("Checkpoint freeze coverage is incomplete.")
    output = root / "checkpoint_manifest.json"
    _write_json(
        output,
        {
            "schema_version": 1,
            "design_sha256": _sha256(root / "design.json"),
            "code_commit": design["code_commit"],
            "outcomes_inspected_before_freeze": False,
            "checkpoints": records,
        },
    )
    return output


class ResidualOASMetrics(rank._SealedMetricsCallback):
    """Evaluate interventions with the checkpointed residual-OAS score."""

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        """Collect the covariance-aware score rather than legacy scalar MSE."""
        name = list(trainer.test_dataloaders)[dataloader_idx]
        score = outputs[OUTPUT_NAME].detach().view(-1).cpu()
        if not torch.isfinite(score).all():
            raise ValueError(f"Non-finite residual-OAS scores for {name}.")
        self.scores[name].append(score)


def _compose(design: Mapping[str, Any], trajectory: Mapping[str, Any]):
    """Compose the AE and all sealed Causal Chamber test loaders."""
    os.environ["CCHAMBER_VALID_PAIR_TABLE"] = str(design["encoder_validation_pair_table"])
    overrides = [
        f"experiment={EXPERIMENT}",
        f"seed={int(trajectory['reporting_seed'])}",
        "data.seed=314159",
        f"paths.base_data_dir={design['data_dir']}",
        f"data.signal_experiments={rank._hydra_value(design['interventions'])}",
        "logger=none",
        *[f"{name}={rank._hydra_value(value)}" for name, value in trajectory["params"].items()],
    ]
    cfg = compose_config(overrides=overrides)
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup("test")
    loaders = datamodule.test_dataloader()
    loaders["normal"].loader = datamodule.loader
    return cfg, datamodule, loaders


def evaluate(root: Path, trajectory_index: int) -> Path:
    """Evaluate six frozen checkpoints for one trajectory on all interventions."""
    if "SLURM_JOB_ID" not in os.environ or not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires a CUDA Slurm allocation.")
    root = root.resolve()
    design, trajectories = _design(root)
    trajectory = trajectories[trajectory_index]
    marker_path = root / "evaluation" / f"{trajectory_index:03d}.json"
    if marker_path.exists():
        return marker_path
    freeze_manifest = _load(root / "checkpoint_manifest.json")
    branches = [
        row
        for row in freeze_manifest["checkpoints"]
        if row["trajectory_index"] == trajectory_index
    ]
    if {row["strategy"] for row in branches} != set(rank.STRATEGIES):
        raise ValueError("Frozen trajectory does not contain all strategies.")
    cfg, datamodule, loaders = _compose(design, trajectory)
    output_dir = root / "attempts" / "evaluation" / f"{trajectory_index:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for branch in branches:
        checkpoint = Path(branch["checkpoint"])
        if _sha256(checkpoint) != branch["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed after freeze.")
        pl.seed_everything(int(trajectory["reporting_seed"]), workers=True)
        model = hydra.utils.instantiate(cfg.algorithm)
        rank._load_checkpoint_state(model, checkpoint)
        output = output_dir / f"{branch['strategy']}.csv"
        callback = ResidualOASMetrics(
            design["interventions"],
            output,
            {
                "trajectory_index": trajectory_index,
                "model": "ae",
                "candidate_id": trajectory["candidate_id"],
                "reporting_seed": int(trajectory["reporting_seed"]),
                "strategy": branch["strategy"],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": branch["checkpoint_sha256"],
                "selected_epoch": int(branch["selected_epoch"]),
                "monitor": branch["monitor"],
                "monitor_value": float(branch["monitor_value"]),
            },
        )
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            logger=False,
            callbacks=[callback],
            enable_checkpointing=False,
            enable_progress_bar=False,
            deterministic=True,
            inference_mode=True,
        )
        trainer.split = "test"
        trainer.test(model=model, dataloaders=loaders, verbose=False)
        outputs.append(output)
    datamodule.teardown("test")
    combined = root / "evaluation" / f"{trajectory_index:03d}.csv"
    combined.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for output in outputs:
        with output.open("r", encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    rank._write_csv(combined, rows)
    _write_json(
        marker_path,
        {
            "schema_version": 1,
            "trajectory_index": trajectory_index,
            "candidate_id": trajectory["candidate_id"],
            "reporting_seed": int(trajectory["reporting_seed"]),
            "evaluation_rows": str(combined),
            "evaluation_rows_sha256": _sha256(combined),
            "n_rows": len(rows),
        },
    )
    return marker_path


def collect(root: Path) -> Path:
    """Collect and validate every trajectory evaluation table."""
    root = root.resolve()
    rows = []
    for index in range(EXPECTED_TRAJECTORIES):
        marker = _load(root / "evaluation" / f"{index:03d}.json")
        path = Path(marker["evaluation_rows"])
        if _sha256(path) != marker["evaluation_rows_sha256"]:
            raise ValueError("Evaluation rows changed after completion.")
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    expected = EXPECTED_TRAJECTORIES * len(rank.STRATEGIES) * 58 * 2
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} rows, found {len(rows)}.")
    output = root / "results" / "evaluation_rows.csv"
    rank._write_csv(output, rows)
    return output


def _parser() -> argparse.ArgumentParser:
    """Build the stage-oriented command-line parser."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "freeze", "collect"):
        command = subparsers.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
    for name in ("train", "evaluate"):
        command = subparsers.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--trajectory-index", type=int, required=True)
    return parser


def main() -> None:
    """Dispatch one campaign stage."""
    args = _parser().parse_args()
    if args.command == "init":
        print(initialize(args.root))
    elif args.command == "train":
        print(train(args.root, args.trajectory_index))
    elif args.command == "freeze":
        print(freeze(args.root))
    elif args.command == "evaluate":
        print(evaluate(args.root, args.trajectory_index))
    else:
        print(collect(args.root))


if __name__ == "__main__":
    main()
