"""Run inverse-CAP SVDD initialized by the selected residual-OAS AE encoder."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess  # nosec B404
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import cchamber_ae_residual_oas_campaign as common
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
AE_ROOT = Path(
    "/iopsstor/scratch/cscs/vjimenez/adatl1/campaigns/" "cchamber_ae_residual_oas_20260802_6e05000"
)
SCORE_AUDIT_ROOT = Path(
    "/iopsstor/scratch/cscs/vjimenez/adatl1/audits/"
    "cchamber_svdd_score_geometry_20260802_6e05000"
)
EXPERIMENT = "cchamber/svdd_inverse_cap_confirmatory"
OUTPUT_NAME = "ascore/full"
EXPECTED_TRAJECTORIES = 48
EXPECTED_EPOCHS = 200
PRIMARY_STRATEGY = "cap_encoder_nearest"
AE_INITIALIZATION_STRATEGY = "cap_cdf"
CAP_STRATEGIES = {
    "cap_metadata_nearest",
    "cap_encoder_nearest",
    "cap_cdf",
    "cap_random",
}


def _selected_ae() -> dict[str, Any]:
    """Return the outcome-blind CDF checkpoint selected by the AE campaign."""
    path = AE_ROOT / "analysis" / "selection_manifest.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle) if row["strategy"] == AE_INITIALIZATION_STRATEGY
        ]
    if len(rows) != 1:
        raise ValueError("AE analysis does not contain one CDF selection.")
    row = rows[0]
    checkpoint = Path(row["initialization_checkpoint"])
    if common._sha256(checkpoint) != row["initialization_checkpoint_sha256"]:
        raise ValueError("Selected AE checkpoint hash mismatch.")
    if row["selected_candidate"].zfill(3) != "060":
        raise ValueError("The frozen residual-OAS CDF selection is no longer candidate 060.")
    return row


def initialize(root: Path) -> Path:
    """Freeze the SVDD panel, selected AE initialization, and score-audit decision."""
    root = root.resolve()
    output = root / "design.json"
    if output.exists():
        return output
    source = common._load(SOURCE_AUDIT)
    source_trajectories = common._load(Path(source["trajectory_manifest"]))
    selected = [dict(row) for row in source_trajectories if row["model"] == "svdd"]
    if len(selected) != EXPECTED_TRAJECTORIES:
        raise ValueError("Source audit does not contain exactly 48 SVDD trajectories.")
    for index, row in enumerate(selected):
        row["source_trajectory_index"] = int(row["trajectory_index"])
        row["trajectory_index"] = index
    trajectory_path = root / "trajectories.json"
    common._write_json(trajectory_path, selected)
    ae = _selected_ae()
    score_design = common._load(SCORE_AUDIT_ROOT / "design.json")
    score_results = SCORE_AUDIT_ROOT / "analysis" / "score_cap_rank_associations.csv"
    if not score_results.is_file():
        raise FileNotFoundError("The outcome-gated SVDD score audit is not complete.")
    with score_results.open("r", encoding="utf-8", newline="") as handle:
        primary_rows = [
            row
            for row in csv.DictReader(handle)
            if row["score"] == "radial_d2"
            and row["pairing"] == "encoder"
            and row["direction"] == "minimize"
            and row["endpoint"] == "auprc"
        ]
    if len(primary_rows) != 1:
        raise ValueError("SVDD score audit lacks the inverse radial encoder row.")
    primary = primary_rows[0]
    if float(primary["spearman_rho"]) <= 0.0:
        raise ValueError("Inverse radial encoder CAP did not rank SVDD AUPRC positively.")
    design = {
        "schema_version": 1,
        "classification": "confirmatory_svdd_radial_inverse_cap_rerun",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": common._git_commit(),
        "experiment": EXPERIMENT,
        "score_output": OUTPUT_NAME,
        "expected_epochs": EXPECTED_EPOCHS,
        "expected_trajectories": EXPECTED_TRAJECTORIES,
        "strategies": list(rank.STRATEGIES),
        "directions": {
            strategy: "minimize" if strategy in CAP_STRATEGIES else rank.DIRECTIONS[strategy]
            for strategy in rank.STRATEGIES
        },
        "source_audit": str(SOURCE_AUDIT),
        "source_audit_sha256": common._sha256(SOURCE_AUDIT),
        "trajectory_manifest": str(trajectory_path),
        "trajectory_manifest_sha256": common._sha256(trajectory_path),
        "data_dir": source["data_dir"],
        "interventions": source["interventions"],
        "encoder_validation_pair_table": source["encoder_validation_pair_table"],
        "encoder_validation_pair_table_sha256": source["encoder_validation_pair_table_sha256"],
        "ae_selection_manifest": str(AE_ROOT / "analysis" / "selection_manifest.csv"),
        "ae_selection_manifest_sha256": common._sha256(
            AE_ROOT / "analysis" / "selection_manifest.csv"
        ),
        "pretrained_encoder_checkpoint": ae["initialization_checkpoint"],
        "pretrained_encoder_checkpoint_sha256": ae["initialization_checkpoint_sha256"],
        "pretrained_encoder_candidate": ae["selected_candidate"].zfill(3),
        "pretrained_encoder_seed": int(ae["initialization_seed"]),
        "encoder_topology": [24, 8],
        "encoder_transfer": "weights copied; AE biases dropped; all SVDD weights trainable",
        "score_audit_design": str(SCORE_AUDIT_ROOT / "design.json"),
        "score_audit_design_sha256": common._sha256(SCORE_AUDIT_ROOT / "design.json"),
        "score_audit_results": str(score_results),
        "score_audit_results_sha256": common._sha256(score_results),
        "score_audit_preregistered_hypothesis": score_design["primary_hypothesis"],
        "score_audit_selected_rule": {
            "score": "radial_d2",
            "pairing": "encoder",
            "direction": "minimize",
            "endpoint": "auprc",
            "selection_basis": (
                "largest positive non-random association in the complete prespecified grid; "
                "Holm-significant and validation/test stable"
            ),
        },
        "score_audit_selected_result": primary,
        "outcomes_inspected_before_checkpoint_freeze": False,
        "primary_strategy": PRIMARY_STRATEGY,
        "primary_score": "radial_d2",
    }
    common._write_json(output, design)
    return output


def _design(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load and authenticate the frozen SVDD design."""
    root = root.resolve()
    design = common._load(root / "design.json")
    if common._sha256(Path(design["trajectory_manifest"])) != design["trajectory_manifest_sha256"]:
        raise ValueError("SVDD trajectory manifest changed after design freeze.")
    current = common._git_commit()
    if current != design["code_commit"]:
        raise RuntimeError("Current commit differs from the frozen SVDD campaign commit.")
    return design, common._load(Path(design["trajectory_manifest"]))


def _training_command(
    root: Path, design: Mapping[str, Any], trajectory: Mapping[str, Any], attempt: Path
) -> tuple[list[str], dict[str, str]]:
    """Build one exact trainable-pretrained SVDD command."""
    run_name = f"svdd_inverse_t{int(trajectory['trajectory_index']):03d}_{attempt.name}"
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
        "experiment_name=cchamber_svdd_inverse_cap_confirmatory",
        f"run_name={run_name}",
        f"paths.base_data_dir={design['data_dir']}",
        f"paths.log_dir={root / 'logs'}",
        f"paths.checkpoints_dir={attempt / 'checkpoints'}",
        f"hydra.run.dir={attempt / 'hydra'}",
        "extras.print_config=false",
        "algorithm.pretrained_encoder_strict=false",
        "algorithm.pretrained_encoder_ckpt="
        + rank._hydra_value(design["pretrained_encoder_checkpoint"]),
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


def _encoder_delta(initial_checkpoint: Path, trained_checkpoint: Path) -> dict[str, Any]:
    """Prove that transferred encoder weights changed during SVDD training."""
    initial = torch.load(initial_checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    trained = torch.load(trained_checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    names = sorted(
        name
        for name in initial
        if name.startswith("encoder.") and not name.endswith(".bias") and name in trained
    )
    if not names:
        raise ValueError("No compatible AE encoder weights were transferred.")
    deltas = {name: float((trained[name] - initial[name]).abs().max()) for name in names}
    changed = sum(value > 0.0 for value in deltas.values())
    if changed != len(names):
        raise ValueError("One or more transferred encoder tensors remained frozen.")
    return {
        "compatible_weight_tensors": len(names),
        "changed_weight_tensors": changed,
        "maximum_absolute_delta": max(deltas.values()),
    }


def train(root: Path, trajectory_index: int) -> Path:
    """Train one SVDD trajectory and authenticate six trainable branches."""
    if "SLURM_JOB_ID" not in os.environ or not torch.cuda.is_available():
        raise RuntimeError("SVDD training requires a CUDA Slurm allocation.")
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
    for branch in branches:
        branch["encoder_delta"] = _encoder_delta(
            Path(design["pretrained_encoder_checkpoint"]), Path(branch["checkpoint"])
        )
    common._write_json(
        marker,
        {
            "schema_version": 1,
            **{key: value for key, value in trajectory.items() if key != "params"},
            "params": trajectory["params"],
            "code_commit": design["code_commit"],
            "pretrained_encoder_checkpoint": design["pretrained_encoder_checkpoint"],
            "pretrained_encoder_checkpoint_sha256": design["pretrained_encoder_checkpoint_sha256"],
            "branches": branches,
            "branch_manifest": str(branch_manifest),
            "branch_manifest_sha256": common._sha256(branch_manifest),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
    )
    return marker


def freeze(root: Path) -> Path:
    """Freeze all 288 SVDD checkpoint hashes before intervention evaluation."""
    root = root.resolve()
    design, trajectories = _design(root)
    records = []
    for trajectory in trajectories:
        marker = common._load(
            root / "training" / f"{int(trajectory['trajectory_index']):03d}.json"
        )
        for branch in marker["branches"]:
            checkpoint = Path(branch["checkpoint"])
            records.append(
                {
                    "trajectory_index": int(trajectory["trajectory_index"]),
                    "candidate_id": trajectory["candidate_id"],
                    "reporting_seed": int(trajectory["reporting_seed"]),
                    **branch,
                    "checkpoint_sha256": common._sha256(checkpoint),
                }
            )
    if len(records) != EXPECTED_TRAJECTORIES * len(rank.STRATEGIES):
        raise ValueError("SVDD checkpoint freeze coverage is incomplete.")
    output = root / "checkpoint_manifest.json"
    common._write_json(
        output,
        {
            "schema_version": 1,
            "design_sha256": common._sha256(root / "design.json"),
            "code_commit": design["code_commit"],
            "outcomes_inspected_before_freeze": False,
            "checkpoints": records,
        },
    )
    return output


class RadialMetrics(rank._SealedMetricsCallback):
    """Evaluate interventions using the native radial SVDD distance."""

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        """Collect finite radial scores."""
        del pl_module, batch, batch_idx
        name = list(trainer.test_dataloaders)[dataloader_idx]
        score = outputs[OUTPUT_NAME].detach().view(-1).cpu()
        if not torch.isfinite(score).all():
            raise ValueError(f"Non-finite radial SVDD scores for {name}.")
        self.scores[name].append(score)


def _compose(design: Mapping[str, Any], trajectory: Mapping[str, Any]):
    """Compose the SVDD model and exact sealed intervention loaders."""
    os.environ["CCHAMBER_VALID_PAIR_TABLE"] = str(design["encoder_validation_pair_table"])
    overrides = [
        f"experiment={EXPERIMENT}",
        f"seed={int(trajectory['reporting_seed'])}",
        "data.seed=314159",
        f"paths.base_data_dir={design['data_dir']}",
        f"data.signal_experiments={rank._hydra_value(design['interventions'])}",
        "logger=none",
        'algorithm.pretrained_encoder_ckpt=""',
        *[f"{name}={rank._hydra_value(value)}" for name, value in trajectory["params"].items()],
    ]
    cfg = compose_config(overrides=overrides)
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup("test")
    all_loaders = datamodule.test_dataloader()
    loaders = {"normal": all_loaders["normal"]}
    loaders.update({name: all_loaders[name] for name in design["interventions"]})
    loaders["normal"].loader = datamodule.loader
    return cfg, datamodule, loaders


def evaluate(root: Path, trajectory_index: int) -> Path:
    """Evaluate six frozen SVDD checkpoints on all 58 interventions."""
    if "SLURM_JOB_ID" not in os.environ or not torch.cuda.is_available():
        raise RuntimeError("SVDD evaluation requires a CUDA Slurm allocation.")
    root = root.resolve()
    design, trajectories = _design(root)
    trajectory = trajectories[trajectory_index]
    marker = root / "evaluation" / f"{trajectory_index:03d}.json"
    if marker.exists():
        return marker
    frozen = common._load(root / "checkpoint_manifest.json")
    branches = [
        row for row in frozen["checkpoints"] if row["trajectory_index"] == trajectory_index
    ]
    if {row["strategy"] for row in branches} != set(rank.STRATEGIES):
        raise ValueError("Frozen SVDD trajectory does not contain all strategies.")
    cfg, datamodule, loaders = _compose(design, trajectory)
    attempt = root / "attempts" / "evaluation" / f"{trajectory_index:03d}"
    attempt.mkdir(parents=True, exist_ok=True)
    outputs = []
    for branch in branches:
        checkpoint = Path(branch["checkpoint"])
        if common._sha256(checkpoint) != branch["checkpoint_sha256"]:
            raise ValueError("SVDD checkpoint changed after freeze.")
        pl.seed_everything(int(trajectory["reporting_seed"]), workers=True)
        model = hydra.utils.instantiate(cfg.algorithm)
        rank._load_checkpoint_state(model, checkpoint)
        output = attempt / f"{branch['strategy']}.csv"
        callback = RadialMetrics(
            design["interventions"],
            output,
            {
                "trajectory_index": trajectory_index,
                "model": "svdd",
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
    common._write_json(
        marker,
        {
            "schema_version": 1,
            "trajectory_index": trajectory_index,
            "candidate_id": trajectory["candidate_id"],
            "reporting_seed": int(trajectory["reporting_seed"]),
            "evaluation_rows": str(combined),
            "evaluation_rows_sha256": common._sha256(combined),
            "n_rows": len(rows),
        },
    )
    return marker


def collect(root: Path) -> Path:
    """Collect exact complete SVDD intervention results."""
    root = root.resolve()
    rows = []
    for index in range(EXPECTED_TRAJECTORIES):
        marker = common._load(root / "evaluation" / f"{index:03d}.json")
        path = Path(marker["evaluation_rows"])
        if common._sha256(path) != marker["evaluation_rows_sha256"]:
            raise ValueError("SVDD evaluation rows changed after completion.")
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    expected = EXPECTED_TRAJECTORIES * len(rank.STRATEGIES) * 58 * 2
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} rows, found {len(rows)}.")
    output = root / "results" / "evaluation_rows.csv"
    rank._write_csv(output, rows)
    return output


def main() -> None:
    """Dispatch one SVDD campaign stage."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "freeze", "collect"):
        command = subparsers.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
    for name in ("train", "evaluate"):
        command = subparsers.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--trajectory-index", type=int, required=True)
    args = parser.parse_args()
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
