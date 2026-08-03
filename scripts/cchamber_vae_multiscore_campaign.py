#!/usr/bin/env python3
"""Run the AE-initialized, multi-score Causal Chamber VAE search."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import shutil
import subprocess  # nosec B404
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytorch_lightning as pl  # noqa: E402
import torch  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from scripts import cchamber_candidate_rank_audit as rank  # noqa: E402
from src.data.utils import unpack_batch  # noqa: E402
from src.utils.pairing.io import compose_config  # noqa: E402

SOURCE_AUDIT = Path(
    "/iopsstor/scratch/cscs/vjimenez/adatl1/audits/"
    "cchamber_real_20260801_3789655_candidate_rank_3789655/audit.json"
)
AE_CAMPAIGN = Path(
    "/iopsstor/scratch/cscs/vjimenez/adatl1/campaigns/" "cchamber_ae_residual_oas_20260802_6e05000"
)
AE_INITIALIZATION_TRAJECTORY = 42
EXPERIMENT = "cchamber/vae_multiscore_search"
EXPECTED_EPOCHS = 200
REPORTING_SEEDS = (1001, 1002, 1003)
SCORES = {
    "kl_raw": "ascore/kl_raw",
    "reconstruction_mse": "ascore/reconstruction_mse",
    "residual_diagonal": "ascore/residual_diagonal",
    "residual_oas": "ascore/residual_oas",
}
SELECTORS = {
    "cap_encoder": "maximize",
    "cap_cdf": "maximize",
    "drift": "minimize",
    "wasserstein": "minimize",
}
BRANCHES = tuple(f"{score}__{selector}" for score in SCORES for selector in SELECTORS)
MONITORS = {
    **{
        f"{score}__cap_encoder": (
            f"val/summary/cap_{score}_encoder_ema_normal_vs_reference_normal"
        )
        for score in SCORES
    },
    **{
        f"{score}__cap_cdf": f"val/summary/cap_{score}_cdf_ema_normal_vs_reference_normal"
        for score in SCORES
    },
    **{f"{score}__drift": f"val/summary/{score}_operational_drift_ema" for score in SCORES},
    **{
        f"{score}__wasserstein": (f"val/summary/w1dist_{score}_ema_normal_vs_reference_normal")
        for score in SCORES
    },
}
GRID = tuple(
    {
        "algorithm.kl_scale": kl_scale,
        "algorithm.kl_warmup_frac": warmup,
        "algorithm.optimizer.lr": learning_rate,
        "algorithm.optimizer.betas": [0.9, 0.999],
        "algorithm.optimizer.weight_decay": 1.0e-6,
        "trainer.gradient_clip_val": 1.0,
    }
    for kl_scale, warmup, learning_rate in itertools.product(
        (3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3),
        (0.3, 0.5),
        (1.0e-4, 3.0e-4),
    )
)
EXPECTED_TRAJECTORIES = len(GRID) * len(REPORTING_SEEDS)


def _sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    """Serialize one JSON-compatible value deterministically."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _load(path: Path) -> Any:
    """Load a UTF-8 JSON artifact."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any, *, create: bool = False) -> None:
    """Atomically write a JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    mode = "x" if create else "w"
    with temporary.open(mode, encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if create and path.exists():
        temporary.unlink()
        raise FileExistsError(path)
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Atomically write a non-empty rectangular CSV table."""
    rows = list(rows)
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError("CSV rows do not share one ordered schema.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _git_commit(*, clean: bool) -> str:
    """Return HEAD, optionally requiring a clean worktree."""
    git = shutil.which("git")
    if git is None:
        raise FileNotFoundError("git is required for campaign provenance.")
    if clean:
        status = subprocess.run(  # nosec B603
            [git, "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if status.strip():
            raise RuntimeError("Campaign initialization requires a clean worktree.")
    return subprocess.run(  # nosec B603
        [git, "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _ae_initialization() -> dict[str, Any]:
    """Authenticate the fixed residual-OAS AE initialization checkpoint."""
    marker_path = AE_CAMPAIGN / "training" / f"{AE_INITIALIZATION_TRAJECTORY:03d}.json"
    marker = _load(marker_path)
    if marker.get("candidate_id") != "060" or int(marker.get("reporting_seed", -1)) != 1001:
        raise ValueError("Unexpected AE initialization trajectory identity.")
    matches = [row for row in marker["branches"] if row["strategy"] == "cap_cdf"]
    if len(matches) != 1:
        raise ValueError("AE initialization must contain exactly one CDF-CAP checkpoint.")
    branch = dict(matches[0])
    checkpoint = Path(branch["checkpoint"])
    if _sha256(checkpoint) != branch["checkpoint_sha256"]:
        raise ValueError("AE initialization checkpoint hash mismatch.")
    return {
        "campaign": str(AE_CAMPAIGN),
        "training_marker": str(marker_path),
        "training_marker_sha256": _sha256(marker_path),
        "candidate_id": "060",
        "seed": 1001,
        "strategy": "cap_cdf",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": branch["checkpoint_sha256"],
        "selected_epoch": int(branch["selected_epoch"]),
    }


def initialize(root: Path) -> Path:
    """Freeze the complete search grid and external provenance."""
    root = root.expanduser().resolve()
    path = root / "design.json"
    if path.exists():
        return path
    source = _load(SOURCE_AUDIT)
    initialization = _ae_initialization()
    trajectories = []
    for candidate_index, params in enumerate(GRID):
        for seed in REPORTING_SEEDS:
            trajectories.append(
                {
                    "trajectory_index": len(trajectories),
                    "candidate_id": f"{candidate_index:03d}",
                    "reporting_seed": seed,
                    "params": params,
                    "params_sha256": hashlib.sha256(_canonical(params).encode()).hexdigest(),
                    "pretrained_ae_checkpoint": initialization["checkpoint"],
                    "pretrained_ae_checkpoint_sha256": initialization["checkpoint_sha256"],
                }
            )
    if len(trajectories) != EXPECTED_TRAJECTORIES:
        raise AssertionError("Trajectory construction is incomplete.")
    trajectory_path = root / "trajectories.json"
    _write_json(trajectory_path, trajectories, create=True)
    design = {
        "schema_version": 1,
        "classification": "outcome_optimized_vae_multiscore_auto_research",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": _git_commit(clean=True),
        "experiment": EXPERIMENT,
        "expected_epochs": EXPECTED_EPOCHS,
        "expected_trajectories": EXPECTED_TRAJECTORIES,
        "expected_checkpoints": EXPECTED_TRAJECTORIES * len(BRANCHES),
        "scores": SCORES,
        "selectors": SELECTORS,
        "branches": list(BRANCHES),
        "monitors": MONITORS,
        "grid": list(GRID),
        "reporting_seeds": list(REPORTING_SEEDS),
        "trajectory_manifest": str(trajectory_path),
        "trajectory_manifest_sha256": _sha256(trajectory_path),
        "source_audit": str(SOURCE_AUDIT),
        "source_audit_sha256": _sha256(SOURCE_AUDIT),
        "data_dir": source["data_dir"],
        "interventions": source["interventions"],
        "encoder_validation_pair_table": source["encoder_validation_pair_table"],
        "encoder_validation_pair_table_sha256": source["encoder_validation_pair_table_sha256"],
        "ae_initialization": initialization,
        "selection_policy": {
            "performance_endpoint": "mean intervention AUPRC after seed-first averaging",
            "high_performance_rule": (
                "CAP result within one standard error of the global best selected result"
            ),
            "optimization": (
                "Among eligible CAP results, maximize CAP AUPRC minus the best drift/"
                "Wasserstein AUPRC for the same anomaly score"
            ),
            "fallback": "If no CAP result is eligible, select the highest-AUPRC CAP result.",
        },
        "intervention_outcomes_inspected_before_checkpoint_freeze": False,
    }
    _write_json(path, design, create=True)
    return path


def _design(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load and authenticate one initialized campaign."""
    root = root.expanduser().resolve()
    design = _load(root / "design.json")
    if _git_commit(clean=False) != design["code_commit"]:
        raise RuntimeError("Current code commit differs from the frozen campaign commit.")
    trajectory_path = Path(design["trajectory_manifest"])
    if _sha256(trajectory_path) != design["trajectory_manifest_sha256"]:
        raise ValueError("Trajectory manifest changed after campaign initialization.")
    checkpoint = Path(design["ae_initialization"]["checkpoint"])
    if _sha256(checkpoint) != design["ae_initialization"]["checkpoint_sha256"]:
        raise ValueError("AE initialization changed after campaign initialization.")
    return design, _load(trajectory_path)


def _training_command(
    root: Path,
    design: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    attempt: Path,
    *,
    epochs: int,
) -> tuple[list[str], dict[str, str]]:
    """Build one exact shared-trajectory training command."""
    run_name = f"vae_multi_t{int(trajectory['trajectory_index']):03d}_{attempt.name}"
    command = [
        sys.executable,
        "src/train.py",
        f"experiment={EXPERIMENT}",
        "trainer=gpu",
        "trainer.devices=[0]",
        "trainer.deterministic=true",
        f"trainer.min_epochs={epochs}",
        f"trainer.max_epochs={epochs}",
        "trainer.num_sanity_val_steps=0",
        f"seed={int(trajectory['reporting_seed'])}",
        "data.seed=314159",
        f"data.train_seed={int(trajectory['reporting_seed'])}",
        "data.signal_experiments=[]",
        "train=true",
        "test=false",
        "logger=none",
        "experiment_name=cchamber_vae_multiscore_search",
        f"run_name={run_name}",
        f"paths.base_data_dir={design['data_dir']}",
        f"paths.log_dir={root / 'logs'}",
        f"paths.checkpoints_dir={attempt / 'checkpoints'}",
        f"hydra.run.dir={attempt / 'hydra'}",
        "extras.print_config=false",
        f"algorithm.pretrained_ae_ckpt={trajectory['pretrained_ae_checkpoint']}",
        *[f"{name}={rank._hydra_value(value)}" for name, value in trajectory["params"].items()],
    ]
    if epochs < EXPECTED_EPOCHS:
        command.append("+trainer.limit_train_batches=2")
    environment = os.environ.copy()
    environment["CCHAMBER_VALID_PAIR_TABLE"] = str(design["encoder_validation_pair_table"])
    environment["CCHAMBER_AUDIT_CHECKPOINT_MANIFEST"] = str(attempt / "checkpoint_branches.json")
    environment["CCHAMBER_AUDIT_TRAJECTORY_FINGERPRINT"] = str(
        attempt / "trajectory_fingerprint.json"
    )
    environment["LOG_DIR"] = str(root / "logs")
    return command, environment


def _validate_branches(path: Path, *, expected_epochs: int) -> list[dict[str, Any]]:
    """Validate exact branch coverage and checkpoint identities."""
    manifest = _load(path)
    branches = manifest.get("branches")
    if not isinstance(branches, list) or len(branches) != len(BRANCHES):
        raise ValueError("Checkpoint branch coverage is incomplete.")
    by_name = {str(row["strategy"]): dict(row) for row in branches}
    if set(by_name) != set(BRANCHES):
        raise ValueError("Checkpoint branch identities differ from the design.")
    output = []
    for name in BRANCHES:
        row = by_name[name]
        checkpoint = Path(row["checkpoint"])
        if row["monitor"] != MONITORS[name] or _sha256(checkpoint) != row["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint branch failed authentication: {name}")
        if not 0 <= int(row["selected_epoch"]) < expected_epochs:
            raise ValueError(f"Invalid selected epoch for {name}.")
        if not math.isfinite(float(row["monitor_value"])):
            raise ValueError(f"Non-finite monitor for {name}.")
        output.append(row)
    return output


def canary(root: Path) -> Path:
    """Run a two-epoch GPU canary through all score and checkpoint callbacks."""
    if "SLURM_JOB_ID" not in os.environ or not torch.cuda.is_available():
        raise RuntimeError("The canary requires a CUDA Slurm allocation.")
    root = root.expanduser().resolve()
    design, trajectories = _design(root)
    marker = root / "canary.json"
    if marker.exists():
        return marker
    attempt = root / "attempts" / "canary" / rank._attempt_id()
    command, environment = _training_command(root, design, trajectories[0], attempt, epochs=2)
    subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)  # nosec B603
    branches = _validate_branches(attempt / "checkpoint_branches.json", expected_epochs=2)
    fingerprint = attempt / "trajectory_fingerprint.json"
    value = {
        "schema_version": 1,
        "code_commit": design["code_commit"],
        "branches": branches,
        "fingerprint": str(fingerprint),
        "fingerprint_sha256": _sha256(fingerprint),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    _write_json(marker, value, create=True)
    return marker


def train(root: Path, trajectory_index: int) -> Path:
    """Train or resume one full multi-score trajectory."""
    if "SLURM_JOB_ID" not in os.environ or not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA Slurm allocation.")
    root = root.expanduser().resolve()
    design, trajectories = _design(root)
    if not (root / "canary.json").is_file():
        raise FileNotFoundError("The campaign canary has not completed.")
    trajectory = trajectories[int(trajectory_index)]
    marker = root / "training" / f"{int(trajectory_index):03d}.json"
    if marker.exists():
        return marker
    attempt = root / "attempts" / "training" / f"{int(trajectory_index):03d}" / rank._attempt_id()
    command, environment = _training_command(
        root, design, trajectory, attempt, epochs=EXPECTED_EPOCHS
    )
    subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)  # nosec B603
    branch_path = attempt / "checkpoint_branches.json"
    branches = _validate_branches(branch_path, expected_epochs=EXPECTED_EPOCHS)
    fingerprint = attempt / "trajectory_fingerprint.json"
    fingerprint_data = _load(fingerprint)
    if len(fingerprint_data.get("epochs", [])) != EXPECTED_EPOCHS:
        raise ValueError("Trajectory fingerprint does not contain every epoch.")
    value = {
        "schema_version": 1,
        **trajectory,
        "code_commit": design["code_commit"],
        "branches": branches,
        "branch_manifest": str(branch_path),
        "branch_manifest_sha256": _sha256(branch_path),
        "trajectory_fingerprint": str(fingerprint),
        "trajectory_fingerprint_sha256": _sha256(fingerprint),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    _write_json(marker, value, create=True)
    return marker


def freeze(root: Path) -> Path:
    """Freeze all score-selector checkpoint hashes before intervention evaluation."""
    root = root.expanduser().resolve()
    design, trajectories = _design(root)
    path = root / "checkpoint_manifest.json"
    if path.exists():
        manifest = _load(path)
        records = manifest.get("checkpoints", [])
        if len(records) != int(design["expected_checkpoints"]):
            raise ValueError("Existing checkpoint freeze coverage is incomplete.")
        for record in records:
            if _sha256(Path(record["checkpoint"])) != record["checkpoint_sha256"]:
                raise ValueError("Existing checkpoint freeze failed authentication.")
        return path
    records = []
    for trajectory in trajectories:
        marker = _load(root / "training" / f"{int(trajectory['trajectory_index']):03d}.json")
        for branch in marker["branches"]:
            checkpoint = Path(branch["checkpoint"])
            if _sha256(checkpoint) != branch["checkpoint_sha256"]:
                raise ValueError("A trained checkpoint changed before the freeze.")
            score, selector = str(branch["strategy"]).split("__", maxsplit=1)
            records.append(
                {
                    "trajectory_index": int(trajectory["trajectory_index"]),
                    "candidate_id": trajectory["candidate_id"],
                    "reporting_seed": int(trajectory["reporting_seed"]),
                    "score": score,
                    "selector": selector,
                    **branch,
                }
            )
    if len(records) != int(design["expected_checkpoints"]):
        raise ValueError("Checkpoint freeze coverage is incomplete.")
    _write_json(
        path,
        {
            "schema_version": 1,
            "design_sha256": _sha256(root / "design.json"),
            "code_commit": design["code_commit"],
            "outcomes_inspected_before_freeze": False,
            "checkpoints": records,
        },
        create=True,
    )
    return path


def _compose(design: Mapping[str, Any], trajectory: Mapping[str, Any]):
    """Compose one exact VAE and all validation/test streams."""
    os.environ["CCHAMBER_VALID_PAIR_TABLE"] = str(design["encoder_validation_pair_table"])
    overrides = [
        f"experiment={EXPERIMENT}",
        f"seed={int(trajectory['reporting_seed'])}",
        "data.seed=314159",
        f"data.train_seed={int(trajectory['reporting_seed'])}",
        f"paths.base_data_dir={design['data_dir']}",
        f"data.signal_experiments={rank._hydra_value(design['interventions'])}",
        "logger=none",
        f"algorithm.pretrained_ae_ckpt={trajectory['pretrained_ae_checkpoint']}",
        *[f"{name}={rank._hydra_value(value)}" for name, value in trajectory["params"].items()],
    ]
    cfg = compose_config(overrides=overrides)
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup(None)
    model = hydra.utils.instantiate(cfg.algorithm)
    return datamodule, model


@torch.inference_mode()
def _scores(model, loader, score_name: str, device: torch.device) -> np.ndarray:
    """Evaluate one deterministic score without stochastic decoder sampling."""
    chunks = []
    model.eval().to(device)
    for batch in loader:
        x = torch.flatten(unpack_batch(batch).x, start_dim=1).to(device)
        model_input = model.features(x)
        with model._keras_device_scope(model_input.device):
            mean, log_variance, _ = model.encoder(model_input)
            reconstruction = model.decoder(mean)
        if score_name == "kl_raw":
            score = 0.5 * torch.sum(
                torch.square(mean) + torch.exp(log_variance) - log_variance - 1.0,
                dim=1,
            )
        else:
            mse, diagonal, oas = model.residual_scores(x, reconstruction)
            score = {
                "reconstruction_mse": mse,
                "residual_diagonal": diagonal,
                "residual_oas": oas,
            }[score_name]
        if score.ndim != 1 or not torch.isfinite(score).all():
            raise ValueError(f"Invalid {score_name} score tensor.")
        chunks.append(score.cpu())
    if not chunks:
        raise RuntimeError("Score evaluation received an empty loader.")
    return torch.cat(chunks).numpy().astype(np.float64)


def evaluate(root: Path, trajectory_index: int) -> Path:
    """Evaluate every frozen score-selector checkpoint for one trajectory."""
    if "SLURM_JOB_ID" not in os.environ or not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires a CUDA Slurm allocation.")
    root = root.expanduser().resolve()
    design, trajectories = _design(root)
    trajectory = trajectories[int(trajectory_index)]
    marker = root / "evaluation" / f"{int(trajectory_index):03d}.json"
    if marker.exists():
        return marker
    manifest = _load(root / "checkpoint_manifest.json")
    if manifest.get("outcomes_inspected_before_freeze") is not False:
        raise ValueError("Checkpoint freeze did not precede intervention evaluation.")
    branches = [
        row
        for row in manifest["checkpoints"]
        if int(row["trajectory_index"]) == int(trajectory_index)
    ]
    if {row["strategy"] for row in branches} != set(BRANCHES):
        raise ValueError("Frozen evaluation branch coverage is incomplete.")
    datamodule, model = _compose(design, trajectory)
    validation = datamodule.val_dataloader()
    test = datamodule.test_dataloader()
    device = torch.device("cuda")
    rows = []
    for branch in branches:
        checkpoint = Path(branch["checkpoint"])
        if _sha256(checkpoint) != branch["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed after the outcome freeze.")
        rank._load_checkpoint_state(model, checkpoint)
        score_name = branch["score"]
        validation_normal = _scores(model, validation["normal"], score_name, device)
        threshold = float(np.quantile(validation_normal, 0.99))
        normal = _scores(model, test["normal"], score_name, device)
        for intervention in design["interventions"]:
            signal = _scores(model, test[intervention], score_name, device)
            target = np.concatenate(
                [np.zeros(len(normal), dtype=int), np.ones(len(signal), dtype=int)]
            )
            prediction = np.concatenate([normal, signal])
            for metric, value in (
                ("auprc", average_precision_score(target, prediction)),
                ("efficiency_operational", np.mean(signal >= threshold)),
            ):
                rows.append(
                    {
                        "trajectory_index": int(trajectory_index),
                        "candidate_id": trajectory["candidate_id"],
                        "reporting_seed": int(trajectory["reporting_seed"]),
                        "score": score_name,
                        "selector": branch["selector"],
                        "strategy": branch["strategy"],
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": branch["checkpoint_sha256"],
                        "selected_epoch": int(branch["selected_epoch"]),
                        "monitor": branch["monitor"],
                        "monitor_value": float(branch["monitor_value"]),
                        "intervention": intervention,
                        "metric": metric,
                        "value": float(value),
                        "validation_derived_threshold": threshold,
                    }
                )
    datamodule.teardown(None)
    output = root / "evaluation" / f"{int(trajectory_index):03d}.csv"
    _write_csv(output, rows)
    expected = len(BRANCHES) * len(design["interventions"]) * 2
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} evaluation rows, found {len(rows)}.")
    _write_json(
        marker,
        {
            "schema_version": 1,
            "trajectory_index": int(trajectory_index),
            "rows": str(output),
            "rows_sha256": _sha256(output),
            "n_rows": len(rows),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
        create=True,
    )
    return marker


def collect(root: Path) -> tuple[Path, Path]:
    """Collect complete authenticated checkpoint and outcome tables."""
    root = root.expanduser().resolve()
    design, _ = _design(root)
    outcome_rows = []
    for index in range(EXPECTED_TRAJECTORIES):
        marker = _load(root / "evaluation" / f"{index:03d}.json")
        path = Path(marker["rows"])
        if _sha256(path) != marker["rows_sha256"]:
            raise ValueError(f"Evaluation artifact changed: {path}")
        with path.open("r", encoding="utf-8", newline="") as handle:
            outcome_rows.extend(csv.DictReader(handle))
    expected = EXPECTED_TRAJECTORIES * len(BRANCHES) * len(design["interventions"]) * 2
    if len(outcome_rows) != expected:
        raise ValueError(f"Expected {expected} collected rows, found {len(outcome_rows)}.")
    outcome_path = root / "results" / "evaluation_rows.csv"
    _write_csv(outcome_path, outcome_rows)
    checkpoint_rows = _load(root / "checkpoint_manifest.json")["checkpoints"]
    checkpoint_path = root / "results" / "checkpoint_rows.csv"
    _write_csv(checkpoint_path, checkpoint_rows)
    return checkpoint_path, outcome_path


def analyze(root: Path) -> tuple[Path, Path, Path]:
    """Select high-performing configurations and maximize empirical CAP advantage."""
    root = root.expanduser().resolve()
    design, _ = _design(root)
    checkpoints = pd.read_csv(
        root / "results" / "checkpoint_rows.csv", dtype={"candidate_id": str}
    )
    outcomes = pd.read_csv(root / "results" / "evaluation_rows.csv", dtype={"candidate_id": str})
    checkpoints["candidate_id"] = checkpoints["candidate_id"].str.zfill(3)
    outcomes["candidate_id"] = outcomes["candidate_id"].str.zfill(3)
    seed_first = (
        outcomes.groupby(
            ["candidate_id", "reporting_seed", "score", "selector", "metric"],
            sort=True,
        )["value"]
        .mean()
        .reset_index()
    )
    selected_rows = []
    rank_rows = []
    for score in SCORES:
        for selector, direction in SELECTORS.items():
            branch = checkpoints[(checkpoints.score == score) & (checkpoints.selector == selector)]
            candidate_proxy = branch.groupby("candidate_id").monitor_value.mean()
            candidate = (
                str(candidate_proxy.idxmax())
                if direction == "maximize"
                else str(candidate_proxy.idxmin())
            )
            for metric in ("auprc", "efficiency_operational"):
                values = seed_first[
                    (seed_first.score == score)
                    & (seed_first.selector == selector)
                    & (seed_first.metric == metric)
                ]
                candidate_outcome = values.groupby("candidate_id").value.mean()
                selection_proxy = candidate_proxy if direction == "maximize" else -candidate_proxy
                aligned = selection_proxy.to_frame("proxy").join(
                    candidate_outcome.rename("outcome"), how="inner"
                )
                rank_rows.append(
                    {
                        "score": score,
                        "selector": selector,
                        "metric": metric,
                        "spearman_rho": float(
                            stats.spearmanr(aligned.proxy, aligned.outcome).statistic
                        ),
                        "n_candidates": len(aligned),
                    }
                )
                selected = values[values.candidate_id == candidate].value.to_numpy(float)
                mean = float(selected.mean())
                sd = float(selected.std(ddof=1))
                selected_rows.append(
                    {
                        "score": score,
                        "selector": selector,
                        "direction": direction,
                        "selected_candidate": candidate,
                        "selected_candidate_proxy": float(candidate_proxy.loc[candidate]),
                        "metric": metric,
                        "mean": mean,
                        "sd": sd,
                        "se": sd / math.sqrt(len(selected)),
                        "n_seeds": len(selected),
                    }
                )
    summary = pd.DataFrame(selected_rows)
    auprc = summary[summary.metric == "auprc"].copy()
    best_all = auprc.loc[auprc["mean"].idxmax()]
    threshold = float(best_all["mean"] - best_all["se"])
    comparisons = []
    for score in SCORES:
        score_rows = auprc[auprc.score == score]
        non_cap = score_rows[score_rows.selector.isin(["drift", "wasserstein"])]
        non_cap_best = non_cap.loc[non_cap["mean"].idxmax()]
        for _, cap in score_rows[score_rows.selector.str.startswith("cap_")].iterrows():
            comparisons.append(
                {
                    "score": score,
                    "cap_selector": cap["selector"],
                    "cap_candidate": cap["selected_candidate"],
                    "cap_auprc": float(cap["mean"]),
                    "cap_se": float(cap["se"]),
                    "best_noncap_selector": non_cap_best["selector"],
                    "best_noncap_auprc": float(non_cap_best["mean"]),
                    "cap_advantage": float(cap["mean"] - non_cap_best["mean"]),
                    "high_performance": bool(float(cap["mean"]) >= threshold),
                }
            )
    comparison = pd.DataFrame(comparisons)
    eligible = comparison[comparison.high_performance]
    fallback = eligible.empty
    if fallback:
        selected = comparison.loc[comparison.cap_auprc.idxmax()]
    else:
        selected = eligible.loc[eligible.cap_advantage.idxmax()]
    candidate = str(selected["cap_candidate"])
    trajectories = _load(Path(design["trajectory_manifest"]))
    params = next(row["params"] for row in trajectories if row["candidate_id"] == candidate)
    selection = {
        "schema_version": 1,
        "selection_is_outcome_optimized": True,
        "global_best_selected_result": best_all.to_dict(),
        "high_performance_threshold": threshold,
        "fallback_used": bool(fallback),
        "selected_score": selected["score"],
        "selected_cap_selector": selected["cap_selector"],
        "selected_candidate": candidate,
        "selected_params": params,
        "selected_cap_auprc": float(selected["cap_auprc"]),
        "best_same_score_noncap_selector": selected["best_noncap_selector"],
        "best_same_score_noncap_auprc": float(selected["best_noncap_auprc"]),
        "cap_advantage": float(selected["cap_advantage"]),
    }
    summary_path = root / "analysis" / "selected_performance.csv"
    rank_path = root / "analysis" / "candidate_rank_associations.csv"
    comparison_path = root / "analysis" / "cap_advantage.csv"
    selection_path = root / "analysis" / "selection.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(rank_rows).to_csv(rank_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    _write_json(selection_path, selection)
    return summary_path, comparison_path, selection_path


def _parser() -> argparse.ArgumentParser:
    """Build the stage-oriented command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "canary", "freeze", "collect", "analyze"):
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
    elif args.command == "canary":
        print(canary(args.root))
    elif args.command == "train":
        print(train(args.root, args.trajectory_index))
    elif args.command == "freeze":
        print(freeze(args.root))
    elif args.command == "evaluate":
        print(evaluate(args.root, args.trajectory_index))
    elif args.command == "collect":
        print(*collect(args.root), sep="\n")
    else:
        print(*analyze(args.root), sep="\n")


if __name__ == "__main__":
    main()
