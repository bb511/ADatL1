#!/usr/bin/env python3
"""Outcome-gated audit of AE score constructions for Causal Chamber CAP.

The audit deliberately holds the candidate panel and checkpoint branch fixed.  It first extracts
normal-only scores and freezes all CAP proxies.  Intervention scores cannot be computed until that
freeze exists.  This makes the comparison useful as a diagnostic of the AE score representation
without selecting a favorable result after inspecting the new intervention outcomes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.covariance import OAS
from sklearn.metrics import average_precision_score

from scripts import cchamber_candidate_rank_audit as rank_audit
from src.callbacks.metrics.cap.metric import ApproximationCapacity
from src.utils.pairing.io import compose_config
from src.utils.pairing.table import load_pair_table, sha256_tensor

SCHEMA_VERSION = 1
MODEL = "ae"
BRANCH = "cap_metadata_nearest"
SCORE_NAMES = (
    "mse",
    "huber_matched",
    "residual_diagonal",
    "residual_oas",
    "latent_oas",
)
PAIRING_NAMES = ("metadata", "encoder", "cdf", "random")
SPLITS = ("valid", "test")
PAIRING_SEED = 271828
TARGET_FPR = 0.01
CAP_CONFIG = {
    "beta0": 1.0,
    "normalization_type": "sigmoid",
    "normalization_params": None,
    "energy_type": "adaptive",
    "energy_params": {"scale": 0.5},
    "regularization_type": "none",
    "regularization_params": None,
    "binary": True,
    "lr": 0.01,
    "n_epochs": 20,
    "batch_size": 512,
    "normalize_gradients": True,
}


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    """Serialize a JSON value deterministically."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _atomic_json(path: Path, value: Mapping[str, Any], *, create: bool = False) -> None:
    """Write one JSON mapping atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if create and path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if create and path.exists():
        temporary.unlink()
        raise FileExistsError(path)
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a non-empty sequence of rows atomically."""
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _load_json(path: Path) -> Any:
    """Load one required JSON artifact."""
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_design(path: Path) -> dict[str, Any]:
    """Load a design and require the executing audit code to remain unchanged."""
    design = _load_json(path)
    if design.get("audit_script_sha256") != _sha256(Path(__file__).resolve()):
        raise ValueError("AE score-audit code changed after the design freeze.")
    return design


def _require_gpu() -> None:
    """Require execution on a Slurm GPU allocation."""
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Score inference must run in a Slurm allocation.")
    if not torch.cuda.is_available():
        raise RuntimeError("Score inference requires a visible CUDA device.")


def _hydra_value(value: Any) -> str:
    """Render a JSON-compatible Hydra override value."""
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "null"
    return json.dumps(value, separators=(",", ":"))


def _audit_inputs(
    audit_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, dict]]:
    """Authenticate and return the frozen AE panel inputs."""
    audit_root = audit_root.expanduser().resolve()
    audit_path = audit_root / "audit.json"
    checkpoint_path = audit_root / "checkpoint_manifest.json"
    audit = _load_json(audit_path)
    trajectories = _load_json(Path(audit["trajectory_manifest"]))
    checkpoint_manifest = _load_json(checkpoint_path)
    if checkpoint_manifest.get("audit_sha256") != _sha256(audit_path):
        raise ValueError("Candidate-rank checkpoint manifest does not authenticate the audit.")
    if checkpoint_manifest.get("outcomes_inspected_before_freeze") is not False:
        raise ValueError("Candidate-rank checkpoints were not outcome-blind at freeze time.")
    ae_trajectories = [row for row in trajectories if row["model"] == MODEL]
    if len(ae_trajectories) != 48:
        raise ValueError(f"Expected 48 AE trajectories, found {len(ae_trajectories)}.")
    records = {
        int(row["trajectory_index"]): dict(row)
        for row in checkpoint_manifest["checkpoints"]
        if row["model"] == MODEL and row["strategy"] == BRANCH
    }
    if set(records) != {int(row["trajectory_index"]) for row in ae_trajectories}:
        raise ValueError("AE metadata-CAP checkpoint coverage is not exact.")
    return audit, ae_trajectories, records


def create_design(audit_root: Path, output_root: Path) -> Path:
    """Freeze the complete score comparison before new outcomes exist."""
    audit_root = audit_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    audit, trajectories, checkpoints = _audit_inputs(audit_root)
    design = {
        "schema_version": SCHEMA_VERSION,
        "classification": "post_confirmatory_ae_score_diagnostic",
        "claim_boundary": (
            "The score families were fixed before their intervention outcomes were computed. "
            "The study is diagnostic because the original scalar-MSE outcomes were already known."
        ),
        "audit": str(audit_root / "audit.json"),
        "audit_sha256": _sha256(audit_root / "audit.json"),
        "checkpoint_manifest": str(audit_root / "checkpoint_manifest.json"),
        "checkpoint_manifest_sha256": _sha256(audit_root / "checkpoint_manifest.json"),
        "audit_script": str(Path(__file__).resolve()),
        "audit_script_sha256": _sha256(Path(__file__).resolve()),
        "campaign_id": audit["campaign_id"],
        "model": MODEL,
        "fixed_checkpoint_branch": BRANCH,
        "trajectory_indices": [int(row["trajectory_index"]) for row in trajectories],
        "candidate_ids": sorted({str(row["candidate_id"]) for row in trajectories}),
        "reporting_seeds": sorted({int(row["reporting_seed"]) for row in trajectories}),
        "score_families": {
            "mse": "Mean squared reconstruction residual; exact production control.",
            "huber_matched": "Per-event SmoothL1 score using the candidate's training delta.",
            "residual_diagonal": (
                "Squared signed residual standardized by training-normal feature variances."
            ),
            "residual_oas": (
                "Signed-residual Mahalanobis energy with OAS covariance fitted on training normal."
            ),
            "latent_oas": (
                "Encoder-latent Mahalanobis energy with OAS covariance fitted on training normal."
            ),
        },
        "score_names": list(SCORE_NAMES),
        "pairing_names": list(PAIRING_NAMES),
        "splits": list(SPLITS),
        "cap_config": CAP_CONFIG,
        "pairing_seed": PAIRING_SEED,
        "target_fpr": TARGET_FPR,
        "primary_hypothesis": {
            "score": "residual_oas",
            "pairing": "encoder",
            "metric": "auprc",
            "direction": "positive",
            "rationale": (
                "The current scalar MSE discards correlated residual direction; OAS provides a "
                "fixed label-free covariance correction and encoder pairing is independent of "
                "the candidate AE."
            ),
        },
        "complete_reporting": (
            "Every score x pairing combination is reported; random pairing is a negative control."
        ),
        "multiplicity": "Holm correction across all non-control score x pairing tests per endpoint.",
        "checkpoint_hashes": {
            str(index): row["checkpoint_sha256"] for index, row in sorted(checkpoints.items())
        },
        "new_score_outcomes_computed_at_design": False,
    }
    design["design_identity_sha256"] = hashlib.sha256(
        _canonical_json(design).encode("utf-8")
    ).hexdigest()
    path = output_root / "design.json"
    _atomic_json(path, design, create=True)
    return path


def _compose(
    audit: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    *,
    signals: Sequence[str],
):
    """Compose one frozen AE trajectory and its requested data streams."""
    os.environ["CCHAMBER_VALID_PAIR_TABLE"] = str(audit["encoder_validation_pair_table"])
    overrides = [
        "experiment=cchamber/ae_candidate_rank_audit",
        f"seed={int(trajectory['reporting_seed'])}",
        "data.seed=314159",
        f"paths.base_data_dir={audit['data_dir']}",
        f"data.signal_experiments={_hydra_value(list(signals))}",
        "logger=none",
        *[f"{name}={_hydra_value(value)}" for name, value in trajectory["params"].items()],
    ]
    cfg = compose_config(overrides=overrides)
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup(None)
    model = hydra.utils.instantiate(cfg.algorithm)
    return datamodule, model


@torch.inference_mode()
def _representations(
    model, loader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return inputs, signed residuals, and latent vectors for one loader."""
    xs, residuals, latents = [], [], []
    model.eval().to(device)
    for batch in loader:
        x = torch.flatten(batch["x"], start_dim=1).to(device)
        z, reconstruction = model.forward(x)
        xs.append(x.cpu())
        residuals.append((x - reconstruction).cpu())
        latents.append(z.cpu())
    return tuple(
        torch.cat(values).numpy().astype(np.float64) for values in (xs, residuals, latents)
    )


def _fit_score_state(
    residual: np.ndarray, latent: np.ndarray, delta: float
) -> dict[str, np.ndarray]:
    """Fit label-free score parameters on training-normal representations."""
    if residual.ndim != 2 or latent.ndim != 2 or len(residual) != len(latent):
        raise ValueError("Training residual and latent arrays must be aligned matrices.")
    residual_oas = OAS(store_precision=True, assume_centered=False).fit(residual)
    latent_oas = OAS(store_precision=True, assume_centered=False).fit(latent)
    variance = residual.var(axis=0, ddof=0)
    variance = np.maximum(variance, np.finfo(np.float64).eps)
    return {
        "delta": np.asarray(float(delta)),
        "residual_mean": residual.mean(axis=0),
        "residual_variance": variance,
        "residual_oas_location": residual_oas.location_,
        "residual_oas_precision": residual_oas.precision_,
        "latent_oas_location": latent_oas.location_,
        "latent_oas_precision": latent_oas.precision_,
    }


def _mahalanobis(values: np.ndarray, location: np.ndarray, precision: np.ndarray) -> np.ndarray:
    """Compute dimension-normalized squared Mahalanobis energy."""
    centered = values - location
    return np.einsum("bi,ij,bj->b", centered, precision, centered) / values.shape[1]


def score_arrays(
    residual: np.ndarray,
    latent: np.ndarray,
    state: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Construct every prespecified scalar AE score."""
    absolute = np.abs(residual)
    delta = float(np.asarray(state["delta"]))
    huber = np.where(
        absolute < delta,
        0.5 * residual**2 / delta,
        absolute - 0.5 * delta,
    )
    scores = {
        "mse": np.mean(residual**2, axis=1),
        "huber_matched": np.mean(huber, axis=1),
        "residual_diagonal": np.mean(
            (residual - state["residual_mean"]) ** 2 / state["residual_variance"], axis=1
        ),
        "residual_oas": _mahalanobis(
            residual, state["residual_oas_location"], state["residual_oas_precision"]
        ),
        "latent_oas": _mahalanobis(
            latent, state["latent_oas_location"], state["latent_oas_precision"]
        ),
    }
    if tuple(scores) != SCORE_NAMES or any(
        values.ndim != 1 or not np.isfinite(values).all() for values in scores.values()
    ):
        raise ValueError("AE score construction produced an invalid score vector.")
    return scores


def extract_normal(audit_root: Path, output_root: Path, trajectory_index: int) -> Path:
    """Extract train-fitted validation and test normal scores."""
    _require_gpu()
    audit, trajectories, checkpoints = _audit_inputs(audit_root)
    trajectory_by_index = {int(row["trajectory_index"]): row for row in trajectories}
    if int(trajectory_index) not in trajectory_by_index:
        raise IndexError("Trajectory is not in the fixed AE panel.")
    trajectory = trajectory_by_index[int(trajectory_index)]
    checkpoint = checkpoints[int(trajectory_index)]
    checkpoint_path = Path(checkpoint["checkpoint"])
    if _sha256(checkpoint_path) != checkpoint["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint hash mismatch.")
    output_root = output_root.expanduser().resolve()
    design_path = output_root / "design.json"
    design = _load_design(design_path)
    if design["audit_sha256"] != _sha256(audit_root / "audit.json"):
        raise ValueError("Design/audit identity mismatch.")
    destination = output_root / "normal" / f"{int(trajectory_index):03d}.npz"
    marker_path = destination.with_suffix(".json")
    if destination.exists() and marker_path.exists():
        marker = _load_json(marker_path)
        if marker.get("artifact_sha256") == _sha256(destination):
            return destination
        raise ValueError("Existing normal artifact failed its hash check.")

    datamodule, model = _compose(audit, trajectory, signals=[])
    rank_audit._load_checkpoint_state(model, checkpoint_path)
    device = torch.device("cuda")
    _, train_residual, train_latent = _representations(
        model, datamodule.train_dataloader(), device
    )
    state = _fit_score_state(
        train_residual,
        train_latent,
        float(trajectory["params"]["algorithm.delta"]),
    )
    arrays: dict[str, np.ndarray] = {f"state_{name}": value for name, value in state.items()}
    for split, loaders in (
        ("valid", datamodule.val_dataloader()),
        ("test", datamodule.test_dataloader()),
    ):
        for stream in ("normal", "reference_normal"):
            x, residual, latent = _representations(model, loaders[stream], device)
            arrays[f"{split}_{stream}_x"] = x
            for score_name, values in score_arrays(residual, latent, state).items():
                arrays[f"{split}_{stream}_{score_name}"] = values
    datamodule.teardown(None)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, destination)
    marker = {
        "schema_version": SCHEMA_VERSION,
        "design_sha256": _sha256(design_path),
        "trajectory_index": int(trajectory_index),
        "candidate_id": str(trajectory["candidate_id"]),
        "reporting_seed": int(trajectory["reporting_seed"]),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "selected_epoch": int(checkpoint["selected_epoch"]),
        "artifact": str(destination),
        "artifact_sha256": _sha256(destination),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "gpu": torch.cuda.get_device_name(0),
    }
    _atomic_json(marker_path, marker, create=True)
    return destination


def _pair_indices(
    pairing: str,
    score_1: np.ndarray,
    score_2: np.ndarray,
    *,
    encoder_table: Path | None,
    split: str,
    x_1: np.ndarray,
    x_2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic indices for one prespecified pairing rule."""
    n = min(len(score_1), len(score_2))
    if pairing == "metadata":
        return np.arange(n), np.arange(n)
    if pairing == "random":
        generator = np.random.default_rng(PAIRING_SEED)
        return np.arange(n), generator.permutation(len(score_2))[:n]
    if pairing == "cdf":
        left = np.argsort(score_1)
        right = np.argsort(score_2)
        ranks = np.rint(np.linspace(0, len(score_2) - 1, len(score_1))).astype(int)
        return left, right[ranks]
    if pairing != "encoder" or encoder_table is None:
        raise ValueError(f"Unsupported pairing: {pairing}")
    table = load_pair_table(
        encoder_table,
        expected_dataset_1="normal",
        expected_dataset_2="reference_normal",
        expected_split="validate" if split == "valid" else split,
        n_dataset_1=len(score_1),
        n_dataset_2=len(score_2),
        source_1_sha256=sha256_tensor(torch.as_tensor(x_1, dtype=torch.float32)),
        source_2_sha256=sha256_tensor(torch.as_tensor(x_2, dtype=torch.float32)),
    )
    return table["idx_1"].numpy(), table["idx_2"].numpy()


def _cap(left: np.ndarray, right: np.ndarray) -> float:
    """Evaluate production CAP on one aligned score pair."""
    metric = ApproximationCapacity(**CAP_CONFIG, device="cpu")
    metric.update(
        torch.as_tensor(left, dtype=torch.float32), torch.as_tensor(right, dtype=torch.float32)
    )
    return float(metric.compute())


def freeze_proxies(audit_root: Path, output_root: Path) -> tuple[Path, Path]:
    """Compute and freeze every normal-only proxy before intervention inference."""
    audit, trajectories, _ = _audit_inputs(audit_root)
    output_root = output_root.expanduser().resolve()
    design_path = output_root / "design.json"
    design = _load_design(design_path)
    if design.get("new_score_outcomes_computed_at_design") is not False:
        raise ValueError("The design did not preserve the outcome gate.")
    rows = []
    sources = []
    for trajectory in trajectories:
        index = int(trajectory["trajectory_index"])
        artifact = output_root / "normal" / f"{index:03d}.npz"
        marker_path = artifact.with_suffix(".json")
        marker = _load_json(marker_path)
        if marker["artifact_sha256"] != _sha256(artifact):
            raise ValueError(f"Normal artifact hash mismatch: {artifact}")
        arrays = np.load(artifact)
        sources.append({"trajectory_index": index, "sha256": _sha256(artifact)})
        for split in SPLITS:
            x_1 = arrays[f"{split}_normal_x"]
            x_2 = arrays[f"{split}_reference_normal_x"]
            if split == "valid":
                encoder_table = Path(audit["encoder_validation_pair_table"])
            else:
                pairing_manifest = _load_json(Path(audit["pairing_manifest"]))
                encoder_table = Path(pairing_manifest["primary_test_table"])
            for score_name in SCORE_NAMES:
                score_1 = arrays[f"{split}_normal_{score_name}"]
                score_2 = arrays[f"{split}_reference_normal_{score_name}"]
                for pairing in PAIRING_NAMES:
                    idx_1, idx_2 = _pair_indices(
                        pairing,
                        score_1,
                        score_2,
                        encoder_table=encoder_table,
                        split=split,
                        x_1=x_1,
                        x_2=x_2,
                    )
                    rows.append(
                        {
                            "trajectory_index": index,
                            "candidate_id": str(trajectory["candidate_id"]),
                            "reporting_seed": int(trajectory["reporting_seed"]),
                            "split": split,
                            "score": score_name,
                            "pairing": pairing,
                            "cap": _cap(score_1[idx_1], score_2[idx_2]),
                            "paired_score_spearman": float(
                                stats.spearmanr(score_1[idx_1], score_2[idx_2]).statistic
                            ),
                        }
                    )
    proxy_path = output_root / "proxy_metrics.csv"
    _write_csv(proxy_path, rows)
    freeze = {
        "schema_version": SCHEMA_VERSION,
        "design": str(design_path),
        "design_sha256": _sha256(design_path),
        "proxy_metrics": str(proxy_path),
        "proxy_metrics_sha256": _sha256(proxy_path),
        "n_rows": len(rows),
        "sources": sources,
        "intervention_outcomes_inspected_before_proxy_freeze": False,
    }
    freeze_path = output_root / "proxy_freeze.json"
    _atomic_json(freeze_path, freeze, create=True)
    return proxy_path, freeze_path


def evaluate_outcomes(audit_root: Path, output_root: Path, trajectory_index: int) -> Path:
    """Evaluate new score outcomes only after the proxy freeze gate."""
    _require_gpu()
    audit, trajectories, checkpoints = _audit_inputs(audit_root)
    output_root = output_root.expanduser().resolve()
    freeze_path = output_root / "proxy_freeze.json"
    freeze = _load_json(freeze_path)
    _load_design(Path(freeze["design"]))
    if freeze.get("intervention_outcomes_inspected_before_proxy_freeze") is not False:
        raise ValueError("Proxy freeze gate failed.")
    if freeze["proxy_metrics_sha256"] != _sha256(Path(freeze["proxy_metrics"])):
        raise ValueError("Frozen proxy table changed.")
    trajectory_by_index = {int(row["trajectory_index"]): row for row in trajectories}
    if int(trajectory_index) not in trajectory_by_index:
        raise IndexError("Trajectory is not in the fixed AE panel.")
    trajectory = trajectory_by_index[int(trajectory_index)]
    normal_path = output_root / "normal" / f"{int(trajectory_index):03d}.npz"
    normal = np.load(normal_path)
    state = {
        name: normal[f"state_{name}"]
        for name in (
            "delta",
            "residual_mean",
            "residual_variance",
            "residual_oas_location",
            "residual_oas_precision",
            "latent_oas_location",
            "latent_oas_precision",
        )
    }
    destination = output_root / "outcomes" / f"{int(trajectory_index):03d}.csv"
    marker_path = destination.with_suffix(".json")
    if destination.exists() and marker_path.exists():
        marker = _load_json(marker_path)
        if marker.get("artifact_sha256") == _sha256(destination):
            return destination
        raise ValueError("Existing outcome artifact failed its hash check.")

    datamodule, model = _compose(audit, trajectory, signals=audit["interventions"])
    checkpoint = checkpoints[int(trajectory_index)]
    rank_audit._load_checkpoint_state(model, Path(checkpoint["checkpoint"]))
    device = torch.device("cuda")
    test_loaders = datamodule.test_dataloader()
    _, normal_residual, normal_latent = _representations(model, test_loaders["normal"], device)
    normal_scores = score_arrays(normal_residual, normal_latent, state)
    thresholds = {
        name: float(np.quantile(normal[f"valid_normal_{name}"], 1.0 - TARGET_FPR))
        for name in SCORE_NAMES
    }
    rows = []
    for intervention in audit["interventions"]:
        _, signal_residual, signal_latent = _representations(
            model, test_loaders[intervention], device
        )
        signal_scores = score_arrays(signal_residual, signal_latent, state)
        for score_name in SCORE_NAMES:
            background = normal_scores[score_name]
            signal = signal_scores[score_name]
            target = np.concatenate(
                [np.zeros(len(background), dtype=int), np.ones(len(signal), dtype=int)]
            )
            prediction = np.concatenate([background, signal])
            rows.append(
                {
                    "trajectory_index": int(trajectory_index),
                    "candidate_id": str(trajectory["candidate_id"]),
                    "reporting_seed": int(trajectory["reporting_seed"]),
                    "score": score_name,
                    "intervention": intervention,
                    "auprc": float(average_precision_score(target, prediction)),
                    "efficiency": float(np.mean(signal >= thresholds[score_name])),
                    "validation_threshold": thresholds[score_name],
                }
            )
    datamodule.teardown(None)
    _write_csv(destination, rows)
    marker = {
        "schema_version": SCHEMA_VERSION,
        "proxy_freeze": str(freeze_path),
        "proxy_freeze_sha256": _sha256(freeze_path),
        "trajectory_index": int(trajectory_index),
        "artifact": str(destination),
        "artifact_sha256": _sha256(destination),
        "n_rows": len(rows),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "gpu": torch.cuda.get_device_name(0),
    }
    _atomic_json(marker_path, marker, create=True)
    return destination


def _holm(pvalues: Sequence[float]) -> list[float]:
    """Return Holm step-down adjusted p-values in original order."""
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def analyze(output_root: Path, *, n_permutations: int = 10_000) -> tuple[Path, Path]:
    """Analyze all frozen score-by-pairing rank associations."""
    output_root = output_root.expanduser().resolve()
    freeze = _load_json(output_root / "proxy_freeze.json")
    _load_design(Path(freeze["design"]))
    if freeze["proxy_metrics_sha256"] != _sha256(Path(freeze["proxy_metrics"])):
        raise ValueError("Frozen proxies changed before analysis.")
    proxy = pd.read_csv(freeze["proxy_metrics"], dtype={"candidate_id": str})
    outcome_frames = []
    for path in sorted((output_root / "outcomes").glob("*.csv")):
        marker = _load_json(path.with_suffix(".json"))
        if marker["artifact_sha256"] != _sha256(path):
            raise ValueError(f"Outcome artifact hash mismatch: {path}")
        outcome_frames.append(pd.read_csv(path, dtype={"candidate_id": str}))
    if len(outcome_frames) != 48:
        raise ValueError(f"Expected 48 outcome artifacts, found {len(outcome_frames)}.")
    outcomes = pd.concat(outcome_frames, ignore_index=True)
    outcome = (
        outcomes.groupby(["candidate_id", "reporting_seed", "score"])[["auprc", "efficiency"]]
        .mean()
        .reset_index()
    )
    validation = proxy[proxy["split"] == "valid"]
    test = proxy[proxy["split"] == "test"]
    rng = np.random.default_rng(904021)
    rows = []
    for score_name in SCORE_NAMES:
        score_outcome = outcome[outcome["score"] == score_name]
        for pairing in PAIRING_NAMES:
            valid = validation[
                (validation["score"] == score_name) & (validation["pairing"] == pairing)
            ]
            held_out = test[(test["score"] == score_name) & (test["pairing"] == pairing)]
            merged = valid.merge(
                score_outcome,
                on=["candidate_id", "reporting_seed"],
                validate="one_to_one",
            )
            candidate = merged.groupby("candidate_id")[["cap", "auprc", "efficiency"]].mean()
            held_candidate = (
                held_out.groupby("candidate_id")["cap"].mean().reindex(candidate.index)
            )
            for endpoint in ("auprc", "efficiency"):
                observed = float(stats.spearmanr(candidate["cap"], candidate[endpoint]).statistic)
                exceedances = 0
                values = candidate[endpoint].to_numpy()
                cap = candidate["cap"].to_numpy()
                for _ in range(n_permutations):
                    exceedances += int(
                        stats.spearmanr(cap, rng.permutation(values)).statistic >= observed
                    )
                rows.append(
                    {
                        "score": score_name,
                        "pairing": pairing,
                        "endpoint": endpoint,
                        "spearman_rho": observed,
                        "one_sided_permutation_p": (exceedances + 1) / (n_permutations + 1),
                        "validation_test_cap_rho": float(
                            stats.spearmanr(candidate["cap"], held_candidate).statistic
                        ),
                        "n_candidates": len(candidate),
                    }
                )
    non_control = [index for index, row in enumerate(rows) if row["pairing"] != "random"]
    for endpoint in ("auprc", "efficiency"):
        indices = [index for index in non_control if rows[index]["endpoint"] == endpoint]
        adjusted = _holm([rows[index]["one_sided_permutation_p"] for index in indices])
        for index, value in zip(indices, adjusted):
            rows[index]["holm_p"] = value
    for row in rows:
        row.setdefault("holm_p", math.nan)
    association_path = output_root / "analysis" / "score_cap_rank_associations.csv"
    _write_csv(association_path, rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "proxy_freeze_sha256": _sha256(output_root / "proxy_freeze.json"),
        "association_table": str(association_path),
        "association_table_sha256": _sha256(association_path),
        "n_permutations": n_permutations,
        "complete_variant_count": len(SCORE_NAMES) * len(PAIRING_NAMES),
        "primary_result": next(
            row
            for row in rows
            if row["score"] == "residual_oas"
            and row["pairing"] == "encoder"
            and row["endpoint"] == "auprc"
        ),
    }
    summary_path = output_root / "analysis" / "summary.json"
    _atomic_json(summary_path, summary, create=True)
    return association_path, summary_path


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("design", "extract-normal", "freeze-proxies", "evaluate-outcomes"):
        command = subparsers.add_parser(name)
        command.add_argument("--audit-root", type=Path, required=True)
        command.add_argument("--output-root", type=Path, required=True)
        if name in {"extract-normal", "evaluate-outcomes"}:
            command.add_argument("--trajectory-index", type=int, required=True)
    command = subparsers.add_parser("analyze")
    command.add_argument("--output-root", type=Path, required=True)
    command.add_argument("--n-permutations", type=int, default=10_000)
    return parser


def main() -> None:
    """Run one AE score-audit stage."""
    args = _parser().parse_args()
    if args.command == "design":
        print(create_design(args.audit_root, args.output_root))
    elif args.command == "extract-normal":
        print(extract_normal(args.audit_root, args.output_root, args.trajectory_index))
    elif args.command == "freeze-proxies":
        print(*freeze_proxies(args.audit_root, args.output_root), sep="\n")
    elif args.command == "evaluate-outcomes":
        print(evaluate_outcomes(args.audit_root, args.output_root, args.trajectory_index))
    elif args.command == "analyze":
        print(*analyze(args.output_root, n_permutations=args.n_permutations), sep="\n")


if __name__ == "__main__":
    main()
