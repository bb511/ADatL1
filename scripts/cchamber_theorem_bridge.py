#!/usr/bin/env python3
"""Empirical Causal Chamber illustration of marginal non-identifiability.

The experiment reuses the outcome-blind candidate-rank audit panel and its
sealed intervention evaluations.  For one fixed checkpoint branch, it extracts
each candidate detector's anomaly scores on independent validation and test
normal pairs.  Within every candidate and view, scores are monotonically
calibrated to the same empirical normal-quantile grid.  Consequently,
Wasserstein distance and threshold drift are identical for every candidate,
while CAP can still use the paired copula.

The calibration preserves anomaly-score ordering, so the previously sealed
intervention AUPRC values remain valid.  The analysis asks whether the
normal-only CAP ordering reproduces on held-out normal pairs and whether it
orders detection power for real process/actuator interventions and
measurement-chain interventions.
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
import matplotlib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts import cchamber_candidate_rank_audit as audit_tools  # noqa: E402
from src.analytical import cap_lift_for_score_matrix  # noqa: E402
from src.utils.pairing.io import compose_config  # noqa: E402

SCHEMA_VERSION = 1
SEED = 271828
BETAS = np.linspace(0.0, 8.0, 81)
FPR = 0.01
BRANCH = "cap_metadata_nearest"
PRESENTATION_MODELS = ("svdd", "vae", "realnvp")
MODEL_LABELS = {"ae": "AE", "vae": "VAE", "svdd": "SVDD", "realnvp": "RealNVP"}
MODEL_COLORS = {"svdd": "#D55E00", "vae": "#CC79A7", "realnvp": "#0072B2"}
GROUP_COLORS = {"process_or_actuator": "#0072B2", "measurement_chain": "#E69F00"}
CONFIRMATORY_STRATEGIES = (
    "cap_metadata_nearest",
    "cap_encoder_nearest",
    "drift",
    "wasserstein",
)
STRATEGY_LABELS = {
    "cap_metadata_nearest": "CAP\n(metadata)",
    "cap_encoder_nearest": "CAP\n(encoder)",
    "drift": "Drift",
    "wasserstein": "Wasserstein",
}
STRATEGY_COLORS = {
    "cap_metadata_nearest": "#0072B2",
    "cap_encoder_nearest": "#56B4E9",
    "drift": "#E69F00",
    "wasserstein": "#009E73",
}


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    """Serialize a JSON-compatible value deterministically."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write one strict JSON object atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _prepare_output(path: Path) -> Path:
    """Create an empty output directory without overwriting artifacts."""
    path = path.expanduser().resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_json(path: Path) -> Any:
    """Load a required JSON file."""
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _require_slurm_gpu() -> None:
    """Refuse candidate inference outside a Slurm GPU allocation."""
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Score extraction must run inside a Slurm allocation.")
    if not torch.cuda.is_available():
        raise RuntimeError("Score extraction requires a visible CUDA device.")


def _hydra_value(value: Any) -> str:
    """Render a JSON-compatible Hydra override."""
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "null"
    return json.dumps(value, separators=(",", ":"))


def _load_audit_inputs(
    audit_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, dict[str, Any]]]:
    """Authenticate the frozen manifests without repeatedly hashing 960 checkpoints."""
    audit_root = audit_root.expanduser().resolve()
    audit_path = audit_root / "audit.json"
    checkpoint_path = audit_root / "checkpoint_manifest.json"
    audit_sha = _sha256(audit_path)
    checkpoint_sha = _sha256(checkpoint_path)
    audit = _load_json(audit_path)
    if (
        audit.get("campaign_id") != "cchamber_real_20260725_63b941a"
        or audit.get("outcome_blind_at_design") is not True
        or int(audit.get("expected_trajectories", -1)) != audit_tools.EXPECTED_TRAJECTORIES
        or int(audit.get("expected_checkpoints", -1)) != audit_tools.EXPECTED_CHECKPOINTS
        or int(audit.get("expected_rows", -1)) != audit_tools.EXPECTED_ROWS
    ):
        raise ValueError("Frozen candidate-audit identity or coverage changed.")
    authenticated_inputs = (
        ("campaign_manifest", "campaign_manifest_sha256"),
        ("candidate_metrics", "candidate_metrics_sha256"),
        ("candidate_metrics_provenance", "candidate_metrics_provenance_sha256"),
        ("encoder_validation_pair_table", "encoder_validation_pair_table_sha256"),
        ("execution_contract", "execution_contract_sha256"),
        ("pairing_manifest", "pairing_manifest_sha256"),
        ("panel", "panel_sha256"),
        ("trajectory_manifest", "trajectory_manifest_sha256"),
    )
    for path_key, hash_key in authenticated_inputs:
        path = Path(audit[path_key])
        if _sha256(path) != audit[hash_key]:
            raise ValueError(f"Frozen audit input hash mismatch: {path}")
    trajectories = _load_json(Path(audit["trajectory_manifest"]))
    if (
        not isinstance(trajectories, list)
        or len(trajectories) != audit_tools.EXPECTED_TRAJECTORIES
        or [int(row["trajectory_index"]) for row in trajectories]
        != list(range(audit_tools.EXPECTED_TRAJECTORIES))
    ):
        raise ValueError("Frozen trajectory manifest coverage or ordering changed.")
    checkpoint_manifest = _load_json(checkpoint_path)
    if (
        checkpoint_manifest.get("audit_sha256") != audit_sha
        or checkpoint_manifest.get("outcomes_inspected_before_freeze") is not False
        or int(checkpoint_manifest.get("expected_checkpoints", -1))
        != audit_tools.EXPECTED_CHECKPOINTS
        or len(checkpoint_manifest.get("checkpoints", ())) != audit_tools.EXPECTED_CHECKPOINTS
    ):
        raise ValueError(f"Checkpoint freeze manifest failed validation ({checkpoint_sha}).")
    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in checkpoint_manifest["checkpoints"]:
        grouped.setdefault(int(record["trajectory_index"]), []).append(dict(record))
    if set(grouped) != set(range(len(trajectories))):
        raise ValueError("Checkpoint freeze trajectory coverage is incomplete.")
    branches: dict[int, dict[str, Any]] = {}
    for trajectory_index, records in grouped.items():
        matches = [record for record in records if record["strategy"] == BRANCH]
        if len(matches) != 1:
            raise ValueError(f"Trajectory {trajectory_index} lacks exactly one {BRANCH} branch.")
        branches[int(trajectory_index)] = dict(matches[0])
    return audit, trajectories, branches


class _NormalScoreCollector(pl.Callback):
    """Collect finite anomaly scores from exactly two named normal streams."""

    def on_test_epoch_start(self, trainer, pl_module) -> None:
        """Reset buffers for one validation/test split."""
        del pl_module
        names = tuple(trainer.test_dataloaders)
        if names != ("normal", "reference_normal"):
            raise ValueError(f"Unexpected normal stream order: {names}")
        self.names = names
        self.scores = {name: [] for name in names}

    def on_test_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ) -> None:
        """Append one finite score batch."""
        del pl_module, batch, batch_idx
        name = self.names[dataloader_idx]
        score = outputs["ascore/full"].detach().view(-1).float().cpu()
        if not torch.isfinite(score).all():
            raise ValueError(f"Non-finite scores in {name}.")
        self.scores[name].append(score)

    def arrays(self) -> dict[str, np.ndarray]:
        """Return copied score arrays after one completed test call."""
        return {
            name: torch.cat(values).numpy().astype(np.float64, copy=True)
            for name, values in self.scores.items()
        }


def _compose_normal_datamodule(audit: Mapping[str, Any], trajectory: Mapping[str, Any]):
    """Compose the frozen model config without loading intervention tables."""
    os.environ["CCHAMBER_VALID_PAIR_TABLE"] = str(audit["encoder_validation_pair_table"])
    overrides = [
        f"experiment=cchamber/{trajectory['model']}_candidate_rank_audit",
        f"seed={int(trajectory['reporting_seed'])}",
        "data.seed=314159",
        f"paths.base_data_dir={audit['data_dir']}",
        "data.signal_experiments=[]",
        "logger=none",
        *[f"{name}={_hydra_value(value)}" for name, value in trajectory["params"].items()],
    ]
    cfg = compose_config(overrides=overrides)
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup(None)
    return cfg, datamodule


def extract_scores(
    audit_root: Path,
    output_root: Path,
    trajectory_index: int,
) -> tuple[Path, Path]:
    """Extract paired validation/test normal scores for one frozen detector."""
    _require_slurm_gpu()
    audit, trajectories, branches = _load_audit_inputs(audit_root)
    if not 0 <= int(trajectory_index) < len(trajectories):
        raise IndexError(f"trajectory-index must be in [0, {len(trajectories) - 1}].")
    trajectory = trajectories[int(trajectory_index)]
    branch = branches[int(trajectory_index)]
    if _sha256(Path(branch["checkpoint"])) != branch["checkpoint_sha256"]:
        raise ValueError(f"Frozen checkpoint hash mismatch: {branch['checkpoint']}")
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    npz_path = output_root / f"{int(trajectory_index):03d}.npz"
    marker_path = output_root / f"{int(trajectory_index):03d}.json"
    if npz_path.exists() or marker_path.exists():
        if not (npz_path.is_file() and marker_path.is_file()):
            raise FileExistsError(f"Partial score artifact for trajectory {trajectory_index}.")
        marker = _load_json(marker_path)
        if (
            marker.get("score_sha256") != _sha256(npz_path)
            or int(marker.get("trajectory_index", -1)) != int(trajectory_index)
            or marker.get("checkpoint_sha256") != branch["checkpoint_sha256"]
        ):
            raise ValueError(f"Existing score artifact failed validation: {marker_path}")
        return npz_path, marker_path

    cfg, datamodule = _compose_normal_datamodule(audit, trajectory)
    pl.seed_everything(int(trajectory["reporting_seed"]), workers=True)
    model = hydra.utils.instantiate(cfg.algorithm)
    checkpoint = Path(branch["checkpoint"])
    audit_tools._load_checkpoint_state(model, checkpoint)
    collector = _NormalScoreCollector()
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        logger=False,
        callbacks=[collector],
        enable_checkpointing=False,
        enable_progress_bar=False,
        deterministic=True,
        inference_mode=True,
    )
    trainer.split = "test"

    valid_loaders = datamodule.val_dataloader()
    valid = {name: valid_loaders[name] for name in ("normal", "reference_normal")}
    trainer.test(model=model, dataloaders=valid, verbose=False)
    valid_scores = collector.arrays()

    test_loaders = datamodule.test_dataloader()
    test = {name: test_loaders[name] for name in ("normal", "reference_normal")}
    trainer.test(model=model, dataloaders=test, verbose=False)
    test_scores = collector.arrays()
    datamodule.teardown(None)

    arrays = {
        "valid_normal": valid_scores["normal"],
        "valid_reference_normal": valid_scores["reference_normal"],
        "test_normal": test_scores["normal"],
        "test_reference_normal": test_scores["reference_normal"],
    }
    if {len(values) for values in arrays.values()} != {1000}:
        raise ValueError(
            f"Expected four 1,000-score arrays, got {list(map(len, arrays.values()))}."
        )
    temporary = npz_path.with_name(f".{npz_path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, npz_path)
    marker = {
        "schema_version": SCHEMA_VERSION,
        "classification": "post_confirmatory_theorem_bridge_score_extraction",
        "trajectory_index": int(trajectory_index),
        "model": trajectory["model"],
        "candidate_id": str(trajectory["candidate_id"]),
        "reporting_seed": int(trajectory["reporting_seed"]),
        "branch": BRANCH,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": branch["checkpoint_sha256"],
        "selected_epoch": int(branch["selected_epoch"]),
        "score_path": str(npz_path),
        "score_sha256": _sha256(npz_path),
        "array_lengths": {name: int(len(values)) for name, values in arrays.items()},
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "gpu": torch.cuda.get_device_name(0),
    }
    _atomic_json(marker_path, marker)
    return npz_path, marker_path


def common_quantile_calibration(
    scores: np.ndarray,
    *,
    tie_seed: int = SEED,
) -> np.ndarray:
    """Map scores to one grid, using a deterministic distributional transform for ties."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not np.isfinite(scores).all() or len(scores) < 100:
        raise ValueError("Score calibration requires a finite nontrivial vector.")
    tie_breaker = np.random.default_rng(int(tie_seed)).random(len(scores))
    order = np.lexsort((tie_breaker, scores))
    quantiles = stats.norm.ppf((np.arange(len(scores), dtype=float) + 0.5) / len(scores))
    calibrated = np.empty_like(scores)
    calibrated[order] = quantiles
    return calibrated


def matched_marginal_metrics(
    scores_1: np.ndarray,
    scores_2: np.ndarray,
    *,
    betas: np.ndarray = BETAS,
    fpr: float = FPR,
    permutation_seed: int | None = None,
) -> dict[str, float]:
    """Return CAP and the two marginal criteria after exact calibration."""
    seed = SEED if permutation_seed is None else int(permutation_seed)
    first = common_quantile_calibration(scores_1, tie_seed=seed + 1)
    second = common_quantile_calibration(scores_2, tie_seed=seed + 2)
    cap, _, beta = cap_lift_for_score_matrix(first[:, None], second[:, None], betas)
    wasserstein = float(np.mean(np.abs(np.sort(first) - np.sort(second))))
    threshold = float(np.quantile(first, 1.0 - fpr, method="higher"))
    exceedance = float(np.mean(second >= threshold))
    drift = abs(exceedance - float(fpr))
    paired_spearman = float(stats.spearmanr(first, second).statistic)
    result = {
        "cap": float(cap[0]),
        "cap_beta": float(beta[0]),
        "wasserstein": wasserstein,
        "threshold_drift": drift,
        "tail_exceedance": exceedance,
        "paired_spearman": paired_spearman,
        "n_ties_view_1": int(len(scores_1) - np.unique(scores_1).size),
        "n_ties_view_2": int(len(scores_2) - np.unique(scores_2).size),
    }
    if permutation_seed is not None:
        rng = np.random.default_rng(int(permutation_seed))
        permuted = second[rng.permutation(len(second))]
        random_cap, _, random_beta = cap_lift_for_score_matrix(
            first[:, None],
            permuted[:, None],
            betas,
        )
        result["random_pair_cap"] = float(random_cap[0])
        result["random_pair_cap_beta"] = float(random_beta[0])
        result["random_pair_spearman"] = float(stats.spearmanr(first, permuted).statistic)
    return result


def _physical_catalog(path: Path) -> tuple[dict[str, dict[str, Any]], str]:
    """Load the frozen intervention target catalog."""
    value = _load_json(path)
    records = value.get("targets")
    if not isinstance(records, list) or len(records) != 29:
        raise ValueError("Physical catalog must contain 29 target records.")
    by_target = {str(record["target"]): dict(record) for record in records}
    if len(by_target) != len(records):
        raise ValueError("Physical catalog target names are not unique.")
    return by_target, _sha256(path)


def _intervention_target(name: str) -> str:
    """Strip the fixed prefix and strength suffix from an intervention name."""
    stem = str(name).removeprefix("uniform_")
    parts = stem.split("_")
    if parts[-1] in {"weak", "mid", "strong"}:
        parts = parts[:-1]
    return "_".join(parts)


def _load_outcomes(
    audit_root: Path,
    catalog: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """Load and authenticate the already sealed candidate intervention outcomes."""
    rows = []
    evaluation_root = audit_root / "evaluation"
    for marker_path in sorted(evaluation_root.glob("*.json")):
        marker = _load_json(marker_path)
        path = Path(marker["evaluation_rows"])
        if _sha256(path) != marker["evaluation_rows_sha256"]:
            raise ValueError(f"Evaluation hash mismatch: {path}")
        frame = pd.read_csv(path)
        rows.append(frame)
    outcomes = pd.concat(rows, ignore_index=True)
    expected = audit_tools.EXPECTED_ROWS
    if len(outcomes) != expected:
        raise ValueError(f"Expected {expected} sealed rows, found {len(outcomes)}.")
    outcomes = outcomes[(outcomes["strategy"] == BRANCH) & (outcomes["metric"] == "auprc")].copy()
    outcomes["candidate_id"] = outcomes["candidate_id"].map(lambda value: f"{int(value):03d}")
    outcomes["target"] = outcomes["intervention"].map(_intervention_target)
    outcomes["physical_class"] = outcomes["target"].map(
        {target: record["physical_class"] for target, record in catalog.items()}
    )
    outcomes["semantic_family"] = outcomes["target"].map(
        {target: record["semantic_family"] for target, record in catalog.items()}
    )
    if outcomes[["physical_class", "semantic_family"]].isna().any().any():
        raise ValueError("Sealed outcomes contain interventions absent from the physical catalog.")
    identity = ["trajectory_index", "intervention"]
    if outcomes.duplicated(identity).any() or len(outcomes) != 192 * 58:
        raise ValueError("Sealed branch outcomes lack exact 192x58 coverage.")
    return outcomes


def _score_metric_table(
    audit_root: Path,
    scores_root: Path,
    trajectories: Sequence[Mapping[str, Any]],
    branches: Mapping[int, Mapping[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Load every score artifact and compute matched-marginal criteria."""
    rows = []
    markers = []
    for index, trajectory in enumerate(trajectories):
        marker_path = scores_root / f"{index:03d}.json"
        npz_path = scores_root / f"{index:03d}.npz"
        marker = _load_json(marker_path)
        if (
            int(marker.get("trajectory_index", -1)) != index
            or marker.get("checkpoint_sha256") != branches[index]["checkpoint_sha256"]
            or marker.get("score_sha256") != _sha256(npz_path)
        ):
            raise ValueError(f"Score marker identity mismatch: {marker_path}")
        markers.append(marker)
        with np.load(npz_path) as values:
            validation = matched_marginal_metrics(
                values["valid_normal"],
                values["valid_reference_normal"],
                permutation_seed=SEED + index,
            )
            test = matched_marginal_metrics(
                values["test_normal"],
                values["test_reference_normal"],
                permutation_seed=SEED + 10_000 + index,
            )
        rows.append(
            {
                "trajectory_index": index,
                "model": trajectory["model"],
                "candidate_id": str(trajectory["candidate_id"]),
                "reporting_seed": int(trajectory["reporting_seed"]),
                **{f"validation_{name}": value for name, value in validation.items()},
                **{f"test_{name}": value for name, value in test.items()},
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 192 or frame.duplicated(["model", "candidate_id", "reporting_seed"]).any():
        raise ValueError("Score metric table lacks exact 192-trajectory coverage.")
    for column in ("validation_wasserstein", "test_wasserstein"):
        if not np.allclose(frame[column], 0.0, atol=1.0e-15, rtol=0.0):
            raise ValueError(f"Exact marginal matching failed for {column}.")
    for column in ("validation_threshold_drift", "test_threshold_drift"):
        if frame[column].nunique() != 1:
            raise ValueError(f"Marginal drift is not tied across candidates: {column}.")
    return frame, markers


def _mean_rank(values: pd.Series) -> pd.Series:
    """Return higher-is-better ranks normalized to [0, 1]."""
    if len(values) < 2:
        raise ValueError("At least two candidates are needed for ranking.")
    return values.rank(method="average", ascending=True).sub(1.0).div(len(values) - 1.0)


def candidate_summary(
    score_metrics: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate normal-only ranks and physical outcomes by candidate."""
    ranked = score_metrics.copy()
    ranked["validation_cap_rank"] = ranked.groupby(["model", "reporting_seed"], sort=False)[
        "validation_cap"
    ].transform(_mean_rank)
    ranked["test_cap_rank"] = ranked.groupby(["model", "reporting_seed"], sort=False)[
        "test_cap"
    ].transform(_mean_rank)
    candidate = (
        ranked.groupby(["model", "candidate_id"], sort=True)
        .agg(
            validation_cap=("validation_cap", "mean"),
            validation_cap_mean_rank=("validation_cap_rank", "mean"),
            test_cap=("test_cap", "mean"),
            test_cap_mean_rank=("test_cap_rank", "mean"),
            validation_random_pair_cap=("validation_random_pair_cap", "mean"),
            test_random_pair_cap=("test_random_pair_cap", "mean"),
            n_reporting_seeds=("reporting_seed", "nunique"),
        )
        .reset_index()
    )
    trajectory_outcome = (
        outcomes.groupby(
            ["trajectory_index", "model", "candidate_id", "reporting_seed", "physical_class"],
            sort=True,
        )["value"]
        .mean()
        .rename("mean_auprc")
        .reset_index()
    )
    outcome_by_candidate = (
        trajectory_outcome.groupby(["model", "candidate_id", "physical_class"], sort=True)[
            "mean_auprc"
        ]
        .mean()
        .unstack("physical_class")
        .reset_index()
        .rename(
            columns={
                "process_or_actuator": "process_auprc",
                "measurement_chain": "measurement_auprc",
            }
        )
    )
    all_outcome = (
        outcomes.groupby(["model", "candidate_id"], sort=True)["value"]
        .mean()
        .rename("all_auprc")
        .reset_index()
    )
    candidate = candidate.merge(
        outcome_by_candidate,
        on=["model", "candidate_id"],
        validate="one_to_one",
    ).merge(all_outcome, on=["model", "candidate_id"], validate="one_to_one")
    return ranked, candidate


def _spearman_permutation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    repetitions: int = 50_000,
) -> tuple[float, float]:
    """Return a directional Spearman coefficient and Monte Carlo p-value."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_rank = stats.rankdata(x).astype(float)
    y_rank = stats.rankdata(y).astype(float)
    x_rank -= x_rank.mean()
    y_rank -= y_rank.mean()
    denominator = float(np.linalg.norm(x_rank) * np.linalg.norm(y_rank))
    if denominator <= 0:
        raise ValueError("Spearman permutation requires nonconstant inputs.")
    observed = float(np.dot(x_rank, y_rank) / denominator)
    rng = np.random.default_rng(seed)
    exceed = 0
    remaining = int(repetitions)
    while remaining:
        batch_size = min(10_000, remaining)
        permutations = np.stack([rng.permutation(len(y_rank)) for _ in range(batch_size)])
        permuted = np.sum(x_rank[None, :] * y_rank[permutations], axis=1) / denominator
        exceed += int(np.sum(permuted >= observed))
        remaining -= batch_size
    return observed, (exceed + 1.0) / (repetitions + 1.0)


def association_table(candidate: pd.DataFrame) -> pd.DataFrame:
    """Summarize CAP ordering versus physical intervention power."""
    rows = []
    outcomes = (
        ("all", "all_auprc"),
        ("process_or_actuator", "process_auprc"),
        ("measurement_chain", "measurement_auprc"),
    )
    for model_index, (model, group) in enumerate(candidate.groupby("model", sort=True)):
        for outcome_index, (physical_class, column) in enumerate(outcomes):
            rho, pvalue = _spearman_permutation(
                group["validation_cap_mean_rank"].to_numpy(),
                group[column].to_numpy(),
                seed=SEED + 100 * model_index + outcome_index,
            )
            winner = group.loc[group["validation_cap_mean_rank"].idxmax()]
            oracle = group.loc[group[column].idxmax()]
            rows.append(
                {
                    "model": model,
                    "physical_class": physical_class,
                    "n_candidates": int(len(group)),
                    "spearman_rho": rho,
                    "one_sided_permutation_p": pvalue,
                    "cap_selected_candidate": str(winner["candidate_id"]),
                    "cap_selected_auprc": float(winner[column]),
                    "marginal_tie_mean_auprc": float(group[column].mean()),
                    "marginal_tie_min_auprc": float(group[column].min()),
                    "marginal_tie_max_auprc": float(group[column].max()),
                    "panel_oracle_candidate": str(oracle["candidate_id"]),
                    "panel_oracle_auprc": float(oracle[column]),
                    "cap_selected_regret": float(oracle[column] - winner[column]),
                }
            )
    return pd.DataFrame(rows)


def intervention_contrasts(
    score_metrics: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    """Compare top and bottom CAP quartiles for every physical intervention."""
    ranked = score_metrics.copy()
    ranked["seed_cap_rank"] = ranked.groupby(["model", "reporting_seed"], sort=False)[
        "validation_cap"
    ].transform(_mean_rank)
    joined = outcomes.merge(
        ranked[["trajectory_index", "seed_cap_rank"]],
        on="trajectory_index",
        validate="many_to_one",
    )
    joined["cap_quartile"] = np.where(
        joined["seed_cap_rank"] >= 0.75,
        "top",
        np.where(joined["seed_cap_rank"] <= 0.25, "bottom", "middle"),
    )
    selected = joined[joined["cap_quartile"].isin(["top", "bottom"])]
    summary = (
        selected.groupby(
            [
                "model",
                "intervention",
                "target",
                "physical_class",
                "semantic_family",
                "cap_quartile",
            ],
            sort=True,
        )["value"]
        .mean()
        .unstack("cap_quartile")
        .reset_index()
    )
    summary["top_minus_bottom_auprc"] = summary["top"] - summary["bottom"]
    return summary


def confirmatory_system_group_performance(
    path: Path,
    catalog: Mapping[str, Mapping[str, Any]],
) -> tuple[pd.DataFrame, str]:
    """Summarize the independent ten-seed RealNVP campaign by physical class."""
    path = path.expanduser().resolve()
    frame = pd.read_csv(path)
    identity = ["model", "strategy", "seed", "intervention", "metric"]
    if len(frame) != 23_200 or frame.duplicated(identity).any():
        raise ValueError("Confirmatory result table lacks exact 23,200-row coverage.")
    selected = frame[
        (frame["model"] == "realnvp")
        & (frame["metric"] == "auprc")
        & (frame["strategy"].isin(CONFIRMATORY_STRATEGIES))
    ].copy()
    selected["target"] = selected["intervention"].map(_intervention_target)
    selected["physical_class"] = selected["target"].map(
        {target: record["physical_class"] for target, record in catalog.items()}
    )
    if selected["physical_class"].isna().any():
        raise ValueError("Confirmatory interventions are absent from the physical catalog.")
    seed_first = (
        selected.groupby(["strategy", "physical_class", "seed"], sort=True)["value"]
        .mean()
        .rename("seed_mean_auprc")
        .reset_index()
    )
    summary = (
        seed_first.groupby(["strategy", "physical_class"], sort=True)["seed_mean_auprc"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_auprc",
                "std": "seed_sd",
                "count": "n_reporting_seeds",
            }
        )
    )
    if len(summary) != 8 or set(summary["n_reporting_seeds"]) != {10}:
        raise ValueError("Confirmatory physical-class summary lacks exact 4x2x10 coverage.")
    return summary, _sha256(path)


def _plot(
    score_metrics: pd.DataFrame,
    candidate: pd.DataFrame,
    associations: pd.DataFrame,
    contrasts: pd.DataFrame,
    confirmatory: pd.DataFrame,
    output: Path,
) -> None:
    """Create the information-dense theorem-bridge figure."""
    del contrasts
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 9.2), constrained_layout=True)
    axes = axes.ravel()

    ax = axes[0]
    realnvp = score_metrics[score_metrics["model"] == "realnvp"].copy()
    ordered = realnvp.groupby("candidate_id")["validation_cap"].mean().sort_values().index.tolist()
    means = realnvp.groupby("candidate_id").mean(numeric_only=True).loc[ordered]
    x = np.arange(len(ordered))
    cap_scaled = (means["validation_cap"] - means["validation_cap"].min()) / max(
        means["validation_cap"].max() - means["validation_cap"].min(),
        1.0e-15,
    )
    ax.plot(x, cap_scaled, "o-", color="#0072B2", label="CAP (paired)")
    ax.plot(x, np.zeros_like(x), "s-", color="#E69F00", label="Threshold drift")
    ax.plot(x, np.zeros_like(x), "^-", color="#009E73", label="Wasserstein")
    ax.set_xticks(x[::3], [ordered[index] for index in x[::3]], rotation=45)
    ax.set_xlabel("RealNVP candidate ID (ordered by CAP)")
    ax.set_ylabel("Within-criterion range (0–1)")
    ax.set_title("A. Equal marginals: only CAP identifies")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.2)

    ax = axes[1]
    for model in PRESENTATION_MODELS:
        subset = score_metrics[score_metrics["model"] == model]
        ax.scatter(
            subset["validation_cap"],
            subset["test_cap"],
            s=25,
            alpha=0.70,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
    low = min(score_metrics["validation_cap"].min(), score_metrics["test_cap"].min())
    high = max(score_metrics["validation_cap"].max(), score_metrics["test_cap"].max())
    ax.plot([low, high], [low, high], linestyle="--", color="#777777", linewidth=1)
    ax.set_xlabel("Validation CAP after marginal matching")
    ax.set_ylabel("Held-out normal-pair CAP")
    ax.set_title("B. Paired dependence reproduces")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.2)

    ax = axes[2]
    real_candidates = candidate[candidate["model"] == "realnvp"].sort_values(
        "validation_cap_mean_rank"
    )
    for physical_class, column, marker in (
        ("process_or_actuator", "process_auprc", "o"),
        ("measurement_chain", "measurement_auprc", "s"),
    ):
        row = associations[
            (associations["model"] == "realnvp")
            & (associations["physical_class"] == physical_class)
        ].iloc[0]
        label = (
            "Process shifts" if physical_class == "process_or_actuator" else "Measurement shifts"
        ) + f" ($\\rho$={row['spearman_rho']:.2f})"
        ax.scatter(
            real_candidates["validation_cap_mean_rank"],
            real_candidates[column],
            color=GROUP_COLORS[physical_class],
            marker=marker,
            s=50,
            alpha=0.85,
            label=label,
        )
        fit = stats.theilslopes(
            real_candidates[column],
            real_candidates["validation_cap_mean_rank"],
        )
        grid = np.linspace(0.0, 1.0, 100)
        ax.plot(
            grid,
            fit.intercept + fit.slope * grid,
            color=GROUP_COLORS[physical_class],
            linewidth=1.5,
        )
    ax.set_xlabel("Normal-only CAP rank (mean across seeds)")
    ax.set_ylabel("Mean sealed test AUPRC")
    ax.set_title("C. Real physical interventions")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.2)

    ax = axes[3]
    x = np.arange(len(CONFIRMATORY_STRATEGIES))
    width = 0.36
    for offset, physical_class in ((-0.5, "process_or_actuator"), (0.5, "measurement_chain")):
        subset = confirmatory.set_index(["strategy", "physical_class"])
        means = [
            float(subset.loc[(strategy, physical_class), "mean_auprc"])
            for strategy in CONFIRMATORY_STRATEGIES
        ]
        errors = [
            float(subset.loc[(strategy, physical_class), "seed_sd"]) / math.sqrt(10.0)
            for strategy in CONFIRMATORY_STRATEGIES
        ]
        ax.bar(
            x + offset * width,
            means,
            width,
            yerr=errors,
            capsize=3,
            color=[STRATEGY_COLORS[strategy] for strategy in CONFIRMATORY_STRATEGIES],
            alpha=0.70 if physical_class == "process_or_actuator" else 1.0,
            hatch="//" if physical_class == "process_or_actuator" else None,
            edgecolor="#333333",
            linewidth=0.5,
            label=(
                "Process/actuator shifts"
                if physical_class == "process_or_actuator"
                else "Measurement-chain shifts"
            ),
        )
    ax.set_xticks(x, [STRATEGY_LABELS[strategy] for strategy in CONFIRMATORY_STRATEGIES])
    ax.set_ylabel("Mean test AUPRC")
    ax.set_title("D. Independent full RealNVP campaign")
    ax.set_ylim(0.35, 0.85)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.2)

    figure.suptitle(
        "Causal Chamber theorem bridge: matched marginals, paired reliability, physical power",
        fontsize=14,
    )
    figure.savefig(output, dpi=220)
    plt.close(figure)


def analyze(
    audit_root: Path,
    scores_root: Path,
    target_catalog: Path,
    confirmatory_results: Path,
    output_dir: Path,
) -> list[Path]:
    """Analyze all score artifacts and create the theorem bridge."""
    output_dir = _prepare_output(output_dir)
    audit_root = audit_root.expanduser().resolve()
    scores_root = scores_root.expanduser().resolve()
    audit, trajectories, branches = _load_audit_inputs(audit_root)
    catalog, catalog_sha = _physical_catalog(target_catalog)
    score_metrics, score_markers = _score_metric_table(
        audit_root,
        scores_root,
        trajectories,
        branches,
    )
    outcomes = _load_outcomes(audit_root, catalog)
    ranked, candidate = candidate_summary(score_metrics, outcomes)
    associations = association_table(candidate)
    contrasts = intervention_contrasts(score_metrics, outcomes)
    confirmatory, confirmatory_sha = confirmatory_system_group_performance(
        confirmatory_results,
        catalog,
    )

    outputs = []
    tables = (
        ("matched_marginal_trajectory_metrics.csv", score_metrics),
        ("matched_marginal_candidate_summary.csv", candidate),
        ("matched_marginal_associations.csv", associations),
        ("physical_intervention_cap_quartile_contrasts.csv", contrasts),
        ("confirmatory_realnvp_system_group_performance.csv", confirmatory),
    )
    for name, frame in tables:
        path = output_dir / name
        frame.to_csv(path, index=False)
        outputs.append(path)

    figure = output_dir / "cchamber_theorem_bridge.png"
    _plot(score_metrics, candidate, associations, contrasts, confirmatory, figure)
    outputs.append(figure)

    validation_test = {}
    pairing_control = {}
    for model in audit_tools.MODELS:
        subset = score_metrics[score_metrics["model"] == model]
        validation_test[model] = float(
            stats.spearmanr(subset["validation_cap"], subset["test_cap"]).statistic
        )
        pairing_control[model] = {
            "mean_paired_validation_cap": float(subset["validation_cap"].mean()),
            "mean_random_pair_validation_cap": float(subset["validation_random_pair_cap"].mean()),
            "paired_minus_random_mean": float(
                (subset["validation_cap"] - subset["validation_random_pair_cap"]).mean()
            ),
        }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "classification": "post_confirmatory_empirical_theorem_bridge",
        "scientific_question": (
            "When candidate normal-score marginals are made identical, can paired CAP still "
            "identify reproducible detectors and order their power on controlled physical shifts?"
        ),
        "theorem_instantiation": {
            "calibration": (
                "candidate- and view-specific empirical normal-quantile calibration; exact "
                "floating-point ties use an independent deterministic randomized "
                "probability-integral transform"
            ),
            "ordering_preserved": True,
            "maximum_ties_in_one_1000_score_view": int(
                score_metrics[
                    [
                        "validation_n_ties_view_1",
                        "validation_n_ties_view_2",
                        "test_n_ties_view_1",
                        "test_n_ties_view_2",
                    ]
                ]
                .to_numpy()
                .max()
            ),
            "validation_wasserstein_unique_values": sorted(
                map(float, score_metrics["validation_wasserstein"].unique())
            ),
            "validation_threshold_drift_unique_values": sorted(
                map(float, score_metrics["validation_threshold_drift"].unique())
            ),
            "marginal_selector_conclusion": (
                "Wasserstein and threshold drift tie all candidates and cannot identify a winner."
            ),
        },
        "candidate_panel": {
            "n_trajectories": int(len(score_metrics)),
            "n_candidates_per_model": int(candidate.groupby("model").size().min()),
            "n_reporting_seeds": int(score_metrics["reporting_seed"].nunique()),
            "checkpoint_branch": BRANCH,
            "interventions_sealed_in_original_audit": True,
        },
        "held_out_normal_pair_validation": {
            "validation_test_cap_spearman": validation_test,
            "random_pair_negative_control": pairing_control,
        },
        "physical_intervention_associations": associations.to_dict(orient="records"),
        "independent_confirmatory_realnvp": {
            "classification": (
                "prespecified ten-seed campaign; physical-class split is descriptive"
            ),
            "system_group_performance": confirmatory.to_dict(orient="records"),
        },
        "claim_boundary": (
            "Marginal matching demonstrates non-identifiability, not universal CAP superiority. "
            "Physical power additionally requires the theorem's alignment condition; architectures "
            "that violate it are retained as empirical boundary cases."
        ),
    }
    summary_path = output_dir / "theorem_bridge_summary.json"
    _atomic_json(summary_path, summary)
    outputs.append(summary_path)

    provenance = {
        "schema_version": SCHEMA_VERSION,
        "repository_commit": audit_tools._git("rev-parse", "HEAD"),
        "repository_branch": audit_tools._git("branch", "--show-current"),
        "script": str(Path(__file__).resolve()),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "audit_root": str(audit_root),
        "audit_manifest_sha256": _sha256(audit_root / "audit.json"),
        "checkpoint_manifest_sha256": _sha256(audit_root / "checkpoint_manifest.json"),
        "physical_catalog": str(target_catalog.expanduser().resolve()),
        "physical_catalog_sha256": catalog_sha,
        "confirmatory_results": str(confirmatory_results.expanduser().resolve()),
        "confirmatory_results_sha256": confirmatory_sha,
        "score_artifact_manifest_sha256": hashlib.sha256(
            _canonical_json(
                [
                    {
                        "trajectory_index": marker["trajectory_index"],
                        "score_sha256": marker["score_sha256"],
                        "checkpoint_sha256": marker["checkpoint_sha256"],
                    }
                    for marker in score_markers
                ]
            ).encode("utf-8")
        ).hexdigest(),
        "inputs": {
            "sealed_outcome_rows": int(len(outcomes)),
            "normal_score_trajectories": int(len(score_metrics)),
        },
        "outputs": {path.name: {"path": str(path), "sha256": _sha256(path)} for path in outputs},
    }
    provenance_path = output_dir / "theorem_bridge_provenance.json"
    _atomic_json(provenance_path, provenance)
    outputs.append(provenance_path)
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract-scores")
    extract.add_argument("--audit-root", type=Path, required=True)
    extract.add_argument("--output-root", type=Path, required=True)
    extract.add_argument("--trajectory-index", type=int, required=True)

    analysis = subparsers.add_parser("analyze")
    analysis.add_argument("--audit-root", type=Path, required=True)
    analysis.add_argument("--scores-root", type=Path, required=True)
    analysis.add_argument("--target-catalog", type=Path, required=True)
    analysis.add_argument("--confirmatory-results", type=Path, required=True)
    analysis.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run score extraction or final analysis."""
    args = parse_args(argv)
    if args.command == "extract-scores":
        for path in extract_scores(args.audit_root, args.output_root, args.trajectory_index):
            print(path)
    elif args.command == "analyze":
        for path in analyze(
            args.audit_root,
            args.scores_root,
            args.target_catalog,
            args.confirmatory_results,
            args.output_dir,
        ):
            print(path)
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
