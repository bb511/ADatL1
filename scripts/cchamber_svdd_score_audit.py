"""Outcome-gated Causal Chamber audit of Deep-SVDD score geometry.

The study fixes one checkpoint per trajectory, extracts every normal-only CAP
proxy, freezes those proxies, and only then computes intervention outcomes. It
reports both ordinary CAP maximization and the architecture-motivated inverse
direction: Deep SVDD is trained to erase normal variation, so lower normal-view
capacity may be the appropriate label-free objective.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import cchamber_ae_score_audit as ae_audit
import cchamber_candidate_rank_audit as rank
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from scipy import stats
from sklearn.covariance import OAS
from torchmetrics.classification import BinaryAveragePrecision

from src.utils.pairing.io import compose_config

SCORES = (
    "radial_d2",
    "log_radial",
    "center_oas",
    "latent_oas",
    "max_axis",
    "cosine_center",
)
PAIRINGS = ("metadata", "encoder", "cdf", "random")
DIRECTIONS = ("maximize", "minimize")
FIXED_BRANCH = "cap_encoder_nearest"
N_TRAJECTORIES = 48
SCRIPT_PATH = Path(__file__).resolve()


def _load(path: Path) -> Any:
    """Load one JSON artifact."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically create one JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _inputs(audit_root: Path):
    """Return the source audit, SVDD trajectories, and fixed checkpoints."""
    audit = _load(audit_root / "audit.json")
    trajectories = [
        dict(row) for row in _load(Path(audit["trajectory_manifest"])) if row["model"] == "svdd"
    ]
    checkpoints = [
        dict(row)
        for row in _load(audit_root / "checkpoint_manifest.json")["checkpoints"]
        if row["model"] == "svdd" and row["strategy"] == FIXED_BRANCH
    ]
    if len(trajectories) != N_TRAJECTORIES or len(checkpoints) != N_TRAJECTORIES:
        raise ValueError("SVDD trajectory/checkpoint coverage is not exactly 48.")
    checkpoints = {int(row["trajectory_index"]): row for row in checkpoints}
    for index, trajectory in enumerate(trajectories):
        trajectory["source_trajectory_index"] = int(trajectory["trajectory_index"])
        trajectory["trajectory_index"] = index
    return audit, trajectories, checkpoints


def _validated_design(output_root: Path) -> dict[str, Any]:
    """Load the design and require the exact frozen audit implementation."""
    value = _load(output_root / "design.json")
    if value["audit_script_sha256"] != ae_audit._sha256(SCRIPT_PATH):
        raise RuntimeError("SVDD score-audit code changed after design freeze.")
    return value


def design(audit_root: Path, output_root: Path) -> Path:
    """Freeze score families and hypotheses before new outcomes are computed."""
    audit, trajectories, checkpoints = _inputs(audit_root)
    output = output_root / "design.json"
    value = {
        "schema_version": 1,
        "classification": "post_confirmatory_svdd_score_diagnostic",
        "audit_script": str(SCRIPT_PATH),
        "audit_script_sha256": ae_audit._sha256(SCRIPT_PATH),
        "source_audit": str((audit_root / "audit.json").resolve()),
        "source_audit_sha256": ae_audit._sha256(audit_root / "audit.json"),
        "fixed_checkpoint_branch": FIXED_BRANCH,
        "scores": list(SCORES),
        "pairings": list(PAIRINGS),
        "selection_directions": list(DIRECTIONS),
        "trajectories": trajectories,
        "checkpoints": checkpoints,
        "primary_hypothesis": {
            "score": "center_oas",
            "pairing": "cdf",
            "direction": "minimize",
            "endpoint": "auprc",
            "rationale": (
                "SVDD explicitly removes normal variation; inverse CAP tests normal-view "
                "invariance, while center-OAS respects anisotropic hypersphere geometry."
            ),
        },
        "complete_reporting": "All 6 x 4 x 2 score/pairing/direction rows are retained.",
        "multiplicity": "Holm correction across all non-random variants per endpoint.",
        "new_score_outcomes_computed_at_design": False,
        "interventions": audit["interventions"],
    }
    _write_json(output, value)
    return output


def _compose(audit: Mapping[str, Any], trajectory: Mapping[str, Any], interventions):
    """Compose the frozen SVDD architecture and requested data streams."""
    os.environ["CCHAMBER_VALID_PAIR_TABLE"] = str(audit["encoder_validation_pair_table"])
    overrides = [
        "experiment=cchamber/svdd_candidate_rank_audit",
        f"seed={int(trajectory['reporting_seed'])}",
        "data.seed=314159",
        f"paths.base_data_dir={audit['data_dir']}",
        f"data.signal_experiments={rank._hydra_value(interventions)}",
        "logger=none",
        *[f"{name}={rank._hydra_value(value)}" for name, value in trajectory["params"].items()],
    ]
    cfg = compose_config(overrides=overrides)
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup(None)
    return cfg, datamodule


@torch.no_grad()
def _representations(model, loader, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned inputs and SVDD embeddings."""
    xs, embeddings = [], []
    model.eval().to(device)
    for batch in loader:
        x = torch.flatten(batch["x"], start_dim=1).to(device)
        xs.append(x.cpu())
        embeddings.append(model.forward(x).cpu())
    return tuple(torch.cat(values).double().numpy() for values in (xs, embeddings))


def _fit_state(embedding: np.ndarray, center: np.ndarray) -> dict[str, np.ndarray]:
    """Fit every train-normal-only SVDD score state."""
    deviation = embedding - center
    centered = OAS(store_precision=True, assume_centered=True).fit(deviation)
    latent = OAS(store_precision=True, assume_centered=False).fit(embedding)
    scale = np.maximum(np.mean(deviation**2, axis=0), np.finfo(np.float64).eps)
    return {
        "center": center,
        "center_precision": centered.precision_,
        "latent_location": latent.location_,
        "latent_precision": latent.precision_,
        "axis_scale": scale,
    }


def _mahalanobis(values, location, precision):
    """Return dimension-normalized squared Mahalanobis energy."""
    centered = values - location
    return np.einsum("bi,ij,bj->b", centered, precision, centered) / values.shape[1]


def score_arrays(embedding: np.ndarray, state: Mapping[str, np.ndarray]):
    """Construct the complete prespecified SVDD score family."""
    center = state["center"]
    deviation = embedding - center
    radial = np.sum(deviation**2, axis=1)
    denominator = np.linalg.norm(embedding, axis=1) * np.linalg.norm(center)
    cosine = np.divide(
        embedding @ center,
        denominator,
        out=np.zeros(len(embedding), dtype=np.float64),
        where=denominator > 0,
    )
    return {
        "radial_d2": radial,
        "log_radial": np.log1p(radial),
        "center_oas": _mahalanobis(deviation, np.zeros_like(center), state["center_precision"]),
        "latent_oas": _mahalanobis(embedding, state["latent_location"], state["latent_precision"]),
        "max_axis": np.max(deviation**2 / state["axis_scale"], axis=1),
        "cosine_center": 1.0 - cosine,
    }


def extract(audit_root: Path, output_root: Path, trajectory_index: int) -> Path:
    """Extract normal representations and all frozen normal-only proxies."""
    if not torch.cuda.is_available():
        raise RuntimeError("SVDD score extraction requires CUDA.")
    _validated_design(output_root)
    audit, trajectories, checkpoints = _inputs(audit_root)
    trajectory = trajectories[trajectory_index]
    source_index = int(trajectory["source_trajectory_index"])
    checkpoint = Path(checkpoints[source_index]["checkpoint"])
    cfg, datamodule = _compose(audit, trajectory, [])
    model = hydra.utils.instantiate(cfg.algorithm)
    rank._load_checkpoint_state(model, checkpoint)
    train_x, train_z = _representations(model, datamodule.train_dataloader(), "cuda")
    del train_x
    state = _fit_state(train_z, model.center.detach().cpu().double().numpy())
    arrays = {f"state_{key}": value for key, value in state.items()}
    for split, loaders in (
        ("valid", datamodule.val_dataloader()),
        ("test", datamodule.test_dataloader()),
    ):
        for stream in ("normal", "reference_normal"):
            x, z = _representations(model, loaders[stream], "cuda")
            arrays[f"{split}_{stream}_x"] = x
            for name, values in score_arrays(z, state).items():
                arrays[f"{split}_{stream}_{name}"] = values
    output = output_root / "normal" / f"{trajectory_index:03d}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    _write_json(
        output.with_suffix(".json"),
        {
            "trajectory_index": trajectory_index,
            "source_trajectory_index": source_index,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": ae_audit._sha256(checkpoint),
            "artifact": str(output),
            "artifact_sha256": ae_audit._sha256(output),
        },
    )
    datamodule.teardown(None)
    return output


def freeze(audit_root: Path, output_root: Path) -> Path:
    """Compute and freeze all normal-only CAP proxies."""
    _validated_design(output_root)
    audit, trajectories, _ = _inputs(audit_root)
    pairing_manifest = _load(Path(audit["pairing_manifest"]))
    rows = []
    for trajectory in trajectories:
        index = int(trajectory["trajectory_index"])
        artifact = output_root / "normal" / f"{index:03d}.npz"
        marker = _load(artifact.with_suffix(".json"))
        if marker["artifact_sha256"] != ae_audit._sha256(artifact):
            raise ValueError("Normal artifact changed before proxy freeze.")
        arrays = np.load(artifact)
        for split in ("valid", "test"):
            table = Path(
                audit["encoder_validation_pair_table"]
                if split == "valid"
                else pairing_manifest["primary_test_table"]
            )
            x1, x2 = arrays[f"{split}_normal_x"], arrays[f"{split}_reference_normal_x"]
            for score in SCORES:
                left = arrays[f"{split}_normal_{score}"]
                right = arrays[f"{split}_reference_normal_{score}"]
                for pairing in PAIRINGS:
                    i1, i2 = ae_audit._pair_indices(
                        pairing,
                        left,
                        right,
                        encoder_table=table,
                        split=split,
                        x_1=x1,
                        x_2=x2,
                    )
                    cap = ae_audit._cap(left[i1], right[i2])
                    rows.append(
                        {
                            "trajectory_index": index,
                            "candidate_id": trajectory["candidate_id"],
                            "reporting_seed": trajectory["reporting_seed"],
                            "split": split,
                            "score": score,
                            "pairing": pairing,
                            "cap": cap,
                        }
                    )
    proxy = output_root / "proxy_metrics.csv"
    ae_audit._write_csv(proxy, rows)
    output = output_root / "proxy_freeze.json"
    _write_json(
        output,
        {
            "design": str(output_root / "design.json"),
            "design_sha256": ae_audit._sha256(output_root / "design.json"),
            "proxy": str(proxy),
            "proxy_sha256": ae_audit._sha256(proxy),
            "intervention_outcomes_inspected_before_proxy_freeze": False,
            "n_rows": len(rows),
        },
    )
    return output


def outcomes(audit_root: Path, output_root: Path, trajectory_index: int) -> Path:
    """Evaluate every score on all interventions after the proxy freeze."""
    if not torch.cuda.is_available():
        raise RuntimeError("SVDD outcome evaluation requires CUDA.")
    _validated_design(output_root)
    freeze_marker = _load(output_root / "proxy_freeze.json")
    if freeze_marker["proxy_sha256"] != ae_audit._sha256(Path(freeze_marker["proxy"])):
        raise ValueError("Frozen proxies changed before outcome evaluation.")
    audit, trajectories, checkpoints = _inputs(audit_root)
    trajectory = trajectories[trajectory_index]
    source_index = int(trajectory["source_trajectory_index"])
    checkpoint = Path(checkpoints[source_index]["checkpoint"])
    cfg, datamodule = _compose(audit, trajectory, audit["interventions"])
    model = hydra.utils.instantiate(cfg.algorithm)
    rank._load_checkpoint_state(model, checkpoint)
    normal_artifact = np.load(output_root / "normal" / f"{trajectory_index:03d}.npz")
    state = {
        key: normal_artifact[f"state_{key}"]
        for key in (
            "center",
            "center_precision",
            "latent_location",
            "latent_precision",
            "axis_scale",
        )
    }
    loaders = datamodule.test_dataloader()
    _, normal_z = _representations(model, loaders["normal"], "cuda")
    normal_scores = score_arrays(normal_z, state)
    thresholds = {name: float(np.quantile(values, 0.99)) for name, values in normal_scores.items()}
    rows = []
    for intervention in audit["interventions"]:
        _, signal_z = _representations(model, loaders[intervention], "cuda")
        signal_scores = score_arrays(signal_z, state)
        for score in SCORES:
            normal = normal_scores[score]
            signal = signal_scores[score]
            prediction = torch.from_numpy(np.concatenate((normal, signal))).float()
            target = torch.cat((torch.zeros(len(normal)), torch.ones(len(signal)))).long()
            rows.append(
                {
                    "trajectory_index": trajectory_index,
                    "candidate_id": trajectory["candidate_id"],
                    "reporting_seed": trajectory["reporting_seed"],
                    "score": score,
                    "intervention": intervention,
                    "auprc": float(BinaryAveragePrecision()(prediction, target)),
                    "efficiency": float(np.mean(signal >= thresholds[score])),
                }
            )
    output = output_root / "outcomes" / f"{trajectory_index:03d}.csv"
    ae_audit._write_csv(output, rows)
    _write_json(
        output.with_suffix(".json"),
        {
            "trajectory_index": trajectory_index,
            "artifact": str(output),
            "artifact_sha256": ae_audit._sha256(output),
            "n_rows": len(rows),
        },
    )
    datamodule.teardown(None)
    return output


def analyze(output_root: Path, permutations: int = 10_000) -> Path:
    """Report all score/pairing/direction associations with Holm correction."""
    import pandas as pd

    _validated_design(output_root)
    proxy = pd.read_csv(output_root / "proxy_metrics.csv", dtype={"candidate_id": str})
    frames = [
        pd.read_csv(path, dtype={"candidate_id": str})
        for path in sorted((output_root / "outcomes").glob("*.csv"))
    ]
    if len(frames) != N_TRAJECTORIES:
        raise ValueError("Outcome coverage is not exactly 48 trajectories.")
    outcome = (
        pd.concat(frames)
        .groupby(["candidate_id", "reporting_seed", "score"])[["auprc", "efficiency"]]
        .mean()
        .reset_index()
    )
    rng = np.random.default_rng(741903)
    rows = []
    for score in SCORES:
        score_outcome = outcome[outcome.score == score]
        for pairing in PAIRINGS:
            valid = proxy[
                (proxy.split == "valid") & (proxy.score == score) & (proxy.pairing == pairing)
            ]
            test = proxy[
                (proxy.split == "test") & (proxy.score == score) & (proxy.pairing == pairing)
            ]
            merged = valid.merge(score_outcome, on=["candidate_id", "reporting_seed"])
            candidate = merged.groupby("candidate_id")[["cap", "auprc", "efficiency"]].mean()
            held = test.groupby("candidate_id").cap.mean().reindex(candidate.index)
            for direction in DIRECTIONS:
                utility = candidate.cap if direction == "maximize" else -candidate.cap
                for endpoint in ("auprc", "efficiency"):
                    observed = float(stats.spearmanr(utility, candidate[endpoint]).statistic)
                    values = candidate[endpoint].to_numpy()
                    exceed = sum(
                        stats.spearmanr(utility, rng.permutation(values)).statistic >= observed
                        for _ in range(permutations)
                    )
                    rows.append(
                        {
                            "score": score,
                            "pairing": pairing,
                            "direction": direction,
                            "endpoint": endpoint,
                            "spearman_rho": observed,
                            "one_sided_permutation_p": (exceed + 1) / (permutations + 1),
                            "validation_test_cap_rho": float(
                                stats.spearmanr(candidate.cap, held).statistic
                            ),
                            "n_candidates": len(candidate),
                        }
                    )
    for endpoint in ("auprc", "efficiency"):
        indices = [
            i
            for i, row in enumerate(rows)
            if row["endpoint"] == endpoint and row["pairing"] != "random"
        ]
        adjusted = ae_audit._holm([rows[i]["one_sided_permutation_p"] for i in indices])
        for index, value in zip(indices, adjusted):
            rows[index]["holm_p"] = value
    for row in rows:
        row.setdefault("holm_p", math.nan)
    output = output_root / "analysis" / "score_cap_rank_associations.csv"
    ae_audit._write_csv(output, rows)
    return output


def _parser() -> argparse.ArgumentParser:
    """Build the stage-oriented command-line parser."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("design")
    create.add_argument("--audit-root", type=Path, required=True)
    create.add_argument("--output-root", type=Path, required=True)
    for name in ("extract", "outcomes"):
        command = sub.add_parser(name)
        command.add_argument("--audit-root", type=Path, required=True)
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--trajectory-index", type=int, required=True)
    freeze_command = sub.add_parser("freeze")
    freeze_command.add_argument("--audit-root", type=Path, required=True)
    freeze_command.add_argument("--output-root", type=Path, required=True)
    analysis = sub.add_parser("analyze")
    analysis.add_argument("--output-root", type=Path, required=True)
    analysis.add_argument("--permutations", type=int, default=10_000)
    return parser


def main() -> None:
    """Dispatch one score-audit stage."""
    args = _parser().parse_args()
    if args.command == "design":
        print(design(args.audit_root, args.output_root))
    elif args.command == "extract":
        print(extract(args.audit_root, args.output_root, args.trajectory_index))
    elif args.command == "freeze":
        print(freeze(args.audit_root, args.output_root))
    elif args.command == "outcomes":
        print(outcomes(args.audit_root, args.output_root, args.trajectory_index))
    else:
        print(analyze(args.output_root, args.permutations))


if __name__ == "__main__":
    main()
