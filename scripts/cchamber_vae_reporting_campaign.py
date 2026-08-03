#!/usr/bin/env python3
"""Run fresh reporting retrains for the outcome-optimized Causal Chamber VAE.

The exploratory search is immutable and runs at its frozen code commit.  This campaign reads its
final score choice, independently retrains the candidate selected by each label-free criterion, and
evaluates only those four branches.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import average_precision_score

FROZEN_REPOSITORY = Path(
    "/iopsstor/scratch/cscs/vjimenez/adatl1/worktrees/cchamber_vae_multiscore_9b6f14b"
)
FROZEN_CAMPAIGN = FROZEN_REPOSITORY / "scripts" / "cchamber_vae_multiscore_campaign.py"
REPORTING_SEEDS = (2001, 2002, 2003, 2004, 2005)
STRATEGY_NAMES = {
    "cap_encoder": "cap_encoder_nearest",
    "cap_cdf": "cap_cdf",
    "drift": "drift",
    "wasserstein": "wasserstein",
}


def _holm(values: Sequence[float]) -> list[float]:
    """Return Holm-adjusted p-values in their original order."""
    raw = np.asarray(values, dtype=float)
    order = np.argsort(raw)
    adjusted = np.empty_like(raw)
    running = 0.0
    for position, index in enumerate(order):
        running = max(running, (len(raw) - position) * raw[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def _frozen_module():
    """Load the exact implementation used by the frozen search."""
    spec = importlib.util.spec_from_file_location("frozen_vae_campaign", FROZEN_CAMPAIGN)
    if spec is None or spec.loader is None:
        raise ImportError(FROZEN_CAMPAIGN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    """Return the streaming SHA-256 digest of one artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Any:
    """Load one UTF-8 JSON artifact."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any, *, create: bool = False) -> None:
    """Atomically persist one JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("x" if create else "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if create and path.exists():
        temporary.unlink()
        raise FileExistsError(path)
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Atomically persist one non-empty rectangular CSV table."""
    rows = list(rows)
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError("CSV rows do not have one schema.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def initialize(root: Path, search_root: Path) -> Path:
    """Freeze fresh seeds and the selector-specific winning candidates."""
    root = root.expanduser().resolve()
    search_root = search_root.expanduser().resolve()
    output = root / "design.json"
    if output.exists():
        return output
    search_design_path = search_root / "design.json"
    selection_path = search_root / "analysis" / "selection.json"
    performance_path = search_root / "analysis" / "selected_performance.csv"
    search_design = _load(search_design_path)
    selection = _load(selection_path)
    performance = pd.read_csv(performance_path, dtype={"selected_candidate": str})
    performance["selected_candidate"] = performance["selected_candidate"].str.zfill(3)
    score = str(selection["selected_score"])
    auprc = performance[(performance.score == score) & (performance.metric == "auprc")]
    if set(auprc.selector) != set(STRATEGY_NAMES):
        raise ValueError("Search summary lacks all four selected-score criteria.")
    search_trajectories = _load(Path(search_design["trajectory_manifest"]))
    params_by_candidate = {str(row["candidate_id"]): row["params"] for row in search_trajectories}
    selected_candidates = {
        str(row.selector): str(row.selected_candidate) for row in auprc.itertuples()
    }
    trajectories = []
    for candidate in sorted(set(selected_candidates.values())):
        for seed in REPORTING_SEEDS:
            trajectories.append(
                {
                    "trajectory_index": len(trajectories),
                    "candidate_id": candidate,
                    "reporting_seed": seed,
                    "params": params_by_candidate[candidate],
                    "params_sha256": hashlib.sha256(
                        json.dumps(
                            params_by_candidate[candidate], sort_keys=True, separators=(",", ":")
                        ).encode()
                    ).hexdigest(),
                    "pretrained_ae_checkpoint": search_design["ae_initialization"]["checkpoint"],
                    "pretrained_ae_checkpoint_sha256": search_design["ae_initialization"][
                        "checkpoint_sha256"
                    ],
                }
            )
    trajectory_path = root / "trajectories.json"
    _write_json(trajectory_path, trajectories, create=True)
    design = dict(search_design)
    design.update(
        {
            "schema_version": 1,
            "classification": "fresh_reporting_retrains_after_outcome_optimized_search",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "expected_trajectories": len(trajectories),
            "expected_checkpoints": len(trajectories) * len(search_design["branches"]),
            "reporting_seeds": list(REPORTING_SEEDS),
            "trajectory_manifest": str(trajectory_path),
            "trajectory_manifest_sha256": _sha256(trajectory_path),
            "search_root": str(search_root),
            "search_design_sha256": _sha256(search_design_path),
            "search_selection": str(selection_path),
            "search_selection_sha256": _sha256(selection_path),
            "selected_score": score,
            "selected_cap_selector": selection["selected_cap_selector"],
            "selected_candidates": selected_candidates,
            "fresh_reporting_seeds": list(REPORTING_SEEDS),
            "intervention_outcomes_inspected_before_checkpoint_freeze": True,
        }
    )
    _write_json(output, design, create=True)
    return output


def _design(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load and authenticate a frozen reporting design."""
    design = _load(root / "design.json")
    trajectories = _load(Path(design["trajectory_manifest"]))
    if _sha256(Path(design["trajectory_manifest"])) != design["trajectory_manifest_sha256"]:
        raise ValueError("Reporting trajectory manifest changed.")
    if _sha256(Path(design["search_selection"])) != design["search_selection_sha256"]:
        raise ValueError("Search selection changed after reporting design freeze.")
    return design, trajectories


def canary(root: Path) -> Path:
    """Run the frozen implementation's complete two-epoch GPU canary."""
    return _frozen_module().canary(root.expanduser().resolve())


def train(root: Path, trajectory_index: int) -> Path:
    """Train one fresh reporting trajectory with the frozen implementation."""
    return _frozen_module().train(root.expanduser().resolve(), int(trajectory_index))


def freeze(root: Path) -> Path:
    """Freeze every fresh checkpoint before evaluating reporting outcomes."""
    root = root.expanduser().resolve()
    design, trajectories = _design(root)
    output = root / "checkpoint_manifest.json"
    if output.exists():
        manifest = _load(output)
        records = manifest.get("checkpoints", [])
        if len(records) != int(design["expected_checkpoints"]):
            raise ValueError("Existing reporting checkpoint coverage is incomplete.")
        for record in records:
            if _sha256(Path(record["checkpoint"])) != record["checkpoint_sha256"]:
                raise ValueError("Existing reporting checkpoint hash changed.")
        return output
    records = []
    for trajectory in trajectories:
        marker = _load(root / "training" / f"{trajectory['trajectory_index']:03d}.json")
        if marker["candidate_id"] != trajectory["candidate_id"]:
            raise ValueError("Reporting training marker identity mismatch.")
        for branch in marker["branches"]:
            checkpoint = Path(branch["checkpoint"])
            if _sha256(checkpoint) != branch["checkpoint_sha256"]:
                raise ValueError("Fresh reporting checkpoint changed before freeze.")
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
        raise ValueError("Fresh reporting checkpoint coverage is incomplete.")
    _write_json(
        output,
        {
            "schema_version": 1,
            "design_sha256": _sha256(root / "design.json"),
            "code_commit": design["code_commit"],
            "search_outcomes_used_to_define_reporting_configuration": True,
            "fresh_reporting_outcomes_inspected_before_freeze": False,
            "checkpoints": records,
        },
        create=True,
    )
    return output


def evaluate(root: Path, trajectory_index: int) -> Path:
    """Evaluate the selected score/selector branches for one fresh trajectory."""
    if "SLURM_JOB_ID" not in os.environ or not torch.cuda.is_available():
        raise RuntimeError("Reporting evaluation requires a CUDA Slurm allocation.")
    root = root.expanduser().resolve()
    design, trajectories = _design(root)
    trajectory = trajectories[int(trajectory_index)]
    marker = root / "evaluation" / f"{int(trajectory_index):03d}.json"
    if marker.exists():
        return marker
    frozen = _frozen_module()
    manifest = _load(root / "checkpoint_manifest.json")
    selectors = [
        selector
        for selector, candidate in design["selected_candidates"].items()
        if candidate == trajectory["candidate_id"]
    ]
    branches = [
        row
        for row in manifest["checkpoints"]
        if int(row["trajectory_index"]) == int(trajectory_index)
        and row["score"] == design["selected_score"]
        and row["selector"] in selectors
    ]
    if {row["selector"] for row in branches} != set(selectors):
        raise ValueError("Reporting evaluation branch coverage is incomplete.")
    datamodule, model = frozen._compose(design, trajectory)
    validation = datamodule.val_dataloader()
    test = datamodule.test_dataloader()
    device = torch.device("cuda")
    rows = []
    for branch in branches:
        checkpoint = Path(branch["checkpoint"])
        if _sha256(checkpoint) != branch["checkpoint_sha256"]:
            raise ValueError("Reporting checkpoint changed after freeze.")
        frozen.rank._load_checkpoint_state(model, checkpoint)
        score_name = str(branch["score"])
        validation_normal = frozen._scores(model, validation["normal"], score_name, device)
        threshold = float(np.quantile(validation_normal, 0.99))
        normal = frozen._scores(model, test["normal"], score_name, device)
        for intervention in design["interventions"]:
            signal = frozen._scores(model, test[intervention], score_name, device)
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
                        "model": "vae",
                        "candidate_id": trajectory["candidate_id"],
                        "reporting_seed": int(trajectory["reporting_seed"]),
                        "score": score_name,
                        "strategy": STRATEGY_NAMES[branch["selector"]],
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
    expected = len(selectors) * len(design["interventions"]) * 2
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} reporting rows, found {len(rows)}.")
    _write_json(
        marker,
        {
            "schema_version": 1,
            "trajectory_index": int(trajectory_index),
            "rows": str(output),
            "rows_sha256": _sha256(output),
            "n_rows": len(rows),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        create=True,
    )
    return marker


def collect(root: Path) -> tuple[Path, Path]:
    """Collect and summarize the complete five-seed reporting matrix."""
    root = root.expanduser().resolve()
    design, trajectories = _design(root)
    rows = []
    for trajectory in trajectories:
        marker = _load(root / "evaluation" / f"{trajectory['trajectory_index']:03d}.json")
        path = Path(marker["rows"])
        if _sha256(path) != marker["rows_sha256"]:
            raise ValueError(f"Reporting rows changed: {path}")
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    expected = len(STRATEGY_NAMES) * len(REPORTING_SEEDS) * len(design["interventions"]) * 2
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} collected reporting rows, found {len(rows)}.")
    results = root / "results" / "evaluation_rows.csv"
    _write_csv(results, rows)
    frame = pd.DataFrame(rows)
    frame["value"] = frame["value"].astype(float)
    seed_first = (
        frame.groupby(["model", "score", "strategy", "reporting_seed", "metric"], sort=True)
        .value.mean()
        .reset_index()
    )
    summary_rows = []
    for keys, group in seed_first.groupby(["model", "score", "strategy", "metric"], sort=True):
        values = group.value.to_numpy(float)
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        half_width = float(stats.t.ppf(0.975, len(values) - 1) * sd / math.sqrt(len(values)))
        selector = next(key for key, name in STRATEGY_NAMES.items() if name == keys[2])
        summary_rows.append(
            {
                "model": keys[0],
                "score": keys[1],
                "strategy": keys[2],
                "metric": keys[3],
                "mean": mean,
                "sd": sd,
                "ci_low": mean - half_width,
                "ci_high": mean + half_width,
                "n_reporting_seeds": len(values),
                "selected_candidate": design["selected_candidates"][selector],
            }
        )
    summary = root / "analysis" / "selected_strategy_summary.csv"
    _write_csv(summary, summary_rows)
    selection_rows = []
    for selector, candidate in design["selected_candidates"].items():
        selection_rows.append(
            {
                "score": design["selected_score"],
                "strategy": STRATEGY_NAMES[selector],
                "selected_candidate": candidate,
                "search_root": design["search_root"],
                "n_fresh_reporting_seeds": len(REPORTING_SEEDS),
            }
        )
    selection = root / "analysis" / "selection_manifest.csv"
    _write_csv(selection, selection_rows)
    search_root = Path(design["search_root"])
    search_checkpoints = pd.read_csv(
        search_root / "results" / "checkpoint_rows.csv", dtype={"candidate_id": str}
    )
    search_outcomes = pd.read_csv(
        search_root / "results" / "evaluation_rows.csv", dtype={"candidate_id": str}
    )
    for search_frame in (search_checkpoints, search_outcomes):
        search_frame["candidate_id"] = search_frame["candidate_id"].str.zfill(3)
    search_checkpoints = search_checkpoints[search_checkpoints.score == design["selected_score"]]
    search_outcomes = search_outcomes[search_outcomes.score == design["selected_score"]]
    search_seed_first = (
        search_outcomes.groupby(
            ["candidate_id", "reporting_seed", "selector", "metric"], sort=True
        )
        .value.mean()
        .reset_index()
    )
    rng = np.random.default_rng(2_608_2026)
    association_rows = []
    for selector, direction in design["selectors"].items():
        proxy = (
            search_checkpoints[search_checkpoints.selector == selector]
            .groupby("candidate_id")
            .monitor_value.mean()
        )
        utility = proxy if direction == "maximize" else -proxy
        for metric in ("auprc", "efficiency_operational"):
            outcome = (
                search_seed_first[
                    (search_seed_first.selector == selector) & (search_seed_first.metric == metric)
                ]
                .groupby("candidate_id")
                .value.mean()
                .reindex(utility.index)
            )
            observed = float(stats.spearmanr(utility, outcome).statistic)
            exceed = 0
            for _ in range(10_000):
                permuted = rng.permutation(outcome.to_numpy(float))
                exceed += int(float(stats.spearmanr(utility, permuted).statistic) >= observed)
            association_rows.append(
                {
                    "model": "vae",
                    "score": design["selected_score"],
                    "strategy": STRATEGY_NAMES[selector],
                    "metric": metric,
                    "direction": direction,
                    "spearman_rho": observed,
                    "one_sided_permutation_p": (exceed + 1) / 10_001,
                    "n_candidates": len(utility),
                }
            )
    associations = pd.DataFrame(association_rows)
    for metric in ("auprc", "efficiency_operational"):
        indices = associations.index[associations.metric == metric].tolist()
        associations.loc[indices, "holm_p"] = _holm(
            associations.loc[indices, "one_sided_permutation_p"].tolist()
        )
    associations.to_csv(root / "analysis" / "candidate_rank_associations.csv", index=False)
    return results, summary


def _parser() -> argparse.ArgumentParser:
    """Build the stage-oriented command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    init = subparsers.add_parser("init")
    init.add_argument("--root", type=Path, required=True)
    init.add_argument("--search-root", type=Path, required=True)
    for name in ("canary", "freeze", "collect"):
        command = subparsers.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
    for name in ("train", "evaluate"):
        command = subparsers.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--trajectory-index", type=int, required=True)
    return parser


def main() -> None:
    """Dispatch one reporting-campaign stage."""
    args = _parser().parse_args()
    if args.command == "init":
        print(initialize(args.root, args.search_root))
    elif args.command == "canary":
        print(canary(args.root))
    elif args.command == "train":
        print(train(args.root, args.trajectory_index))
    elif args.command == "freeze":
        print(freeze(args.root))
    elif args.command == "evaluate":
        print(evaluate(args.root, args.trajectory_index))
    else:
        print(*collect(args.root), sep="\n")


if __name__ == "__main__":
    main()
