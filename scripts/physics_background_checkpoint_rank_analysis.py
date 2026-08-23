#!/usr/bin/env python3
"""Compare validation-metric rankings on common physics checkpoint universes."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch


METRICS = {
    "cap_flat": ("cap", "cap_native_flat_physical", 1.0),
    "cap_summary": ("cap", "cap_native_physics_summary", 1.0),
    "cap_typed": ("cap", "cap_native_typed_sliced_wasserstein", 1.0),
    "cap_jetclr": ("cap", "cap_native_jetclr", 1.0),
    "cap_cdf": ("cap", "cap_native_cdf", 1.0),
    "w1": ("w1", "native", -1.0),
    "drift": ("drift", "native", -1.0),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_native_endpoints(path: Path, checkpoint_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "score" in frame.columns:
        frame = frame.loc[frame["score"] == "native"].copy()
    required = {"model", checkpoint_column, "checkpoint_sha256", "anomaly", "endpoint", "value"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks endpoint columns: {sorted(missing)}")
    frame = frame.rename(columns={checkpoint_column: "checkpoint_path"})
    identity = ["model", "checkpoint_path", "checkpoint_sha256", "anomaly", "endpoint"]
    conflicts = frame.groupby(identity, dropna=False)["value"].nunique()
    if (conflicts > 1).any():
        raise ValueError(f"Conflicting duplicate endpoint values in {path}.")
    frame = frame.drop_duplicates(identity)
    counts = frame.groupby(["model", "checkpoint_sha256", "endpoint"])["anomaly"].nunique()
    if set(counts) != {20}:
        raise ValueError(f"Every checkpoint/endpoint must contain exactly 20 anomalies in {path}.")
    return (
        frame.groupby(
            ["model", "checkpoint_path", "checkpoint_sha256", "endpoint"], as_index=False
        )["value"]
        .mean()
        .pivot(
            index=["model", "checkpoint_path", "checkpoint_sha256"],
            columns="endpoint",
            values="value",
        )
        .reset_index()
    )


def _extract_metric_values(checkpoint: Path) -> dict[str, float]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    callbacks = payload.get("callbacks")
    if not isinstance(callbacks, dict):
        raise ValueError(f"Checkpoint has no callback state: {checkpoint}")
    extracted: dict[str, float] = {}
    for label, (kind, metric_name, sign) in METRICS.items():
        matches: list[float] = []
        for state_key, state in callbacks.items():
            key = str(state_key)
            if "'output_name': 'ascore/full'" not in key or not isinstance(state, dict):
                continue
            if kind == "cap":
                if (
                    not key.startswith("CAPCallback")
                    or f"'metric_name': '{metric_name}'" not in key
                ):
                    continue
                value = state.get("cap_ema")
            elif kind == "w1":
                if (
                    not key.startswith("WassersteinCallback")
                    or "'metric_name': 'native'" not in key
                ):
                    continue
                value = state.get("w1dist_ema")
            else:
                if (
                    not key.startswith("ThresholdDriftCallback")
                    or "'metric_name': 'native'" not in key
                ):
                    continue
                drift = state.get("drift_ema")
                value = drift.get("0_25") if isinstance(drift, dict) else None
            if value is not None:
                matches.append(sign * float(value))
        if len(matches) != 1 or not math.isfinite(matches[0]):
            raise ValueError(
                f"Expected one finite {label} callback state in {checkpoint}; got {matches}."
            )
        extracted[label] = matches[0]
    del payload
    gc.collect()
    return extracted


def _checkpoint_table(endpoints: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for index, row in endpoints.iterrows():
        checkpoint = Path(str(row["checkpoint_path"])).resolve(strict=True)
        if _sha256(checkpoint) != row["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {checkpoint}")
        record = row.to_dict()
        record.update(_extract_metric_values(checkpoint))
        records.append(record)
        if (index + 1) % 25 == 0:
            print(f"loaded {index + 1}/{len(endpoints)} checkpoints", flush=True)
    return pd.DataFrame.from_records(records)


def _analyse(stage: str, checkpoints: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail: list[dict[str, Any]] = []
    for model, model_frame in checkpoints.groupby("model", sort=True):
        n = len(model_frame)
        k = max(2, math.ceil(0.2 * n))
        for metric in METRICS:
            metric_values = model_frame[metric]
            for endpoint in ("efficiency_250hz", "auprc"):
                downstream = model_frame[endpoint]
                rho = metric_values.corr(downstream, method="spearman")
                ordered = model_frame.sort_values(metric, ascending=False, kind="mergesort")
                top_threshold = float(ordered.iloc[k - 1][metric])
                top = model_frame.loc[model_frame[metric] >= top_threshold]
                top1 = model_frame.loc[model_frame[metric] == metric_values.max()]
                oracle = float(downstream.max())
                enrichment = float(top[endpoint].mean() - downstream.mean())
                top1_downstream = float(top1[endpoint].mean())
                regret = float(oracle - top1_downstream)
                detail.append(
                    {
                        "stage": stage,
                        "model": model,
                        "n_checkpoints": n,
                        "top_k": k,
                        "top_k_with_boundary_ties": len(top),
                        "top1_ties": len(top1),
                        "metric": metric,
                        "endpoint": endpoint,
                        "spearman_rho": float(rho),
                        "top20_enrichment": enrichment,
                        "top1_downstream": top1_downstream,
                        "oracle_best": oracle,
                        "top1_regret": regret,
                    }
                )
    detail_frame = pd.DataFrame.from_records(detail)
    macro = (
        detail_frame.groupby(["stage", "metric", "endpoint"], as_index=False)
        .agg(
            models=("model", "nunique"),
            mean_spearman=("spearman_rho", "mean"),
            median_spearman=("spearman_rho", "median"),
            positive_models=("spearman_rho", lambda values: int((values > 0).sum())),
            mean_top20_enrichment=("top20_enrichment", "mean"),
            mean_top1_regret=("top1_regret", "mean"),
        )
        .sort_values(["endpoint", "mean_spearman"], ascending=[True, False])
    )
    return detail_frame, macro


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-endpoints", type=Path, required=True)
    parser.add_argument("--confirmation-endpoints", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inputs = {
        "discovery": (args.discovery_endpoints, "checkpoint"),
        "confirmation": (args.confirmation_endpoints, "checkpoint_path"),
    }
    detail_frames: list[pd.DataFrame] = []
    macro_frames: list[pd.DataFrame] = []
    inventories: dict[str, Any] = {}
    for stage, (path, column) in inputs.items():
        endpoints = _read_native_endpoints(path.resolve(strict=True), column)
        print(f"{stage}: {len(endpoints)} unique native checkpoints", flush=True)
        checkpoints = _checkpoint_table(endpoints)
        checkpoints.to_csv(args.output_dir / f"{stage}_checkpoint_values.csv", index=False)
        detail, macro = _analyse(stage, checkpoints)
        detail_frames.append(detail)
        macro_frames.append(macro)
        inventories[stage] = {
            "endpoint_path": str(path.resolve()),
            "endpoint_sha256": _sha256(path.resolve()),
            "checkpoints": len(checkpoints),
            "models": {
                key: int(value) for key, value in checkpoints.groupby("model").size().items()
            },
        }
    detail = pd.concat(detail_frames, ignore_index=True)
    macro = pd.concat(macro_frames, ignore_index=True)
    detail_path = args.output_dir / "per_model_rank_diagnostics.csv"
    macro_path = args.output_dir / "macro_rank_diagnostics.csv"
    detail.to_csv(detail_path, index=False)
    macro.to_csv(macro_path, index=False)
    receipt = {
        "schema_version": 1,
        "kind": "physics_background_common_checkpoint_rank_analysis",
        "metric_direction": "all validation metrics oriented so higher is better",
        "downstream_aggregation": "unweighted mean over 20 anomalies per physical checkpoint",
        "top_k": "ceil(20% of each model universe), minimum 2; include boundary ties",
        "top1_ties": "regret uses mean downstream performance across validation-metric ties",
        "universes": inventories,
        "outputs": {
            detail_path.name: _sha256(detail_path),
            macro_path.name: _sha256(macro_path),
            **{
                f"{stage}_checkpoint_values.csv": _sha256(
                    args.output_dir / f"{stage}_checkpoint_values.csv"
                )
                for stage in inputs
            },
        },
    }
    receipt_path = args.output_dir / "analysis.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(macro.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
