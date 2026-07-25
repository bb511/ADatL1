#!/usr/bin/env python3
"""Smoke-test the checkpoint/evaluation/aggregation workflow on synthetic data."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import paper_pipeline  # noqa: E402

MODELS = ("ae", "vae", "svdd", "realnvp")
CHECKPOINT_RESULTS = {
    "last": Path("plots/test/last/auprc/values.csv"),
    "drift": Path(
        "summary/operational_drift_ema/min/plots/test/" "operational_drift_ema/auprc/values.csv"
    ),
    "wasserstein": Path(
        "summary/w1dist_ema_normal_vs_reference_normal/min/plots/test/"
        "w1dist_ema_normal_vs_reference_normal/auprc/values.csv"
    ),
    "cap": Path(
        "summary/cap_ema_normal_vs_reference_normal/max/plots/test/"
        "cap_ema_normal_vs_reference_normal/auprc/values.csv"
    ),
}


def _csv_items(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated value.")
    return items


def _seeds(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in _csv_items(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Seeds must be integers.") from exc


def train_one(model: str, seed: int) -> None:
    """Run a bounded but structurally complete smoke training job."""
    command = [
        sys.executable,
        "tests/train.py",
        f"experiment=synthetic/{model}",
        f"seed={seed}",
        "trainer.max_epochs=3",
        "+trainer.limit_train_batches=2",
        "+trainer.limit_val_batches=2",
        "+trainer.limit_test_batches=2",
        "callbacks.cap_ref.cap_metric_config.n_epochs=2",
        "evaluation.callbacks.cap_ref.cap_metric_config.n_epochs=2",
    ]
    print(f"[smoke] model={model} seed={seed}", flush=True)
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def build_collect_manifest(
    models: Sequence[str],
    seeds: Sequence[int],
    checkpoints_dir: Path,
) -> pd.DataFrame:
    """Describe raw callback values for all checkpoint-selection strategies."""
    rows = []
    for model in models:
        for seed in seeds:
            run_dir = checkpoints_dir / "synthetic_smoke" / f"{model}_seed{seed}"
            for strategy, relative_path in CHECKPOINT_RESULTS.items():
                path = run_dir / relative_path
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Missing {strategy} callback values for {model}, seed {seed}: " f"{path}"
                    )
                rows.append(
                    {
                        "path": str(path.resolve()),
                        "dataset": "synthetic",
                        "model": model,
                        "strategy": strategy,
                        "seed": seed,
                    }
                )
    return pd.DataFrame(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train synthetic AE/VAE/SVDD/RealNVP jobs, collect callback CSVs, "
            "and smoke-test seed-aware aggregation tables and plots."
        )
    )
    parser.add_argument("--models", type=_csv_items, default=MODELS)
    parser.add_argument("--seeds", type=_seeds, default=(123, 456, 789))
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=REPOSITORY_ROOT / "checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "synthetic-smoke",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Collect and aggregate already completed smoke runs.",
    )
    args = parser.parse_args(argv)
    unknown_models = sorted(set(args.models) - set(MODELS))
    if unknown_models:
        parser.error(f"Unsupported models: {', '.join(unknown_models)}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoints_dir = args.checkpoints_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not args.skip_training:
        for model in args.models:
            for seed in args.seeds:
                train_one(model, seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "callback_manifest.csv"
    build_collect_manifest(args.models, args.seeds, checkpoints_dir).to_csv(
        manifest_path, index=False
    )

    results_path = output_dir / "results.csv"
    paper_pipeline.collect_results(manifest_path, results_path)
    written = paper_pipeline.aggregate_results(
        results_path,
        output_dir / "report",
        main_metric="auprc",
    )
    print(f"[smoke] report: {output_dir / 'report' / 'report.md'}")
    print(f"[smoke] generated {len(written) + 3} result artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
