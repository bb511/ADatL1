#!/usr/bin/env python3
"""Exercise fixed encoder pairing on the real public Causal Chamber CSVs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.utils.pairing.table import (  # noqa: E402
    atomic_json_dump,
    load_pair_table,
    sha256_file,
)


def _run(*args: str) -> None:
    print(f"[cchamber-pairing-smoke] {' '.join(args)}", flush=True)
    subprocess.run(args, cwd=REPOSITORY_ROOT, check=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "cchamber-pairing-smoke",
    )
    parser.add_argument(
        "--skip-encoder-training",
        action="store_true",
        help="Reuse checkpoints/cchamber_pairing_smoke/encoder/last.ckpt.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = (
        REPOSITORY_ROOT / "checkpoints" / "cchamber_pairing_smoke" / "encoder" / "last.ckpt"
    )

    if not args.skip_encoder_training:
        _run(
            sys.executable,
            "src/train.py",
            "experiment=cchamber/ae_pairing",
            "seed=123",
            "experiment_name=cchamber_pairing_smoke",
            "run_name=encoder",
            "logger=none",
            "extras.print_config=false",
            "trainer.min_epochs=1",
            "trainer.max_epochs=1",
            "+trainer.limit_train_batches=4",
            "+trainer.limit_val_batches=3",
            "+trainer.limit_test_batches=3",
            "data.max_val_batches=3",
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Causal Chamber encoder checkpoint is missing: {checkpoint}")

    valid_table = output_dir / "valid_pairs.pt"
    test_table = output_dir / "test_pairs.pt"
    table_overrides = (
        "experiment=cchamber/ae_pairing",
        "data.pairing_strategy=random",
        "data.max_val_batches=3",
        "data.signal_experiments=[]",
    )
    for stage, destination in (("validate", valid_table), ("test", test_table)):
        _run(
            sys.executable,
            "-m",
            "src.utils.pairing.build_pair_table",
            "--ckpt",
            str(checkpoint),
            "--out",
            str(destination),
            "--stage",
            stage,
            "--dataset-1",
            "normal",
            "--dataset-2",
            "reference_normal",
            "--pairing-mode",
            "one_to_one_nearest",
            "--k",
            "0",
            "--no-caliper",
            "--overwrite",
            *table_overrides,
        )

    _run(
        sys.executable,
        "src/train.py",
        "experiment=cchamber/ae_agnostic",
        "seed=123",
        "experiment_name=cchamber_pairing_smoke",
        "run_name=consumer",
        "logger=none",
        "extras.print_config=false",
        "data.pairing_strategy=random",
        "data.max_val_batches=3",
        "data.signal_experiments=[uniform_red_mid]",
        "trainer.min_epochs=1",
        "trainer.max_epochs=1",
        "+trainer.limit_train_batches=4",
        "+trainer.limit_val_batches=3",
        "+trainer.limit_test_batches=3",
        "callbacks.cap_ref.pairing_type=precomputed",
        f"+callbacks.cap_ref.pairing_index_path={valid_table}",
        "callbacks.cap_ref.cap_metric_config.n_epochs=2",
        "evaluation.callbacks.cap_ref.pairing_type=precomputed",
        f"+evaluation.callbacks.cap_ref.pairing_index_path={valid_table}",
        f"+evaluation.callbacks.cap_ref.pairing_test_index_path={test_table}",
        "evaluation.callbacks.cap_ref.cap_metric_config.n_epochs=2",
    )

    valid = load_pair_table(
        valid_table,
        expected_dataset_1="normal",
        expected_dataset_2="reference_normal",
        expected_split="validate",
        n_dataset_1=1000,
        n_dataset_2=1000,
    )
    test = load_pair_table(
        test_table,
        expected_dataset_1="normal",
        expected_dataset_2="reference_normal",
        expected_split="test",
        n_dataset_1=1000,
        n_dataset_2=1000,
    )
    summary_path = atomic_json_dump(
        {
            "status": "passed",
            "dataset": "lt_interventions_standard_v1",
            "feature_set": "readouts",
            "n_features": 11,
            "signal_canary": "uniform_red_mid",
            "encoder_checkpoint": str(checkpoint),
            "encoder_checkpoint_sha256": sha256_file(checkpoint),
            "validation_pairs": int(valid["idx_1"].numel()),
            "test_pairs": int(test["idx_1"].numel()),
            "validation_table_sha256": sha256_file(valid_table),
            "test_table_sha256": sha256_file(test_table),
            "consumer_run": str(
                REPOSITORY_ROOT / "checkpoints" / "cchamber_pairing_smoke" / "consumer"
            ),
        },
        output_dir / "summary.json",
        overwrite=True,
    )
    print(f"[cchamber-pairing-smoke] passed; summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
