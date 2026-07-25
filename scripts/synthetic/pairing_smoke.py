#!/usr/bin/env python3
"""Exercise encoder training, pair-table creation, and strict CAP consumption."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.utils.pairing.table import (  # noqa: E402
    atomic_json_dump,
    load_pair_table,
    sha256_file,
)


def _run(*args: str) -> None:
    print(f"[pairing-smoke] {' '.join(args)}", flush=True)
    subprocess.run(args, cwd=REPOSITORY_ROOT, check=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "pairing-smoke",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=REPOSITORY_ROOT / "checkpoints",
    )
    parser.add_argument("--trainer", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--skip-encoder-training",
        action="store_true",
        help="Reuse checkpoints/demo/l1_jetclr/last.ckpt.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    checkpoints_dir = args.checkpoints_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoints_dir / "demo" / "l1_jetclr" / "last.ckpt"

    if not args.skip_encoder_training:
        _run(
            sys.executable,
            "tests/train.py",
            "experiment=demo/l1_jetclr",
            f"trainer={args.trainer}",
            f"paths.checkpoints_dir={checkpoints_dir}",
            "logger=none",
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Pairing smoke checkpoint was not created: {checkpoint}")

    common_overrides = ("experiment=demo/l1_jetclr", "logger=none")
    stress_dir = output_dir / "stress"
    _run(
        sys.executable,
        "-m",
        "src.utils.pairing.stress_test_encoder",
        "--ckpt",
        str(checkpoint),
        "--out-dir",
        str(stress_dir),
        "--stage",
        "validate",
        "--dataset-1",
        "normal",
        "--dataset-2",
        "reference_normal",
        "--no-caliper",
        "--device",
        args.device,
        "--overwrite",
        *common_overrides,
    )

    valid_table = output_dir / "valid_pairs.pt"
    test_table = output_dir / "test_pairs.pt"
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
            "mutual_nearest",
            "--k",
            "10",
            "--no-caliper",
            "--device",
            args.device,
            "--overwrite",
            *common_overrides,
        )

    _run(
        sys.executable,
        "tests/train.py",
        "experiment=synthetic/ae",
        f"trainer={args.trainer}",
        f"paths.checkpoints_dir={checkpoints_dir}",
        "seed=123",
        "experiment_name=pairing_smoke",
        "run_name=ae_precomputed",
        "data.n_features=57",
        "data.n_train=256",
        "data.n_val=96",
        "data.n_test=96",
        "data.batch_size=32",
        "data.paired_reliability=0.0",
        "data.reference_shift=0.10",
        "trainer.min_epochs=1",
        "trainer.max_epochs=2",
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
        n_dataset_1=96,
        n_dataset_2=96,
    )
    test = load_pair_table(
        test_table,
        expected_dataset_1="normal",
        expected_dataset_2="reference_normal",
        expected_split="test",
        n_dataset_1=96,
        n_dataset_2=96,
    )
    with (stress_dir / "stress_metrics.json").open(encoding="utf-8") as handle:
        stress_metrics = json.load(handle)
    summary = {
        "status": "passed",
        "encoder_checkpoint": str(checkpoint),
        "encoder_checkpoint_sha256": sha256_file(checkpoint),
        "stress_metrics": stress_metrics,
        "validation_pairs": int(valid["idx_1"].numel()),
        "test_pairs": int(test["idx_1"].numel()),
        "validation_table_sha256": sha256_file(valid_table),
        "test_table_sha256": sha256_file(test_table),
        "cap_consumer_run": str(checkpoints_dir / "pairing_smoke" / "ae_precomputed"),
    }
    summary_path = atomic_json_dump(
        summary,
        output_dir / "summary.json",
        overwrite=True,
    )
    print(f"[pairing-smoke] passed; summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
