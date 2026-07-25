from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from src.utils.pairing.io import load_encoder_run
from src.utils.pairing.table import (
    atomic_json_dump,
    atomic_torch_save,
    sha256_file,
    sha256_tensor,
    validate_pair_table,
)
from src.utils.pairing.utils import (
    closure_metrics,
    collect_closure_representations,
    collect_representations,
    mutual_nearest_pairs,
    pair_table_dict,
    standardized_mean_differences,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stress-test a JetCLR pairing encoder.")
    parser.add_argument("--ckpt", required=True, help="JetCLR checkpoint path.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--config-name", default="train")
    parser.add_argument("--stage", default="validate", choices=["validate", "test"])
    parser.add_argument("--dataset-1", default="normal")
    parser.add_argument("--dataset-2", default="reference_normal")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--caliper-quantile", type=float, default=0.95)
    parser.add_argument("--no-caliper", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--random-seed", type=int, default=12345)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace existing stress-test artifacts.",
    )
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.k <= 0:
        raise ValueError("--k must be positive.")
    if args.max_events is not None and args.max_events <= 0:
        raise ValueError("--max-events must be positive.")
    if not 0.0 <= args.caliper_quantile <= 1.0:
        raise ValueError("--caliper-quantile must be between 0 and 1.")
    cfg, datamodule, model = load_encoder_run(
        args.ckpt,
        config_dir=args.config_dir,
        config_name=args.config_name,
        overrides=args.overrides,
        stage=args.stage,
        device=args.device,
    )
    loaders = (
        datamodule.val_dataloader() if args.stage == "validate" else datamodule.test_dataloader()
    )
    for name in (args.dataset_1, args.dataset_2):
        if name not in loaders:
            raise ValueError(f"Dataset '{name}' not available. Found {list(loaders)}.")

    c1, c2 = collect_closure_representations(
        model, loaders[args.dataset_1], args.device, max_events=args.max_events
    )
    close = closure_metrics(c1, c2)
    closure_distance = 1.0 - F.cosine_similarity(
        F.normalize(c1.float(), dim=1),
        F.normalize(c2.float(), dim=1),
        dim=1,
    )
    caliper = None
    if not args.no_caliper:
        caliper = torch.quantile(closure_distance, args.caliper_quantile).item()

    z1, x1, _ = collect_representations(
        model, loaders[args.dataset_1], args.device, max_events=args.max_events
    )
    z2, x2, _ = collect_representations(
        model, loaders[args.dataset_2], args.device, max_events=args.max_events
    )
    pairs = mutual_nearest_pairs(z1, z2, k=args.k, caliper=caliper)

    smd_before = standardized_mean_differences(x1, x2)
    smd_after = standardized_mean_differences(x1, x2, pairs.idx_1, pairs.idx_2)
    generator = torch.Generator().manual_seed(args.random_seed)
    random_idx2 = torch.randperm(x2.shape[0], generator=generator)[: min(x1.shape[0], x2.shape[0])]
    random_idx1 = torch.arange(random_idx2.shape[0])
    smd_random = standardized_mean_differences(x1, x2, random_idx1, random_idx2)

    metrics = {
        **close,
        "caliper": None if caliper is None else float(caliper),
        "n_dataset_1": int(z1.shape[0]),
        "n_dataset_2": int(z2.shape[0]),
        "mnn_pairs": int(pairs.idx_1.numel()),
        "mnn_coverage": float(pairs.idx_1.numel() / max(min(z1.shape[0], z2.shape[0]), 1)),
        "pair_distance_mean": pairs.distance.mean().item()
        if pairs.distance.numel()
        else float("nan"),
        "pair_distance_p95": torch.quantile(pairs.distance, 0.95).item()
        if pairs.distance.numel()
        else float("nan"),
        "smd_before_mean": smd_before.mean().item() if smd_before.numel() else float("nan"),
        "smd_before_max": smd_before.max().item() if smd_before.numel() else float("nan"),
        "smd_random_mean": smd_random.mean().item() if smd_random.numel() else float("nan"),
        "smd_random_max": smd_random.max().item() if smd_random.numel() else float("nan"),
        "smd_after_mean": smd_after.mean().item() if smd_after.numel() else float("nan"),
        "smd_after_max": smd_after.max().item() if smd_after.numel() else float("nan"),
        "random_seed": args.random_seed,
    }
    if not pairs.idx_1.numel():
        raise RuntimeError("Stress test produced no pairs; refusing to write unusable artifacts.")
    if not all(
        torch.isfinite(torch.tensor(value))
        for value in metrics.values()
        if isinstance(value, float)
    ):
        raise RuntimeError("Stress test produced non-finite metrics.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_metadata = {
        **metrics,
        "n_pairs": int(pairs.idx_1.numel()),
        "encoder_checkpoint_sha256": sha256_file(args.ckpt),
        "source_1_sha256": sha256_tensor(x1),
        "source_2_sha256": sha256_tensor(x2),
        "config_name": args.config_name,
        "config_overrides": list(args.overrides),
        "data_seed": int(cfg.data.seed),
        "embedding_dim": int(z1.shape[1]),
    }
    table = pair_table_dict(
        pairs,
        dataset_1=args.dataset_1,
        dataset_2=args.dataset_2,
        split=args.stage,
        encoder_ckpt=str(Path(args.ckpt).resolve()),
        metadata=table_metadata,
    )
    validate_pair_table(table)
    atomic_json_dump(
        metrics,
        out_dir / "stress_metrics.json",
        overwrite=args.overwrite,
    )
    atomic_torch_save(
        table,
        out_dir / "pair_table.pt",
        overwrite=args.overwrite,
    )

    print(__import__("json").dumps(metrics, indent=2, sort_keys=True, allow_nan=False))
    print(f"Saved stress artifacts to {out_dir}")


if __name__ == "__main__":
    main()
