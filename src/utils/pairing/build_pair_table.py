from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.utils.pairing.io import load_encoder_run
from src.utils.pairing.utils import (
    closure_metrics,
    collect_closure_representations,
    collect_representations,
    mutual_nearest_pairs,
    pair_table_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a frozen-encoder pair table.")
    parser.add_argument("--ckpt", required=True, help="JetCLR checkpoint path.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--config-name", default="train")
    parser.add_argument("--stage", default="validate", choices=["validate", "test"])
    parser.add_argument("--dataset-1", default="normal")
    parser.add_argument("--dataset-2", default="reference_normal")
    parser.add_argument("--out", required=True)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--caliper", type=float, default=None)
    parser.add_argument("--caliper-quantile", type=float, default=0.95)
    parser.add_argument("--no-caliper", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg, datamodule, model = load_encoder_run(
        args.ckpt,
        config_dir=args.config_dir,
        config_name=args.config_name,
        overrides=args.overrides,
        stage=args.stage,
        device=args.device,
    )
    loaders = datamodule.val_dataloader() if args.stage == "validate" else datamodule.test_dataloader()

    if args.dataset_1 not in loaders or args.dataset_2 not in loaders:
        raise ValueError(
            f"Requested datasets ({args.dataset_1}, {args.dataset_2}) not in "
            f"available loaders {list(loaders)}."
        )

    z1, _, _ = collect_representations(
        model, loaders[args.dataset_1], args.device, max_events=args.max_events
    )
    z2, _, _ = collect_representations(
        model, loaders[args.dataset_2], args.device, max_events=args.max_events
    )

    caliper = args.caliper
    metadata = {}
    if args.no_caliper:
        caliper = None
    elif caliper is None:
        c1, c2 = collect_closure_representations(
            model, loaders[args.dataset_1], args.device, max_events=args.max_events
        )
        close = closure_metrics(c1, c2)
        caliper = torch.quantile(
            1.0
            - torch.nn.functional.cosine_similarity(
                torch.nn.functional.normalize(c1.float(), dim=1),
                torch.nn.functional.normalize(c2.float(), dim=1),
                dim=1,
            ),
            args.caliper_quantile,
        ).item()
        metadata.update(close)

    pairs = mutual_nearest_pairs(z1, z2, k=args.k, caliper=caliper)
    metadata.update(
        {
            "k": args.k,
            "caliper": None if caliper is None else float(caliper),
            "max_events": args.max_events,
            "n_dataset_1": int(z1.shape[0]),
            "n_dataset_2": int(z2.shape[0]),
            "n_pairs": int(pairs.idx_1.numel()),
            "coverage": float(pairs.idx_1.numel() / max(min(z1.shape[0], z2.shape[0]), 1)),
        }
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        pair_table_dict(
            pairs,
            dataset_1=args.dataset_1,
            dataset_2=args.dataset_2,
            split=args.stage,
            encoder_ckpt=str(Path(args.ckpt).resolve()),
            metadata=metadata,
        ),
        out,
    )
    print(f"Saved {pairs.idx_1.numel()} pairs to {out}")
    print(metadata)


if __name__ == "__main__":
    main()
