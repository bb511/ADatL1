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
    one_to_one_nearest_pairs,
    pair_table_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a frozen-encoder pair table.")
    parser.add_argument("--ckpt", required=True, help="Frozen encoder checkpoint path.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--config-name", default="train")
    parser.add_argument("--stage", default="validate", choices=["validate", "test"])
    parser.add_argument("--dataset-1", default="normal")
    parser.add_argument("--dataset-2", default="reference_normal")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--k",
        type=int,
        default=20,
        help="Top-k neighbors to search. For one_to_one_nearest, k<=0 grows k until coverage saturates.",
    )
    parser.add_argument(
        "--pairing-mode",
        default="mutual_nearest",
        choices=["mutual_nearest", "one_to_one_nearest"],
        help="Nearest-neighbor matching rule used to turn embeddings into fixed pairs.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Do not L2-normalize embeddings before one_to_one_nearest matching.",
    )
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
    loaders = (
        datamodule.val_dataloader() if args.stage == "validate" else datamodule.test_dataloader()
    )

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
    elif caliper is None and hasattr(model, "augment_pair"):
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
    elif caliper is None:
        caliper = None
        metadata["caliper_reason"] = "disabled: encoder has no augment_pair method"

    k_arg = None if args.pairing_mode == "one_to_one_nearest" and args.k <= 0 else args.k
    if args.pairing_mode == "mutual_nearest":
        if args.k <= 0:
            raise ValueError("--k must be positive for mutual_nearest pairing.")
        pairs = mutual_nearest_pairs(z1, z2, k=args.k, caliper=caliper)
    else:
        pairs = one_to_one_nearest_pairs(
            z1,
            z2,
            k=k_arg,
            caliper=caliper,
            normalize=not args.no_normalize,
        )
    metadata.update(
        {
            "pairing_mode": args.pairing_mode,
            "normalized": bool(
                args.pairing_mode == "one_to_one_nearest" and not args.no_normalize
            ),
            "k": k_arg,
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
