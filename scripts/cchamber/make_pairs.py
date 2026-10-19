#!/usr/bin/env python3
"""Store the two constructed pairings of the registered pairing comparison.

Section ``pairing`` of ``registration_limits.yaml`` sets the rank (cdf) pairing of
validation ``normal`` to ``reference_normal`` against two pairings computed once from
the data module's inputs and used at every epoch:

    nearest_neighbour  the one-to-one assignment of rows that minimises the total
                       Euclidean distance in the normalised input space (Hungarian
                       assignment)
    random             one permutation of reference_normal, drawn with ``--seed``

Each is written to ``--out`` as ``pairs_<name>.pt``, a dict of two index tensors,
``normal`` and ``reference_normal``, both (n,), and the ``seed`` (None for the
assignment). The indices count rows in loader order, which is the order in which
``src/callbacks/score_dump.py`` stores the scores; ``offline_select.py --pairs`` reads
the files. ``--data`` names the data config group: ``cchamber``, the light tunnel, by
default, or ``cchamber_wt``, the wind tunnel.

Usage
-----
    python scripts/cchamber/make_pairs.py --data-dir data/causal_chamber \\
        --seed 4242 --out pairs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_inputs(
    data_dir: str, data: str = "cchamber"
) -> dict[str, dict[str, torch.Tensor]]:
    """Normalised inputs of every validation and test set, (n, 11) in loader order."""
    overrides = [f"data={data}", f"data.data_dir={data_dir}"]
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="train.yaml", overrides=overrides)
    datamodule = instantiate(cfg.data)
    datamodule.setup("validate")
    loaders = {"val": datamodule.val_dataloader(), "test": datamodule.test_dataloader()}
    return {
        split: {
            name: torch.cat([x for x, _ in loader]) for name, loader in sets.items()
        }
        for split, sets in loaders.items()
    }


def nearest_neighbour(
    x_normal: torch.Tensor, x_reference: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cost = torch.cdist(x_normal.double(), x_reference.double()).numpy()
    rows, cols = linear_sum_assignment(cost)
    return torch.as_tensor(rows), torch.as_tensor(cols)


def random_pairs(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    permutation = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    return torch.arange(n), permutation


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="cchamber")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    val = load_inputs(args.data_dir, args.data)["val"]
    x_normal, x_reference = val["normal"], val["reference_normal"]
    pairs = {
        "nearest_neighbour": (*nearest_neighbour(x_normal, x_reference), None),
        "random": (*random_pairs(len(x_reference), args.seed), args.seed),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for name, (normal, reference, seed) in pairs.items():
        record = {"normal": normal, "reference_normal": reference, "seed": seed}
        torch.save(record, args.out / f"pairs_{name}.pt")


if __name__ == "__main__":
    main()
