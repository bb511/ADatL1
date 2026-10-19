#!/usr/bin/env python3
"""Score the untrained-model reference row of the chamber tables.

The reference is the whitened residual score of a model that reconstructs nothing: the
Mahalanobis energy of the normalised inputs, (x - mu)^T P (x - mu) / 11, with mu and P
the mean and the OAS-shrunk precision of the training normals
(``secondary.reference_row`` of ``scripts/cchamber/registration_limits.yaml``). Two
commands:

    dump   write it as a score dump in the layout of ``src/callbacks/score_dump.py``,
           ``meta.pt`` and a single ``epoch_0000.pt`` holding every validation and test
           set of the data module under ``ascore/full`` and ``ascore/residual_oas``
           (no auxiliary sets), so that ``offline_select.py`` scores it like a run;
           name the directory like one, e.g. ``reference_000_s0/scores``
    check  for every pool run of a ``selection.csv`` under one CAP configuration,
           whether its declared score on validation ``normal`` at the selected epoch
           equals the reference to a relative tolerance of 1e-5; writes one row per
           run and prints the counts per family

Usage
-----
    python scripts/cchamber/reference_row.py dump \\
        --data-dir /data/deodagiu/claude/adl1t/sub1/data/causal_chamber \\
        --out design/reference_000_s0/scores
    python scripts/cchamber/reference_row.py check \\
        --reference design/reference_000_s0/scores \\
        --selection design_out/selection.csv --config <frozen configuration> \\
        --out design/reference_check.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sklearn.covariance import OAS

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# The model's own quadratic form and the dump's own collector, so that the reference
# is computed and stored exactly as a run's residual score is.
from src.algorithms.ae import _whitened_energy  # noqa: E402
from src.callbacks.score_dump import _Collector  # noqa: E402
from src.data.utils import unpack_batch  # noqa: E402

DECLARED_KEY = "ascore/full"
KEYS = [DECLARED_KEY, "ascore/residual_oas"]
# algorithm.target_rate of the cchamber experiments, whose base_rate is null.
TARGET_RATE = 0.01
META = {
    "target_rate": TARGET_RATE,
    "base_rate": None,
    "target_fpr": TARGET_RATE,
    "module": "scripts.cchamber.reference_row",
}
RTOL = 1e-5
DATA_CONFIG = ROOT / "configs" / "data" / "cchamber.yaml"
REFERENCE = "reference"


def _datamodule(data_config: Path, data_dir: str):
    cfg = OmegaConf.load(data_config)
    cfg.pop("defaults", None)
    cfg.data_dir = data_dir
    # It orders only the training batches, which the reference never draws.
    cfg.train_seed = None
    datamodule = instantiate(cfg)
    datamodule.setup("fit")
    return datamodule


def _state(x_train: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and OAS precision of the training inputs, fitted in float64 as
    ResidualStateCallback fits them, held in float32 as the AE buffers hold them."""
    oas = OAS(store_precision=True, assume_centered=False).fit(x_train.double().numpy())
    location, precision = oas.location_, oas.precision_
    return torch.from_numpy(location).float(), torch.from_numpy(precision).float()


def _score(loaders: dict, location: torch.Tensor, precision: torch.Tensor):
    collector = _Collector(KEYS)
    for name, loader in loaders.items():
        for batch in loader:
            x = unpack_batch(batch).x
            score = _whitened_energy(x, location, precision, None)
            collector.add(name, dict.fromkeys(KEYS, score), batch)
    return collector


def _write_dump(out: Path, splits: dict, state: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    scores = {split: c.scores() for split, c in splits.items()}
    torch.save({"epoch": 0, **scores, "state_dict": state}, out / "epoch_0000.pt")
    layouts = {split: c.layout() for split, c in splits.items()}
    torch.save({**META, **layouts}, out / "meta.pt")


def dump(data_config: Path, data_dir: str, out: Path) -> None:
    if any(out.glob("epoch_*.pt")):
        raise SystemExit(f"{out} already holds a score dump.")
    datamodule = _datamodule(data_config, data_dir)
    location, precision = _state(datamodule.data_train.tensors[0])
    loaders = {"val": datamodule.val_dataloader(), "test": datamodule.test_dataloader()}
    splits = {s: _score(loader, location, precision) for s, loader in loaders.items()}
    _write_dump(out, splits, {"location": location, "precision": precision})


def _compare(unit: dict, expected: torch.Tensor) -> dict:
    """Whether a run's declared score on validation normal at its selected epoch equals
    the reference."""
    epoch = int(unit["selected_epoch"])
    record = torch.load(Path(unit["run"]) / f"epoch_{epoch:04d}.pt", weights_only=True)
    scores = record["val"]["normal"][DECLARED_KEY]
    relative = ((scores - expected).abs() / expected.abs()).max().item()
    equal = torch.allclose(scores, expected, rtol=RTOL, atol=0.0)
    head = {key: unit[key] for key in ("family", "candidate", "seed", "selected_epoch")}
    return {**head, "max_relative_difference": relative, "equal": equal}


def _pool_units(selection: Path, config: str) -> list[dict]:
    """Pool runs scored with the declared score under one CAP configuration."""
    with open(selection, newline="") as file:
        units = [u for u in csv.DictReader(file) if u["config"] == config]
    units = [u for u in units if u["strategy"] == "cap" and u["family"] != REFERENCE]
    return [u for u in units if u["score_key"] == DECLARED_KEY]


def check(reference: Path, selection: Path, config: str, out: Path) -> None:
    record = torch.load(reference / "epoch_0000.pt", weights_only=True)
    expected = record["val"]["normal"][DECLARED_KEY]
    rows = [_compare(unit, expected) for unit in _pool_units(selection, config)]
    with open(out, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for family in sorted({row["family"] for row in rows}):
        equal = [row["equal"] for row in rows if row["family"] == family]
        print(f"{family}: {sum(equal)} of {len(equal)} runs equal the reference")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    dump_args = commands.add_parser("dump")
    dump_args.add_argument("--data-config", type=Path, default=DATA_CONFIG)
    dump_args.add_argument("--data-dir", required=True)
    dump_args.add_argument("--out", type=Path, required=True)
    check_args = commands.add_parser("check")
    check_args.add_argument("--reference", type=Path, required=True)
    check_args.add_argument("--selection", type=Path, required=True)
    check_args.add_argument("--config", required=True)
    check_args.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.command == "dump":
        dump(args.data_config, args.data_dir, args.out)
    else:
        check(args.reference, args.selection, args.config, args.out)


if __name__ == "__main__":
    main()
