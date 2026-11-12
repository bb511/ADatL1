#!/usr/bin/env python3
"""Build ANCAP v8's pushed points at their frozen scales, one subcommand per domain.

``lhc`` pushes the LHC's held-out rows H by the four kinds of
``src/data/components/push_lhc.py``, each at its frozen scale. From the tree
``--robust``, the ml-ready ``robust`` directory, it reads the files that the frozen page
``scratchpad/adl1t/theory/theory_xxxi/step4/push/LHC.md`` names, and no other:
``{train,test}/torch_cache.pt``, ``{train,test}/torch_mask.pt`` and the
``<name>_norm_params.pkl`` of FET and the four collections.

Into ``--out`` it writes ``lhc_kind<k>.npz`` for each kind k, holding ``x`` float32
(10000, 39, 3), ``mask`` bool (10000, 39, 3) and ``rows`` int64 (10000,), the test-split
indices of H. The md5 of an npz file changes from run to run, since ``numpy.savez``
stamps the time into the archive, so the log ``lhc_log.json`` records the md5 of each
array's bytes. It also counts each kind's rows outside the possible events and records
the sizes of the splits, the md5 of T₁'s indices and the constants' sizes. As the page's
check of the encoding, H is also written back unpushed, and the log counts the inputs
that differ from the stored float32 and the rows outside the possible events. Every
count is expected to be 0.

Each of the six plain-vector domains (``cchamber``, ``cchamber_wt``, ``cifar10``,
``mscct``, ``dijet``, ``sbc``) pushes its held-out typical rows by
``src/data/components/push_plain.py`` at its frozen scale. Its datamodule is composed
from ``experiment=<domain>/ae`` with ``paths.base_data_dir`` set to ``--data-root``,
seeded as ``src/train.py`` seeds it, and set up for fit and for test. The signal sets
are emptied, so that no anomaly row is read, and the dijet's nulls, its simulated
background, are dropped; MS-CCT is built as configured, and only its typical tensors are
read. CIFAR-10 and MS-CCT write their caches beside the data when these are missing.

Into ``--out`` it writes ``<domain>_pushed.npz``, holding ``x`` float32 (n, d), or (n,
3, 32, 32) on the images, ``rows`` int64 (n,), the index in X_H of the first row equal
to each point's start row, and ``scale``. The log ``<domain>_log.json`` records the
sizes of X_train, R, X_H and RH, the possible events (flags counted, bounds and flags
by md5), the rows of X_train and of RH outside them, the entries of RH that the
projection changes, the pushed points outside them (expected 0), the md5 of R, RH,
``x`` and ``rows``, and whether the signal sets were emptied.

Usage::

    python scripts/analysis/build_pushed_points.py lhc --robust <tree> --out <dir>
    python scripts/analysis/build_pushed_points.py dijet --data-root <dir> --out <dir>
"""
import argparse
import hashlib
import json
import pickle  # nosec B403: only the record's own normalisation parameters are read
from pathlib import Path

import numpy as np
import rootutils
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, seed_everything

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components import push_lhc, push_plain
from src.utils.omegaconf import register_resolvers

register_resolvers()


def _md5(a: np.ndarray) -> str:
    """Return the md5 of an array's bytes in C order."""
    return hashlib.md5(a.tobytes(), usedforsecurity=False).hexdigest()


def _norm(robust: Path) -> push_lhc.Norm:
    """Return the scale and the shift of every (slot, feature), read from the tree."""
    params = {}
    for name in ("FET", *push_lhc.COLLECTIONS):
        with open(robust / f"{name}_norm_params.pkl", "rb") as f:
            params[name] = pickle.load(f)  # nosec B301
    return push_lhc.norm_arrays(params)


def _load_rows(split: Path, seed: int, size: int) -> tuple:
    """Return the draw's float32 rows and mask, its indices and the split's size.

    The tensors are memory-mapped, so that neither is held in memory whole.
    """
    x = torch.load(split / "torch_cache.pt", mmap=True, weights_only=True)
    mask = torch.load(split / "torch_mask.pt", mmap=True, weights_only=True)
    layout = (tuple(x.shape[1:]), x.dtype, tuple(mask.shape), mask.dtype)
    if layout != ((push_lhc.N_SLOTS, 3), torch.float32, tuple(x.shape), torch.bool):
        raise ValueError(f"{split}: rows and mask of layout {layout}")
    rows = push_lhc.draw_rows(x.shape[0], size, seed)
    index = torch.from_numpy(rows)
    return x[index].numpy(), mask[index].numpy(), rows, x.shape[0]


def _t1_constants(robust: Path, norm: push_lhc.Norm) -> tuple[push_lhc.Constants, dict]:
    """Return the constants over T₁ and their entries of the log."""
    x, mask, t1, n_train = _load_rows(robust / "train", push_lhc.T1_SEED, push_lhc.N_T1)
    consts = push_lhc.constants(push_lhc.decode(x, mask, norm), mask)
    log = {"n_train": n_train, "t1_md5": _md5(t1)}
    log["low"] = {c: v["low"] for c, v in consts.items()}
    log["n_eta_values"] = {c: len(v["v_eta"]) for c, v in consts.items()}
    log["n_phi_values"] = {c: len(v["v_phi"]) for c, v in consts.items()}
    return consts, log


def _h_rows(robust: Path, norm: push_lhc.Norm, consts: push_lhc.Constants) -> tuple:
    """Return H's hardware integers, mask and indices, and the check of H unpushed."""
    x, mask, rows, n_test = _load_rows(robust / "test", push_lhc.H_SEED, push_lhc.N_H)
    hw = push_lhc.decode(x, mask, norm)
    log = {"n_test": n_test}
    log["unpushed_float_mismatch"] = int((push_lhc.encode(hw, mask, norm) != x).sum())
    log["unpushed_impossible"] = push_lhc.count_impossible(x, mask, norm, consts)
    return (hw, mask, rows), log


def _push_kind(
    kind: int, h: tuple, consts: push_lhc.Constants, norm: push_lhc.Norm, out: Path
) -> dict:
    """Write one kind's rows of H at its frozen scale and return its log entry."""
    hw, mask, rows = h
    scale = push_lhc.SCALES[kind]
    hw_p, mask_p = push_lhc.push(kind, hw, mask, scale, consts)
    x = push_lhc.encode(hw_p, mask_p, norm)
    np.savez(out / f"lhc_kind{kind}.npz", x=x, mask=mask_p, rows=rows)
    arrays = {"x": x, "mask": mask_p, "rows": rows}
    md5s = {f"md5_{name}": _md5(a) for name, a in arrays.items()}
    impossible = push_lhc.count_impossible(x, mask_p, norm, consts)
    return {"scale": scale, **md5s, "impossible": impossible}


def build_lhc(args: argparse.Namespace) -> None:
    """Write the four kinds' rows of H and their log into ``args.out``."""
    norm = _norm(args.robust)
    consts, log = _t1_constants(args.robust, norm)
    h, h_log = _h_rows(args.robust, norm, consts)
    args.out.mkdir(parents=True, exist_ok=True)
    kinds = {str(k): _push_kind(k, h, consts, norm, args.out) for k in push_lhc.SCALES}
    text = json.dumps(log | h_log | {"kinds": kinds}, indent=1)
    (args.out / "lhc_log.json").write_text(text + "\n")
    print(text)


def _flat(x: torch.Tensor) -> np.ndarray:
    return x.reshape(len(x), -1).numpy()


def _compose(domain: str, data_root: Path) -> DictConfig:
    overrides = [f"experiment={domain}/ae", f"paths.base_data_dir={data_root}"]
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        return compose(config_name="train.yaml", overrides=overrides)


def _empty_signals(cfg: DictConfig, domain: str) -> bool:
    """Empty the signal sets of ``cfg.data`` and return whether they were emptied.

    MS-CCT keeps its classes: its patch caches are keyed on the class list, so an empty
    list would force a cold extraction of every scan. The dijet's nulls, its simulated
    background, are dropped too.
    """
    if domain == "mscct":
        return False
    if domain == "cifar10":
        cfg.data.signal_classes = []
    else:
        cfg.data.signals = {}
    if domain == "dijet":
        cfg.data.nulls = None
    return True


def _datamodule(
    domain: str, data_root: Path
) -> tuple[LightningDataModule, DictConfig, bool]:
    """Return the domain's datamodule set up for fit and for test, its config, and
    whether its signal sets were emptied."""
    cfg = _compose(domain, data_root)
    signals_empty = _empty_signals(cfg, domain)
    # As src/train.py, which seeds before it instantiates the datamodule.
    seed_everything(cfg.seed, workers=True)
    dm = instantiate(cfg.data)
    dm.prepare_data()
    dm.setup("fit")
    dm.setup("test")
    return dm, cfg, signals_empty


def _test_set(dm: LightningDataModule, domain: str, name: str) -> torch.Tensor:
    """Return the inputs of one test set, read from the datamodule's tensors."""
    if domain == "mscct":
        return dm._sets["test"][name].x
    return dm._main["test"].x if name == "normal" else dm._aux["test"][name].x


def _held_out(dm: LightningDataModule, domain: str) -> torch.Tensor:
    return torch.cat([_test_set(dm, domain, n) for n in push_plain.HELD_OUT[domain]])


def _events(
    cfg: DictConfig, dm: LightningDataModule, domain: str, x_train: np.ndarray
) -> push_plain.Possible:
    """Return the domain's possible events, read from X_train."""
    if domain in push_plain.IMAGE_DOMAINS:
        return push_plain.image_events(x_train, dm.mean.numpy(), dm.std.numpy())
    names = list(cfg.data.get("readouts") or cfg.data.inputs)
    logged = list(cfg.data.get("log_inputs") or [])
    scaling = dm.median.numpy(), dm.spread.numpy()
    return push_plain.tabular_events(x_train, *scaling, names, logged)


def _rows_log(
    x_train: torch.Tensor, x_h: torch.Tensor, r: torch.Tensor, rh: torch.Tensor
) -> dict:
    log = {"n_train": len(x_train), "d": r.shape[1], "m": len(r), "n_h": len(x_h)}
    md5s = {"md5_r": _md5(r.double().numpy()), "md5_rh": _md5(rh.double().numpy())}
    return log | {"m_h": len(rh)} | md5s


def _events_log(
    events: push_plain.Possible, x_train: np.ndarray, rh: np.ndarray
) -> dict:
    """Return the log's entries on the possible events; the rows are float32 (n, d)."""
    changed = push_plain.project(rh.astype(np.float64), events).astype(np.float32) != rh
    return {
        "n_never_negative": int(events.never_negative.sum()),
        "n_integer": int(events.integer.sum()),
        "md5_lo": _md5(events.lo),
        "md5_hi": _md5(events.hi),
        "md5_integer": _md5(events.integer),
        "train_violations": push_plain.n_violations(x_train, events),
        "h_violations": push_plain.n_violations(rh, events),
        "h_changed": int(changed.sum()),
    }


def _push_log(
    domain: str,
    x_h: torch.Tensor,
    rh: torch.Tensor,
    events: push_plain.Possible,
    out: Path,
) -> dict:
    """Write the points pushed from RH into ``out`` and return their log entries."""
    scale = push_plain.SCALES[domain]
    x, start_rows = push_plain.push(rh.double().numpy(), scale, events)
    rows = push_plain.first_rows(_flat(x_h), rh.numpy())[start_rows]
    log = {"scale": scale, "n_points": len(x), "md5_x": _md5(x), "md5_rows": _md5(rows)}
    log["violations_projected"] = push_plain.n_violations(x, events)
    if domain in push_plain.IMAGE_DOMAINS:
        x = x.reshape(len(x), *push_plain.IMAGE_SHAPE)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{domain}_pushed.npz", x=x, rows=rows, scale=scale)
    return log


def build_plain(args: argparse.Namespace) -> None:
    """Write one plain-vector domain's pushed points and their log into ``args.out``."""
    dm, cfg, signals_empty = _datamodule(args.domain, args.data_root)
    x_train, x_h = dm.data_train.tensors[0], _held_out(dm, args.domain)
    events = _events(cfg, dm, args.domain, x_train.numpy())
    r, rh = push_plain.distinct_rows(x_train), push_plain.distinct_rows(x_h)
    log = _rows_log(x_train, x_h, r, rh)
    log |= _events_log(events, _flat(x_train), rh.numpy())
    log |= _push_log(args.domain, x_h, rh, events, args.out)
    text = json.dumps(log | {"signals_empty": signals_empty}, indent=1)
    (args.out / f"{args.domain}_log.json").write_text(text + "\n")
    print(text)


def _add_plain(domains: argparse._SubParsersAction, domain: str) -> None:
    plain = domains.add_parser(domain, help=f"The pushed points of {domain}.")
    plain.add_argument(
        "--data-root", type=Path, required=True, help="paths.base_data_dir."
    )
    plain.add_argument("--out", type=Path, required=True, help="The output directory.")
    plain.set_defaults(build=build_plain)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Return the arguments; each subcommand sets ``build``, the function it runs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    domains = parser.add_subparsers(dest="domain", required=True)
    lhc = domains.add_parser("lhc", help="The LHC's four kinds of move.")
    lhc.add_argument("--robust", type=Path, required=True, help="The ml-ready tree.")
    lhc.add_argument("--out", type=Path, required=True, help="The output directory.")
    lhc.set_defaults(build=build_lhc)
    for domain in push_plain.SCALES:
        _add_plain(domains, domain)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    args.build(args)


if __name__ == "__main__":
    main()
