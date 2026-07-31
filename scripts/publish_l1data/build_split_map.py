# Maps the frozen train/valid/test indices back onto raw parquet row positions.
#
# The split is defined on *processed* rows, which are the raw rows surviving the
# ET-saturation event mask. This walks that back so the exporter can partition the raw
# files directly, and records what the dataset card has to state about the split.
import argparse
import hashlib
import json
import sys
from pathlib import Path

import awkward as ak
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import anonymise

SPLITS = ("train", "valid", "test")


def _compose(raw_data_dir: str):
    """Compose the physics `data=basis` config the paper trains on."""
    import hydra
    from hydra.core.global_hydra import GlobalHydra

    from src.utils.omegaconf import register_resolvers

    register_resolvers()
    GlobalHydra.instance().clear()
    configs = Path(__file__).resolve().parents[2] / "configs"
    with hydra.initialize_config_dir(version_base="1.3", config_dir=str(configs)):
        return hydra.compose(
            config_name="train",
            overrides=[
                "experiment=physics/ae",
                f"paths.raw_data_dir={raw_data_dir}",
                "run_name=export",
            ],
        )


def _keep_mask(processed_dir: Path) -> np.ndarray:
    """Per-raw-row boolean: did this event survive the saturation event filter."""
    mask = processed_dir / "event_masks" / "intersection.parquet"

    return ak.to_numpy(ak.from_parquet(mask)).astype(bool)


def _assign(mask: np.ndarray, per_split_local: dict) -> tuple:
    """Turn processed-row split indices into per-raw-row (split, order) arrays.

    Rows the event filter dropped were never seen by the pipeline. They still ship, and
    take the split of the nearest preceding kept event, with order -1 to mark them.
    """
    n_raw = mask.size
    keep = np.flatnonzero(mask)
    split_of = np.empty(n_raw, dtype="U5")
    order = np.full(n_raw, -1, dtype=np.int64)

    for name, local in per_split_local.items():
        raw_rows = keep[local]
        split_of[raw_rows] = name
        order[raw_rows] = np.arange(len(local), dtype=np.int64)

    # nearest preceding kept row; rows before the first kept one borrow from it
    last_kept = np.where(mask, np.arange(n_raw), -1)
    np.maximum.accumulate(last_kept, out=last_kept)
    source = np.where(last_kept >= 0, last_kept, keep[0])
    dropped = ~mask
    split_of[dropped] = split_of[source[dropped]]

    return split_of, order


def build(raw_data_dir: Path, out_dir: Path) -> dict:
    """Write one (split, order) map per dataset plus the split summary."""
    cfg = _compose(str(raw_data_dir))
    mlready = cfg.data.data_mlready
    processed_root = Path(mlready.processed_datapath)
    splits_dir = Path(mlready.cache_root_dir) / "mlready" / mlready.name / "splits"
    if not splits_dir.is_dir():
        raise SystemExit(f"No frozen splits at {splits_dir}. Run a training job first.")

    map_dir = out_dir / "_splitmap"
    map_dir.mkdir(parents=True, exist_ok=True)
    # Config key -> raw folder, which differ (ZB_run396102 lives in EphZB_2025E_run396102).
    # Kept beside the maps, outside the published tree, because it holds absolute paths.
    raw_dirs = {}
    summary = {
        "dataset_version": "adl1t-l1ad-v1",
        "split_seed": int(mlready.seed),
        "split_fractions": {"train": 0.6, "valid": 0.2, "test": "remainder"},
        "split_aux": float(mlready.split_aux),
        "event_filter": "ET.Et < 4095 (rows failing it carry order = -1)",
        "dropped_row_rule": "assigned the split of the nearest preceding kept event",
        "datasets": {},
    }

    for category, names in [
        ("zerobias", dict(cfg.data.zerobias)),
        ("background", dict(cfg.data.background)),
        ("signal", dict(cfg.data.signal)),
    ]:
        frozen = _load_frozen(splits_dir, category, names)
        offset = 0
        # The pipeline concatenates zerobias in sorted path order, which is NOT the order
        # the config lists them in -- data_2025E+G.yaml puts ZB_run398183 first while the
        # split indices are global over ZB_run396102 then ZB_run398183.
        for name in sorted(names):
            raw_dir = Path(names[name])
            processed_dir = processed_root / category / name
            mask = _keep_mask(processed_dir)
            n_proc = int(mask.sum())
            local = {
                key: idx[(idx >= offset) & (idx < offset + n_proc)] - offset
                for key, idx in frozen[name if category != "zerobias" else "shared"].items()
            }
            split_of, order = _assign(mask, local)
            np.savez(map_dir / f"{category}__{name}.npz", split=split_of, order=order)
            raw_dirs[name] = {"category": category, "raw_dir": str(raw_dir)}
            summary["datasets"][name] = {
                "category": category,
                "raw_events": int(mask.size),
                "events_passing_filter": n_proc,
                "counts": {k: int((split_of == k).sum()) for k in set(split_of)},
                "objects": sorted(
                    p.name for p in raw_dir.iterdir() if p.is_dir() and p.name != "PLOTS"
                ),
            }
            offset += n_proc if category == "zerobias" else 0

    scrubbed = anonymise.scrub_resolved_config(
        json.loads(json.dumps(_to_container(cfg.data))), str(raw_data_dir)
    )
    blob = json.dumps(scrubbed, sort_keys=True).encode()
    summary["config_sha256"] = hashlib.sha256(blob).hexdigest()

    (map_dir / "index.json").write_text(json.dumps(raw_dirs, indent=2) + "\n")

    (out_dir / "metadata").mkdir(parents=True, exist_ok=True)
    _export_norm_params(Path(mlready.cache_root_dir) / "mlready" / mlready.name, out_dir)
    _export_scales(cfg, out_dir)
    (out_dir / "metadata" / "split_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (out_dir / "metadata" / "data_config_resolved.yaml").write_text(
        _to_yaml(scrubbed)
    )

    return summary


def _export_norm_params(mlready_dir: Path, out_dir: Path) -> None:
    """Convert the pickled shift/scale into JSON the deposition can ship.

    One file per normalizer variant that was fitted, so `robust` (the paper's) and
    `standard` (which the dte model uses) both travel with the data.
    """
    import pickle

    for variant in sorted(p for p in mlready_dir.iterdir() if p.is_dir()):
        params = {}
        for pkl in sorted(variant.glob("*_norm_params.pkl")):
            obj = pkl.name.removesuffix("_norm_params.pkl")
            params[obj] = {
                feat: {k: float(v) for k, v in values.items()}
                for feat, values in pickle.loads(pkl.read_bytes()).items()
            }
        if not params:
            continue
        target = out_dir / "metadata" / f"norm_params_{variant.name}.json"
        target.write_text(json.dumps(params, indent=2, sort_keys=True) + "\n")

        feature_map = variant / "object_feature_map.json"
        if feature_map.is_file():
            (out_dir / "metadata" / "object_feature_map.json").write_text(
                feature_map.read_text()
            )


def _export_scales(cfg, out_dir: Path) -> None:
    """Ship the hardware-to-physical scale factors as reference values."""
    header = (
        "# L1 trigger scales: multiply the integer hardware values by these to get\n"
        "# GeV, radians and pseudorapidity. The published data is NOT scaled -- these\n"
        "# are provided so you can convert, and are what a pure-rate calculation needs.\n"
    )
    (out_dir / "metadata" / "l1_scales.yaml").write_text(
        header + _to_yaml(_to_container(cfg.data.l1_scales))
    )


def _load_frozen(splits_dir: Path, category: str, names: dict) -> dict:
    """Read the frozen .npz files for a category into {name: {split: indices}}."""
    if category == "zerobias":
        with np.load(splits_dir / "zerobias.npz") as z:
            return {"shared": {k: z[f"i{k}"] for k in SPLITS}}

    out = {}
    for name in names:
        with np.load(splits_dir / f"aux__{name}.npz") as z:
            out[name] = {k: z[f"i{k}"] for k in ("valid", "test")}

    return out


def _to_container(node):
    from omegaconf import OmegaConf

    return OmegaConf.to_container(node, resolve=True)


def _to_yaml(obj) -> str:
    from omegaconf import OmegaConf

    return OmegaConf.to_yaml(OmegaConf.create(obj))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    summary = build(args.raw_data_dir, args.out)
    print(f"mapped {len(summary['datasets'])} datasets into {args.out}/_splitmap")
    for name, info in summary["datasets"].items():
        print(f"  {name:46s} raw={info['raw_events']:>9d} {info['counts']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
