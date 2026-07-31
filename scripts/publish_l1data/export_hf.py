# Builds the HuggingFace mirror from the finished Zenodo tree.
#
# Same events, same order, same values -- only the shape differs. Zenodo ships one
# directory per object collection, which is faithful to the source; HuggingFace wants
# one row per event, so the collections are joined side by side with prefixed columns
# and the raw ntuple names are mapped onto the short ones.
#
# Published after acceptance, not before: HuggingFace has no private-but-shareable mode,
# so an anonymous mirror would have to be fully public under a throwaway namespace.
import argparse
import json
import sys
from pathlib import Path

import awkward as ak
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "hf_assets"))
import adl1t_l1ad as l1
import anonymise

SHARD_BYTES = 500 * 1024**2
# Collections joined into the main table. `seeds` is a third of the volume on its own,
# so it gets its own config and travels separately.
MAIN_OBJECTS = ["ET", "FET", "FHT", "HT", "MET", "MHT", "cica", "egammas", "jets", "muons", "taus"]


def joined_table(split_dir: Path, dataset: str, label: int) -> pa.Table:
    """Zip one split's object collections into a single row-per-event table."""
    objects = l1.read_split(split_dir)
    renamed = l1.rename_fields(objects)

    columns = {}
    for name in MAIN_OBJECTS:
        if name not in renamed:
            continue
        for field in renamed[name].fields:
            columns[f"{name}_{field}"] = renamed[name][field]

    for field in objects.get("event_info", ak.Array([])).fields:
        columns[field] = objects["event_info"][field]

    if "seeds" in objects and "L1bit" in objects["seeds"].fields:
        columns["L1bit"] = objects["seeds"]["L1bit"]

    n = len(columns["run"]) if "run" in columns else len(next(iter(columns.values())))
    table = ak.to_arrow_table(ak.Array(columns), extensionarray=False)
    table = table.append_column("dataset", pa.array([dataset] * n))

    return table.append_column("label", pa.array([label] * n, type=pa.int16()))


def seeds_table(split_dir: Path, dataset: str) -> pa.Table | None:
    """The full trigger menu decision, kept apart from the kinematics."""
    seeds_dir = split_dir / "seeds"
    if not seeds_dir.is_dir():
        return None

    table = pq.read_table(sorted(seeds_dir.glob("*.parquet")))
    table = table.replace_schema_metadata(None)

    return table.append_column("dataset", pa.array([dataset] * table.num_rows))


def write_shards(table: pa.Table, out_dir: Path, split: str) -> int:
    """Write a table as HuggingFace-style `<split>-NNNNN-of-NNNNN.parquet` shards."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_per_shard = max(1, int(table.num_rows * SHARD_BYTES / max(table.nbytes, 1)))
    chunks = [
        table.slice(start, rows_per_shard) for start in range(0, table.num_rows, rows_per_shard)
    ]
    for index, chunk in enumerate(chunks):
        target = out_dir / f"{split}-{index:05d}-of-{len(chunks):05d}.parquet"
        pq.write_table(chunk, target, compression="snappy")

    return len(chunks)


def labels_for(summary: dict) -> dict:
    """Zero bias 0, simulated background negative, signals positive by sorted name."""
    labels, signal, background = {}, 0, 0
    for name in sorted(summary["datasets"]):
        category = summary["datasets"][name]["category"]
        if category == "zerobias":
            labels[name] = 0
        elif category == "background":
            background -= 1
            labels[name] = background
        else:
            signal += 1
            labels[name] = signal

    return labels


def configs_block(written: dict) -> str:
    """The `configs:` YAML the dataset card needs so `load_dataset` works scriptlessly."""
    lines = ["configs:"]
    for name in sorted(written):
        lines.append(f"- config_name: {name}")
        lines.append("  data_files:")
        for split, pattern in sorted(written[name].items()):
            hf_split = "validation" if split == "valid" else split
            lines.append(f"  - split: {hf_split}")
            lines.append(f"    path: {pattern}")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="publication root")
    parser.add_argument("--only", nargs="*", help="restrict to these dataset names")
    args = parser.parse_args()

    tree = args.out / "adl1t-l1ad-v1"
    hf_root = args.out / "huggingface"
    summary = json.loads((args.out / "metadata" / "split_summary.json").read_text())
    labels = labels_for(summary)
    written = {}

    for category in sorted(p for p in tree.iterdir() if p.is_dir()):
        for dataset in sorted(p for p in category.iterdir() if p.is_dir()):
            if args.only and dataset.name not in args.only:
                continue
            for split_dir in sorted(p for p in dataset.iterdir() if p.is_dir()):
                split = split_dir.name
                table = joined_table(split_dir, dataset.name, labels[dataset.name])
                out_dir = hf_root / "data" / dataset.name
                write_shards(table, out_dir, split)
                written.setdefault(dataset.name, {})[split] = (
                    f"data/{dataset.name}/{split}-*.parquet"
                )

                seeds = seeds_table(split_dir, dataset.name)
                if seeds is not None:
                    write_shards(seeds, hf_root / "data" / dataset.name / "seeds", split)
                    written.setdefault(f"{dataset.name}-seeds", {})[split] = (
                        f"data/{dataset.name}/seeds/{split}-*.parquet"
                    )
                print(f"  {dataset.name:46s} {split:5s} {table.num_rows:>9,} rows")

    (hf_root / "configs_block.yaml").write_text(configs_block(written))
    for asset in sorted((Path(__file__).resolve().parent / "hf_assets").glob("*.py")):
        (hf_root / asset.name).write_text(asset.read_text())
    (hf_root / "README.md").write_text((tree / "README.md").read_text())
    (hf_root / "LICENSE").write_text((tree / "LICENSE").read_text())
    anonymise.prune_stray(hf_root)

    return 0


if __name__ == "__main__":
    sys.exit(main())
