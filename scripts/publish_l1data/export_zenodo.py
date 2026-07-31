# Writes the raw parquet out physically partitioned by train/valid/test.
#
# Two passes. Pass A consolidates each object's thousands of raw shards into one file, in
# the lexicographic order the pipeline reads them, so pass B can random-access it cheaply.
# Pass B takes the rows of each split in the pipeline's permutation order and writes them
# back out as ~500 MB shards.
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import anonymise

SHARD_BYTES = 500 * 1024**2
TAKE_CHUNK = 1_000_000


def raw_objects(dataset_dir: Path) -> list[str]:
    """Object collections present for this dataset, ignoring the plot directory."""
    return sorted(
        p.name for p in dataset_dir.iterdir() if p.is_dir() and p.name != "PLOTS"
    )


def consolidate(dataset_dir: Path, obj: str, out_path: Path) -> int:
    """Stream one object's shards into a single file, preserving pipeline row order."""
    if out_path.is_file():
        return pq.ParquetFile(out_path).metadata.num_rows

    shards = sorted((dataset_dir / obj).glob("*.parquet"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    rows = 0
    try:
        for shard in shards:
            table = pq.read_table(shard)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
            writer.write_table(table)
            rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()

    return rows


def split_row_order(split_of: np.ndarray, order: np.ndarray, name: str) -> np.ndarray:
    """Raw row numbers for one split, in the order the deposition writes them.

    Rows the pipeline saw come first, in its permutation order, so reading a split
    front-to-back reproduces the training tensor. The saturated rows it never saw are
    appended afterwards in raw order, and carry order = -1.
    """
    in_split = np.flatnonzero(split_of == name)
    seen = in_split[order[in_split] >= 0]
    seen = seen[np.argsort(order[seen], kind="stable")]
    unseen = in_split[order[in_split] < 0]

    return np.concatenate([seen, unseen])


def write_split(
    source: Path, rows: np.ndarray, out_dir: Path, extra: dict | None = None
) -> int:
    """Take rows from a consolidated object file and write them as sharded parquet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = pds.dataset(source, format="parquet")
    writer, shard_idx, in_shard, written = None, 0, 0, 0

    for start in range(0, len(rows), TAKE_CHUNK):
        chunk = rows[start : start + TAKE_CHUNK]
        table = dataset.take(pa.array(chunk))
        if extra:
            for column, values in extra.items():
                table = table.append_column(column, pa.array(values[start : start + len(chunk)]))
            # The inherited awkward metadata describes the original columns only, and
            # awkward refuses to read a file whose metadata omits a column. These tables
            # are flat, so dropping it loses nothing.
            table = table.replace_schema_metadata(None)
        if writer is None:
            writer = pq.ParquetWriter(
                out_dir / f"{shard_idx:05d}.parquet", table.schema, compression="snappy"
            )
        writer.write_table(table)
        in_shard += table.nbytes
        written += table.num_rows
        if in_shard >= SHARD_BYTES:
            writer.close()
            shard_idx, in_shard, writer = shard_idx + 1, 0, None

    if writer is not None:
        writer.close()

    return written


def export_dataset(
    name: str, category: str, raw_dir: Path, split_map: Path, work: Path, out: Path
) -> dict:
    """Partition every object of one dataset into its split directories."""
    with np.load(split_map, allow_pickle=False) as data:
        split_of, order = data["split"], data["order"]

    objects = {}
    for obj in raw_objects(raw_dir):
        consolidated = work / name / f"{obj}.parquet"
        n_raw = consolidate(raw_dir, obj, consolidated)
        if n_raw != split_of.size:
            raise ValueError(
                f"{name}/{obj} has {n_raw} rows but the split map covers {split_of.size}"
            )
        objects[obj] = consolidated

    counts = {}
    for split in sorted(set(split_of.tolist())):
        rows = split_row_order(split_of, order, split)
        counts[split] = len(rows)
        for obj, consolidated in objects.items():
            # event_info carries the split assignment so a stray file is self-describing
            extra = None
            if obj == "event_info":
                extra = {"split": [split] * len(rows), "order": order[rows].tolist()}
            write_split(consolidated, rows, out / category / name / split / obj, extra)

    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="publication root")
    parser.add_argument("--work", type=Path, help="consolidation scratch (default <out>/_work)")
    parser.add_argument("--only", nargs="*", help="restrict to these dataset names")
    parser.add_argument("--tar", action="store_true", help="also pack the finished tree")
    args = parser.parse_args()

    work = args.work or args.out / "_work"
    maps = sorted((args.out / "_splitmap").glob("*.npz"))
    if not maps:
        raise SystemExit(f"No split maps under {args.out}/_splitmap. Run build_split_map.py.")

    index = json.loads((args.out / "_splitmap" / "index.json").read_text())
    tree = args.out / "adl1t-l1ad-v1"
    for split_map in maps:
        category, name = split_map.stem.split("__", 1)
        if args.only and name not in args.only:
            continue
        raw_dir = Path(index[name]["raw_dir"])
        counts = export_dataset(name, category, raw_dir, split_map, work, tree)
        print(f"  {name:46s} {counts}")

    anonymise.prune_stray(tree)
    if args.tar:
        _pack(tree, args.out / "tarballs")

    return 0


def _pack(tree: Path, tar_dir: Path) -> None:
    """One anonymised tarball per top-level dataset directory."""
    tar_dir.mkdir(parents=True, exist_ok=True)
    for category in sorted(p for p in tree.iterdir() if p.is_dir()):
        for dataset in sorted(p for p in category.iterdir() if p.is_dir()):
            member = str(dataset.relative_to(tree))
            out_path = tar_dir / f"{member.replace('/', '__')}.tar"
            subprocess.run(anonymise.tar_cmd(tree, member, out_path), check=True)
            print(f"  packed {out_path.name}")


if __name__ == "__main__":
    sys.exit(main())
