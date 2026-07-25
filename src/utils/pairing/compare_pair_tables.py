from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from src.utils.pairing.table import atomic_json_dump, load_pair_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare pair tables from encoder seeds.")
    parser.add_argument("--tables", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing comparison artifact.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.tables) < 2:
        raise ValueError("At least two pair tables are required for comparison.")
    tables = [(Path(path).resolve(), load_pair_table(path)) for path in args.tables]
    identity = _comparison_identity(tables[0][1])
    for path, table in tables[1:]:
        if _comparison_identity(table) != identity:
            raise ValueError(
                "Pair tables are not comparable because their dataset, split, source "
                f"fingerprints, sizes, or data seed differ: {path}"
            )
    rows = []
    for (path_a, table_a), (path_b, table_b) in itertools.combinations(tables, 2):
        set_a = _pair_set(table_a)
        set_b = _pair_set(table_b)
        inter = set_a & set_b
        union = set_a | set_b
        rows.append(
            {
                "table_a": str(path_a),
                "table_b": str(path_b),
                "pairs_a": len(set_a),
                "pairs_b": len(set_b),
                "intersection": len(inter),
                "jaccard": len(inter) / max(len(union), 1),
                "overlap_min": len(inter) / max(min(len(set_a), len(set_b)), 1),
            }
        )

    summary = {
        "n_tables": len(tables),
        "comparisons": rows,
        "mean_jaccard": sum(r["jaccard"] for r in rows) / max(len(rows), 1),
        "mean_overlap_min": sum(r["overlap_min"] for r in rows) / max(len(rows), 1),
    }
    atomic_json_dump(summary, args.out, overwrite=args.overwrite)
    print(__import__("json").dumps(summary, indent=2, sort_keys=True, allow_nan=False))


def _pair_set(table: dict) -> set[tuple[int, int]]:
    idx1 = table["idx_1"].long().cpu().tolist()
    idx2 = table["idx_2"].long().cpu().tolist()
    return set(zip(idx1, idx2))


def _comparison_identity(table: dict) -> tuple:
    metadata = table["metadata"]
    return (
        table["dataset_1"],
        table["dataset_2"],
        table["split"],
        metadata["n_dataset_1"],
        metadata["n_dataset_2"],
        metadata["source_1_sha256"],
        metadata["source_2_sha256"],
        metadata.get("data_seed"),
    )


if __name__ == "__main__":
    main()
