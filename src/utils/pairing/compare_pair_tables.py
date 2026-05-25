from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare pair tables from encoder seeds.")
    parser.add_argument("--tables", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = [(Path(path), torch.load(path, map_location="cpu", weights_only=False)) for path in args.tables]
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
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _pair_set(table: dict) -> set[tuple[int, int]]:
    idx1 = table["idx_1"].long().cpu().tolist()
    idx2 = table["idx_2"].long().cpu().tolist()
    return set(zip(idx1, idx2))


if __name__ == "__main__":
    main()
