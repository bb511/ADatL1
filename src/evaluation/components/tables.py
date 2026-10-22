"""The per-run tables of the evaluation and the stored scores of the dumps.

``write_outputs`` writes a run's val.csv, test.csv, picks.csv and run.json, whose
columns ``src.evaluation.dump.evaluate_dump`` lists; the readers load them and the
dumps for ``src.evaluation.lattice``.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCORE_KEY = "ascore/full"
SCORE_SPLITS = ("val", "test", "aux", "train")
# The per-run tables of the evaluation, val.csv and test.csv.
SPLITS = ("val", "test")


def write_outputs(out: Path, tables: dict[str, list[dict]], run: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        columns = list(dict.fromkeys(key for row in rows for key in row))
        with open(out / f"{name}.csv", "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    (out / "run.json").write_text(json.dumps(run, indent=2) + "\n")


def load_record(path: Path, epoch: int) -> dict:
    record = torch.load(path, weights_only=True)
    if record["epoch"] != epoch:
        raise RuntimeError(f"{path} holds epoch {record['epoch']}, not {epoch}.")
    return record


def _record_scores(path: Path) -> tuple[int, dict]:
    record = torch.load(path, map_location="cpu", weights_only=True)
    scores = {
        split: {
            name: {key: value.numpy() for key, value in keys.items()}
            for name, keys in record[split].items()
        }
        for split in SCORE_SPLITS
        if split in record
    }
    return record["epoch"], scores


def load_scores(run_dir: Path) -> dict[int, dict]:
    """Every stored score of a dump, ``{epoch: {split: {set: {key: (n,)}}}}``.

    ``initial.pt`` is epoch -1; the splits are those its records hold (``val``,
    ``test``, ``aux``, ``train``), each with every set and key it stores.
    """
    run_dir = Path(run_dir)
    paths = [run_dir / "initial.pt", *sorted(run_dir.glob("epoch_*.pt"))]
    scores = dict(_record_scores(path) for path in paths)
    if list(scores) != list(range(-1, len(paths) - 1)):
        raise ValueError(f"{run_dir} does not hold epochs -1 to {len(paths) - 2}.")
    return scores


def _epoch_table(path: Path) -> pd.DataFrame:
    # Round-trip parsing reads every value exactly as written, so that ties survive.
    table = pd.read_csv(path, float_precision="round_trip")
    return table.set_index("epoch").sort_index()


def _check_epochs(info: list[dict], tables: list[pd.DataFrame]) -> None:
    counts = sorted({run["epochs"] for run in info})
    if len(counts) != 1:
        raise ValueError(f"The runs hold different numbers of epochs: {counts}.")
    if any(list(table.index) != list(range(-1, counts[0])) for table in tables):
        raise ValueError(f"A per-run table does not hold epochs -1 to {counts[0] - 1}.")


def read_values(dirs: list[Path], info: list[dict]) -> pd.DataFrame:
    """Every run's val.csv and test.csv by (run, epoch), with the columns (split,
    column)."""
    splits = {s: [_epoch_table(path / f"{s}.csv") for path in dirs] for s in SPLITS}
    _check_epochs(info, splits["val"] + splits["test"])
    for split, tables in splits.items():
        if any(list(table.columns) != list(tables[0].columns) for table in tables):
            raise ValueError(f"The runs' {split}.csv hold different columns.")
    runs = [pd.concat(dict(zip(splits, run)), axis=1) for run in zip(*splits.values())]
    return pd.concat(runs, keys=[path.name for path in dirs], names=["run"])


def read_picks(dirs: list[Path]) -> pd.DataFrame:
    """Every run's picked epoch by (strategy, k), one column per run, k NaN for a pick
    by Q' alone."""
    tables = [pd.read_csv(path / "picks.csv").assign(run=path.name) for path in dirs]
    table = pd.concat(tables)
    picks = table.pivot(index=["strategy", "k"], columns="run", values="epoch")
    picks = picks.reindex(columns=[path.name for path in dirs])
    if picks.isna().to_numpy().any():
        raise ValueError("The runs' picks.csv name different strategies or k.")
    return picks.astype(np.int64)


def normal_scores(run_dir: Path) -> np.ndarray:
    scores = load_scores(run_dir)
    return np.stack([scores[epoch]["val"]["normal"][SCORE_KEY] for epoch in scores])


def load_lattice(lattice_dir: Path, names: list[str]) -> tuple[np.ndarray, list]:
    """Stored validation normal scores, (runs, epochs + 1, n), and every meta.pt."""
    normal = np.stack([normal_scores(lattice_dir / name) for name in names])
    metas = [torch.load(lattice_dir / n / "meta.pt", weights_only=True) for n in names]
    return normal, metas
