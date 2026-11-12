"""The per-run tables of the evaluation and the stored scores of the dumps.

``write_outputs`` writes a run's val.csv, test.csv, picks.csv and run.json, whose
columns ``src.evaluation.dump.evaluate_dump`` lists, and ``write_partial`` the table
of the runs left partial; the readers load them and the dumps for
``src.evaluation.lattice``.
"""

import csv
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.callbacks.components.utils import row_subset

log = logging.getLogger(__name__)

SCORE_KEY = "ascore/full"
SCORE_SPLITS = ("val", "test", "aux", "train")
# The per-run tables of the evaluation, val.csv and test.csv.
SPLITS = ("val", "test")
# The table of the runs with fewer epoch files than their max_epochs, in out_dir.
PARTIAL_RUNS = "partial_runs.csv"


def write_partial(out_dir: Path, rows: list[list]) -> None:
    """Write ``PARTIAL_RUNS`` (run, epochs, max_epochs) where a run is partial, and
    remove an earlier one otherwise."""
    path = Path(out_dir) / PARTIAL_RUNS
    if not rows:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as file:
        csv.writer(file).writerows([["run", "epochs", "max_epochs"], *rows])


def read_partial(out_dir: Path) -> set[str]:
    """The runs that ``PARTIAL_RUNS`` lists, none without the table."""
    path = Path(out_dir) / PARTIAL_RUNS
    return set(pd.read_csv(path, dtype=str)["run"]) if path.is_file() else set()


def write_outputs(out: Path, tables: dict[str, list[dict]], run: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    # Removed first and written last, so that a run.json on disk vouches for the tables
    # beside it however the writing was cut (the resume of src.evaluation.dump).
    (out / "run.json").unlink(missing_ok=True)
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


def _normal_at(path: Path, epoch: int, max_rows: int | None, row_seed: int):
    """The validation normal scores of an epoch file at its seeded row subset, None
    where the file lacks them."""
    record = load_record(path, epoch)
    if "normal" not in record.get("val", {}):
        return None
    scores = record["val"]["normal"][SCORE_KEY]
    subset = row_subset(len(scores), max_rows, row_seed)
    return (scores if subset is None else scores[subset]).numpy()


def normal_rows(
    run_dir: Path, max_rows: int | None, row_seed: int
) -> np.ndarray | None:
    """The validation normal scores at ``row_subset(n, max_rows, row_seed)`` of every
    epoch file of a run, (epochs + 1, rows), -1 first, or None where the run or a file
    lacks them; each file is cut as it is read, so that no full epoch stays held."""
    run_dir = Path(run_dir)
    if not (run_dir / "initial.pt").is_file():
        return None
    paths = [run_dir / "initial.pt", *sorted(run_dir.glob("epoch_*.pt"))]
    rows = []
    for epoch, path in enumerate(paths, start=-1):
        rows.append(_normal_at(path, epoch, max_rows, row_seed))
        if rows[-1] is None:
            return None
    return np.stack(rows)


def run_normal(run_dir: Path, rows: dict | None = None) -> np.ndarray | None:
    """``normal_rows`` of a run at ``rows`` (``max_rows``, ``row_seed``), every row for
    None."""
    rows = rows or {"max_rows": None, "row_seed": 0}
    return normal_rows(run_dir, rows["max_rows"], rows["row_seed"])


def load_lattice(
    lattice_dir: Path,
    names: list[str],
    scores_dir: Path | None = None,
    rows: dict | None = None,
) -> tuple[np.ndarray | None, list]:
    """Validation normal scores, (runs, epochs + 1, n), and every meta.pt.

    The scores are read from ``scores_dir``, ``lattice_dir`` when None, by
    ``run_normal``; they are None where a run lacks them, as the dumps of the physics
    lattices and of every lattice trained with ``stored_sets: []`` do.
    """
    scores_dir = lattice_dir if scores_dir is None else Path(scores_dir)
    metas = [torch.load(lattice_dir / n / "meta.pt", weights_only=True) for n in names]
    normal = [run_normal(scores_dir / name, rows) for name in names]
    if any(scores is None for scores in normal):
        log.warning(f"{scores_dir} lacks validation normal: UDR and MC are left out.")
        return None, metas
    return np.stack(normal), metas
