"""The tables of the lattice step in the old schema, that of
scripts/analysis/harvest_pool_effs.py, which the notebooks read; the module can be
removed once they read the tables of ``src.evaluation.report``."""

import functools
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.callbacks.components.selection import W_STEPS
from src.evaluation.components.tables import SPLITS
from src.evaluation.lattice import Lattice, Strategy

# The harvest's labels where they differ from the strategy names.
LABELS = {"drift": "stability", "w1": "wasserstein"}
# The column lists of scripts/analysis/harvest_pool_effs.py, which the notebooks read.
# They are copied rather than imported, since src/ does not depend on scripts/;
# tests/test_lattice.py keeps the two equal.
OLD_COLUMNS = (
    "run_name strategy ckpt epoch optimized_main optimized_sec mean_sig_eff n_signals"
).split()
NEW_COLUMNS = (
    "experiment run run_id tier monitor maximize monitor_value q_source is_pareto "
    "test_mean_sig_eff test_n_signals"
).split()
EVAL_COLUMNS = "run_name strategy split context experiment run run_id tier".split()


def _tier_name(rate: float, base_rate: float | None) -> str:
    # The operational tier as harvest_pool_effs._tier_name names it: by its rate in Hz
    # (rates are in kHz) with a base rate, else by the quantile its rate sets.
    if base_rate is not None:
        return f"{1e3 * rate:g}"
    return f"q{100 * (1 - rate):g}"


def _harvest_name(name: str, k: int | None) -> str:
    label = LABELS.get(name, name)
    return label if k is None else f"{label}_w{k / W_STEPS:g}"


def _contexts(columns) -> dict[str, list[str]]:
    """The recorded <cb_name>/<scalar> columns of a per-run table, by callback."""
    contexts = {}
    for column in columns:
        callback, _, scalar = column.partition("/")
        if scalar and callback != "ema":
            contexts.setdefault(callback, []).append(column)
    return contexts


@dataclass(frozen=True, eq=False)
class Harvest:
    """<lattice>.csv and <lattice>_eval.csv, in the schema of
    scripts/analysis/harvest_pool_effs.py."""

    lattice: Lattice
    name: str  # the lattice's name, the experiment column of every row

    @functools.cached_property
    def fixed(self) -> dict:
        meta = self.lattice.metas[0]
        tier = _tier_name(meta["target_rate"], meta["base_rate"])
        fixed = {"experiment": self.name, "tier": tier, "ckpt": "epoch"}
        return {**fixed, "q_source": "eval", "is_pareto": None}

    def head(self, i: int) -> dict:
        """The columns every row of run i shares."""
        info, run = self.lattice.info[i], self.lattice.names[i]
        head = {**self.fixed, "run": run, "run_id": info["mlflow_run_id"]}
        return {**head, **{f"hp/{key}": v for key, v in info["hyperpoint"].items()}}

    def row(self, strategy: Strategy, k: int | None, i: int) -> dict:
        lattice, name = self.lattice, _harvest_name(strategy.name, k)
        epoch, run = int(strategy.picks[k][i]), lattice.names[i]
        at, effs = (i, epoch + 1), lattice.set_effs["val"][i, epoch + 1]
        row = {"run_name": f"{name}_{run}", "strategy": name, "epoch": epoch}
        row.update(optimized_main=strategy.q_main[at], optimized_sec=strategy.q_sec[at])
        row.update(mean_sig_eff=effs.mean(), monitor=strategy.key)
        row.update(maximize=strategy.mode == "max", monitor_value=strategy.ema[at])
        row.update(n_signals=len(effs), test_mean_sig_eff=lattice.test_eff[at])
        return {**row, "test_n_signals": len(effs), **dict(zip(lattice.signals, effs))}

    def run_rows(self, i: int) -> list[dict]:
        """The rows of run i: every strategy at k = 10 and k = 0, or by Q' alone."""
        head, rows = self.head(i), []
        for strategy in self.lattice.strategies:
            ks = [None] if None in strategy.picks else [W_STEPS, 0]
            rows += [{**self.row(strategy, k, i), **head} for k in ks]
        return rows

    def table(self) -> pd.DataFrame:
        rows = [row for i in range(len(self.lattice.names)) for row in self.run_rows(i)]
        table = pd.DataFrame(rows).sort_values("run_name", kind="stable")
        hp = sorted(column for column in table.columns if column.startswith("hp/"))
        return table[OLD_COLUMNS + NEW_COLUMNS + sorted(self.lattice.signals) + hp]

    @functools.cached_property
    def contexts(self) -> dict[str, dict[str, list[str]]]:
        """The recorded columns of the per-run tables, by split and callback."""
        values = self.lattice.values
        return {split: _contexts(values[split].columns) for split in SPLITS}

    def records(self, row: dict) -> list[dict]:
        """The rows of <lattice>_eval.csv of one row of <lattice>.csv."""
        head = {column: row[column] for column in EVAL_COLUMNS if column in row}
        records = []
        for split, contexts in sorted(self.contexts.items()):
            values = self.lattice.values.loc[(row["run"], row["epoch"]), split]
            for context, columns in sorted(contexts.items()):
                scalars = {c.partition("/")[2]: values[c] for c in columns}
                records.append({**head, "split": split, "context": context, **scalars})
        return records

    def eval_table(self, harvest: pd.DataFrame) -> pd.DataFrame:
        records = [r for row in harvest.to_dict("records") for r in self.records(row)]
        table = pd.DataFrame(records).sort_values(["run_name", "split", "context"])
        return table[EVAL_COLUMNS + sorted(set(table.columns) - set(EVAL_COLUMNS))]

    def write(self, out_dir: Path) -> None:
        table = self.table()
        table.to_csv(out_dir / f"{self.name}.csv", index=False)
        self.eval_table(table).to_csv(out_dir / f"{self.name}_eval.csv", index=False)
