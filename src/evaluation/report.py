"""The selection tables of the lattice step; ``src.evaluation.lattice`` lists their
columns and the rules behind them."""

import functools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.lattice import Lattice, Strategy


def _holm(p: np.ndarray) -> np.ndarray:
    """Holm's adjusted p-values over the finite entries: the running maximum of
    (m - j) p_(j), capped at 1."""
    adjusted = np.full(len(p), math.nan)
    finite = np.flatnonzero(np.isfinite(p))
    order = finite[np.argsort(p[finite], kind="stable")]
    scaled = (len(order) - np.arange(len(order))) * p[order]
    adjusted[order] = np.minimum(np.maximum.accumulate(scaled), 1.0)
    return adjusted


def _write(path: Path, table: pd.DataFrame) -> None:
    if "k" in table:
        # Nullable, so that the k of a pick by Q' alone is written empty rather than
        # as a NaN that turns every k into a float.
        table = table.astype({"k": "Int64"})
    table.to_csv(path, index=False)


@dataclass(frozen=True, eq=False)
class Report:
    """The six tables of the lattice step, built row by row by the strategies."""

    lattice: Lattice

    @functools.cached_property
    def pairs(self) -> list[tuple[Strategy, int | None]]:
        """Every strategy at each of its k."""
        return [(s, k) for s in self.lattice.strategies for k in s.ks]

    @functools.cached_property
    def grid(self) -> list[tuple[Strategy, int]]:
        """The pairs of the strategies that pick on the grid of w."""
        return [(strategy, k) for strategy, k in self.pairs if k is not None]

    def consensus(self) -> pd.DataFrame:
        """The lattice criteria and their EMAs by run and epoch, NaN where undefined."""
        found = {strategy.name: strategy for strategy in self.lattice.strategies}
        names = self.lattice.selection["lattice_criteria"]
        table = self.lattice.values.index.to_frame(index=False)
        for name in names:
            table[name] = found[name].q_main.ravel() if name in found else math.nan
        for name in names:
            values = found[name].ema.ravel() if name in found else math.nan
            table[f"ema/{name}"] = values
        return table

    def selection(self) -> pd.DataFrame:
        return pd.DataFrame([s.selection_row(k, self.lattice) for s, k in self.pairs])

    def validity(self) -> pd.DataFrame:
        table = pd.DataFrame([s.validity_row(k, self.lattice) for s, k in self.pairs])
        table["p_holm"] = _holm(table["p"].to_numpy(dtype=np.float64))
        return table

    def tradeoff(self) -> pd.DataFrame:
        rows = [r for s, k in self.grid for r in s.tradeoff_rows(k, self.lattice)]
        return pd.DataFrame(rows)

    def oracle_position(self) -> pd.DataFrame:
        rows = [s.oracle_position_row(k, self.lattice) for s, k in self.grid]
        return pd.DataFrame(rows)

    def interventions(self) -> pd.DataFrame:
        rows = [r for s, k in self.pairs for r in s.intervention_rows(k, self.lattice)]
        return pd.DataFrame(rows)

    def write(self, out_dir: Path) -> None:
        """Write ``<out_dir>/<table>.csv``, once every table is built."""
        tables = {
            "consensus": self.consensus(),
            "selection": self.selection(),
            "validity": self.validity(),
            "tradeoff": self.tradeoff(),
            "oracle_position": self.oracle_position(),
            "interventions": self.interventions(),
        }
        for name, table in tables.items():
            _write(out_dir / f"{name}.csv", table)
