"""The lattice step of the evaluation stage: from the per-run outputs in
``<out_dir>/runs/<run>/`` and the score dumps, ``select_lattice`` writes the tables of
``src.evaluation.report``:

- consensus.csv: run, epoch, udr, mc, ema/udr, ema/mc (udr empty without a partner
  and for a run left without one)
- selection.csv: strategy, k, w, point, epoch, q_main, q_sec, test_eff,
  oracle_test_eff, regret, regret_lo, regret_hi, regret_epoch, regret_point,
  realised_fpr, null_rate, floor_mean, floor_p05, floor_p95, limit, best_test_eff
- validity.csv: strategy, k, rho, p, p_holm, n
- tradeoff.csv: strategy, k_epoch, k_point (0 to 10, gate), point, epoch, test_eff,
  regret, joint_only
- oracle_position.csv: strategy, k_epoch, oracle_point, rank_q_main, rank_q_sec, n
- interventions.csv: strategy, k, intervention, target_cluster, test_eff,
  oracle_test_eff

and those of ``src.evaluation.harvest``:

- <lattice>.csv, <lattice>_eval.csv: the schema of scripts/analysis/harvest_pool_effs.py

The readers of the per-run outputs and the dumps, the pairing with the partner
lattice, the lattice criteria and the statistics of the tables are in
``src.evaluation.components`` (``tables``, ``pairing``, ``udr``, ``statistics``).
"""

import functools
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.callbacks.components.selection import W_STEPS, ema, epoch_pick, pick, ranks
from src.evaluation.components.pairing import partner_scores
from src.evaluation.components.statistics import at, operational_se, spearman_test
from src.evaluation.components.tables import (
    SPLITS,
    load_lattice,
    read_picks,
    read_values,
)
from src.evaluation.components.udr import LATTICE_CRITERIA

BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 10_000, 20261020


@dataclass(frozen=True, eq=False)
class Strategy:
    """Q' and its EMA, (runs, epochs + 1) with epoch -1 first, the Q'' it picks with and
    every run's epoch by k_epoch, a single None key for a pick by Q' alone.

    The row builders take the lattice for what every strategy shares: the run names,
    the test values and the oracle's pick.
    """

    name: str
    mode: str
    q_main: np.ndarray
    q_sec: np.ndarray
    ema: np.ndarray  # NaN at epoch -1
    picks: dict[int | None, np.ndarray]
    key: str = ""  # the MLflow key of the EMA, empty when nothing logs one

    @classmethod
    def by_epoch_rule(
        cls, name: str, mode: str, q_main: np.ndarray, q_sec: np.ndarray
    ) -> "Strategy":
        """The strategy whose EMA and epochs the epoch rule gives from Q' and Q''."""
        trained = list(zip(q_main[:, 1:], q_sec[:, 1:]))
        ema_values = np.array([[math.nan, *ema(v)] for v, _ in trained])
        picks = {
            k: np.array([epoch_pick(v, s, k, mode) for v, s in trained])
            for k in range(W_STEPS + 1)
        }
        return cls(name, mode, q_main, q_sec, ema_values, picks)

    @property
    def ks(self) -> list[int | None]:
        return [None] if None in self.picks else list(range(W_STEPS + 1))

    def candidates(self, k_epoch: int | None) -> tuple[np.ndarray, ...]:
        """Every run's epoch at k_epoch, and its Q' and Q'' there."""
        epochs = self.picks[k_epoch]
        return epochs, at(self.q_main, epochs), at(self.q_sec, epochs)

    def choose(
        self, k_epoch: int | None, k_point: int | None = None
    ) -> tuple[int, int]:
        """(run index, epoch) of the pick, at k_point = k_epoch unless given, and at
        W_STEPS for Q' alone."""
        if k_point is None:
            k_point = W_STEPS if k_epoch is None else k_epoch
        epochs, q_main, q_sec = self.candidates(k_epoch)
        point = pick(q_main, q_sec, k_point, self.mode)
        return point, int(epochs[point])

    def ranking_value(self, k: int | None) -> np.ndarray:
        """The value a pick at k ranks the runs by, larger for the better run."""
        _, q_main, q_sec = self.candidates(k)
        if k is None or k == W_STEPS:
            return q_main if self.mode == "max" else -q_main
        if k == 0:
            return -q_sec
        rank_sum = k * ranks(q_main, self.mode) + (W_STEPS - k) * ranks(q_sec, "min")
        return -rank_sum

    def gate(self, k_epoch: int, lattice: "Lattice") -> tuple[int, int]:
        """(run index, epoch) of the best Q' among the runs whose Q'' lies within one
        bootstrap standard error of the best Q''."""
        epochs, q_main, q_sec = self.candidates(k_epoch)
        best = pick(q_main, q_sec, 0, self.mode)
        bound = q_sec[best] + lattice.gate_se(best, int(epochs[best]))
        within = np.union1d(np.flatnonzero(q_sec <= bound), [best])
        point = int(within[pick(q_main[within], q_sec[within], W_STEPS, self.mode)])
        return point, int(epochs[point])

    def selection_row(self, k: int | None, lattice: "Lattice") -> dict:
        point, epoch = self.choose(k)
        cell, limit = (point, epoch + 1), lattice.limits.get(self.name, "")
        row = {"strategy": self.name, "k": k, "w": None if k is None else k / W_STEPS}
        row.update(point=lattice.names[point], epoch=epoch, q_main=self.q_main[cell])
        row.update(q_sec=self.q_sec[cell], **lattice.regret(point, epoch))
        row.update({name: rate[cell] for name, rate in lattice.rates.items()})
        row.update(lattice.floor(self.picks[k]), limit=limit)
        best = np.nanmax(lattice.test_eff[:, 1:]) if self.name == "oracle" else None
        return {**row, "best_test_eff": best}

    def validity_row(self, k: int | None, lattice: "Lattice") -> dict:
        value, test = self.ranking_value(k), at(lattice.test_eff, self.picks[k])
        kept = np.isfinite(value) & np.isfinite(test)
        rho, p = spearman_test(value[kept], test[kept])
        row = {"strategy": self.name, "k": k, "rho": rho, "p": p}
        return {**row, "p_holm": math.nan, "n": int(kept.sum())}

    def tradeoff_rows(self, k_epoch: int, lattice: "Lattice") -> list[dict]:
        ends = {self.choose(k_epoch, k)[0] for k in (0, W_STEPS)}
        k_points = [*range(W_STEPS + 1), "gate"]
        return [self._tradeoff_row(k_epoch, k, lattice, ends) for k in k_points]

    def _tradeoff_row(
        self, k_epoch: int, k_point: int | str, lattice: "Lattice", ends: set[int]
    ) -> dict:
        if k_point == "gate":
            point, epoch = self.gate(k_epoch, lattice)
        else:
            point, epoch = self.choose(k_epoch, k_point)
        test_eff = lattice.test_eff[point, epoch + 1]
        row = {"strategy": self.name, "k_epoch": k_epoch, "k_point": k_point}
        row.update(point=lattice.names[point], epoch=epoch, test_eff=test_eff)
        row.update(regret=test_eff - lattice.oracle_test_eff)
        return {**row, "joint_only": point not in ends}

    def oracle_position_row(self, k_epoch: int, lattice: "Lattice") -> dict:
        epochs, q_main, q_sec = self.candidates(k_epoch)
        point = lattice.oracle_pick[0]
        row = {"strategy": self.name, "k_epoch": k_epoch}
        row.update(oracle_point=lattice.names[point])
        row.update(rank_q_main=ranks(q_main, self.mode)[point])
        return {**row, "rank_q_sec": ranks(q_sec, "min")[point], "n": len(epochs)}

    def intervention_rows(self, k: int | None, lattice: "Lattice") -> list[dict]:
        (point, epoch), (o_point, o_epoch) = self.choose(k), lattice.oracle_pick
        picked = lattice.set_effs["test"][point, epoch + 1]
        oracle = lattice.set_effs["test"][o_point, o_epoch + 1]
        keys = ("intervention", "target_cluster", "test_eff", "oracle_test_eff")
        values = zip(lattice.signals, lattice.clusters, picked, oracle)
        return [{"strategy": self.name, "k": k, **dict(zip(keys, v))} for v in values]


@dataclass(frozen=True, eq=False)
class Lattice:
    """The per-run outputs of a lattice and the stored scores of its dumps.

    ``values`` holds every run's val.csv and test.csv by (run, epoch), with the columns
    (split, column), and ``picks`` every run's epoch by (strategy, k), one column per
    run; the runs are in name order, so that a tie between runs goes to the first.
    Every array taken from them is (runs, epochs + 1) with epoch -1 first, followed by
    an axis over the signal sets for their efficiencies.
    """

    info: list[dict]  # run.json, in run order
    values: pd.DataFrame
    picks: pd.DataFrame
    selection: dict  # the selection node of the eval domain config
    normal: np.ndarray  # stored validation normal scores, (runs, epochs + 1, n)
    # Those of each run's partner, (epochs + 1, n) or None for a run left without one;
    # None without a partner lattice.
    partner: list[np.ndarray | None] | None
    metas: list[dict]  # meta.pt, in run order

    def __post_init__(self) -> None:
        if self.partner is None:
            return
        shapes = {scores.shape for scores in self.partner if scores is not None}
        if len(self.partner) != len(self.normal) or shapes - {self.normal.shape[1:]}:
            raise ValueError(
                f"Partner scores of {len(self.partner)} runs, {sorted(shapes)}; "
                f"lattice {self.normal.shape}."
            )

    @classmethod
    def read(
        cls,
        out_dir: Path,
        selection: dict,
        scores_dir: Path,
        lattice: str,
        partner: str | None,
        drop_unpaired: bool = False,
    ) -> "Lattice":
        """Read a lattice; the arguments are those of ``select_lattice``."""
        dirs = sorted(path for path in (out_dir / "runs").iterdir() if path.is_dir())
        names = [path.name for path in dirs]
        info = [json.loads((path / "run.json").read_text()) for path in dirs]
        values, picks = read_values(dirs, info), read_picks(dirs)
        normal, metas = load_lattice(scores_dir / lattice, names)
        paired = partner_scores(
            scores_dir, lattice, partner, names, info, drop_unpaired
        )
        return cls(info, values, picks, selection, normal, paired, metas)

    @functools.cached_property
    def names(self) -> list[str]:
        return self.values.index.unique("run").tolist()

    def matrix(self, split: str, columns: list[str]) -> np.ndarray:
        """Columns of the per-run tables of a split, (runs, epochs + 1, columns)."""
        table = self.values[[(split, column) for column in columns]]
        shape = len(self.names), -1, len(columns)
        return table.to_numpy(dtype=np.float64).reshape(shape)

    def column(self, split: str, column: str) -> np.ndarray:
        return self.matrix(split, [column])[..., 0]

    @functools.cached_property
    def q_sec(self) -> np.ndarray:
        return self.column("val", self.selection["q_sec"]["column"])

    @functools.cached_property
    def test_eff(self) -> np.ndarray:
        return self.column("test", "test_mean_eff")

    @functools.cached_property
    def rates(self) -> dict[str, np.ndarray]:
        """The realised test false-positive rate and the null rate."""
        names = ("realised_fpr", "null_rate")
        return {name: self.column("test", name) for name in names}

    @functools.cached_property
    def set_effs(self) -> dict[str, np.ndarray]:
        """The efficiencies of the signal sets by split."""
        columns = [self.selection["efficiency"] + name for name in self.signals]
        return {split: self.matrix(split, columns) for split in SPLITS}

    @functools.cached_property
    def signals(self) -> list[str]:
        signals = [list(run["signals"]) for run in self.info]
        if any(names != signals[0] for names in signals):
            raise ValueError("The runs name different signal sets.")
        return signals[0]

    @functools.cached_property
    def clusters(self) -> list[str]:
        """The target cluster of every signal set; without a file each is its own."""
        path = self.selection.get("clusters")
        if path is None:
            return list(self.signals)
        table = pd.read_csv(path, dtype=str).set_index("intervention")["target_cluster"]
        missing = sorted(set(self.signals) - set(table.index))
        if missing:
            raise ValueError(f"{path} gives no target cluster for {missing}.")
        return [table[name] for name in self.signals]

    @functools.cached_property
    def limits(self) -> dict[str, str]:
        return dict(self.selection.get("limits") or {})

    def picked_epochs(self, name: str) -> dict[int | None, np.ndarray]:
        """Every run's epoch of a strategy by k, None for a pick by Q' alone."""
        rows = self.picks[self.picks.index.get_level_values("strategy") == name]
        if rows.empty:
            raise ValueError(f"picks.csv holds no epochs of {name}.")
        ks = [None if pd.isna(k) else int(k) for k in rows.index.get_level_values("k")]
        return dict(zip(ks, rows.to_numpy()))

    def _column_strategy(self, name: str, column: str, mode: str, key: str) -> Strategy:
        """The strategy of a column of val.csv, with its EMA in ``ema/<name>``."""
        picks = self.picked_epochs(name)
        q_main, ema_values = (self.column("val", c) for c in (column, f"ema/{name}"))
        return Strategy(name, mode, q_main, self.q_sec, ema_values, picks, key)

    def _criterion_strategies(self) -> list[Strategy]:
        modes = self.selection["lattice_criteria"]
        values = {n: LATTICE_CRITERIA[n](self.normal, self.partner) for n in modes}
        return [
            Strategy.by_epoch_rule(name, modes[name], v, self.q_sec)
            for name, v in values.items()
            if v is not None
        ]

    @functools.cached_property
    def strategies(self) -> list[Strategy]:
        """The criteria, the lattice criteria, the labelled rows and q_sec, in order."""
        criteria = self.selection["criteria"].items()
        specs = [(name, c["column"], c["mode"], c["key"]) for name, c in criteria]
        strategies = [self._column_strategy(*spec) for spec in specs]
        strategies += self._criterion_strategies()
        labelled = self.selection["labelled"].items()
        specs = [(name, name, "max", c["key"]) for name, c in labelled]
        q_sec = self.selection["q_sec"]
        specs.append(("q_sec", q_sec["column"], "min", q_sec["key"]))
        return strategies + [self._column_strategy(*spec) for spec in specs]

    @functools.cached_property
    def oracle(self) -> Strategy:
        return next(s for s in self.strategies if s.name == "oracle")

    @functools.cached_property
    def oracle_pick(self) -> tuple[int, int]:
        """(run index, epoch) of the oracle's pick."""
        return self.oracle.choose(None)

    @functools.cached_property
    def oracle_test_eff(self) -> float:
        point, epoch = self.oracle_pick
        return self.test_eff[point, epoch + 1]

    @functools.cached_property
    def gate_se(self) -> Callable[[int, int], float]:
        """Bootstrap standard error of Q'' by (run index, epoch), computed once each."""

        @functools.cache
        def standard_error(point: int, epoch: int) -> float:
            meta = self.metas[point]
            layout = meta["val"]["batch_sizes"]["normal"]
            scores = self.normal[point, epoch + 1]
            return operational_se(scores, layout, meta["target_fpr"])

        return standard_error

    @functools.cached_property
    def bootstrap_draws(self) -> np.ndarray:
        """Indices into the sorted target clusters, (BOOTSTRAP_DRAWS, clusters)."""
        # One stream for every row, so that the intervals of two strategies share draws.
        clusters = len(set(self.clusters))
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        return rng.integers(clusters, size=(BOOTSTRAP_DRAWS, clusters))

    def cluster_interval(self, differences: np.ndarray) -> tuple:
        """95 % percentile interval of the mean per-set difference, the target clusters
        resampled with replacement and every set of a drawn cluster counted."""
        codes = np.unique(self.clusters, return_inverse=True)[1]
        sums, sizes = np.bincount(codes, weights=differences), np.bincount(codes)
        draws = self.bootstrap_draws
        means = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
        return tuple(np.percentile(means, [2.5, 97.5]))

    def regret(self, point: int, epoch: int) -> dict:
        """The regret columns of selection.csv for a pick (run index, epoch)."""
        (o_point, o_epoch), sets = self.oracle_pick, self.set_effs["test"]
        paired = sets[point, epoch + 1] - sets[o_point, o_epoch + 1]
        test_eff, oracle_eff = self.test_eff[point, epoch + 1], self.oracle_test_eff
        regret, oracle_epoch = test_eff - oracle_eff, self.oracle.picks[None][point]
        regret_epoch = test_eff - self.test_eff[point, oracle_epoch + 1]
        row = {"test_eff": test_eff, "oracle_test_eff": oracle_eff, "regret": regret}
        lo, hi = self.cluster_interval(paired)
        row.update(regret_lo=lo, regret_hi=hi, regret_epoch=regret_epoch)
        return {**row, "regret_point": regret - regret_epoch}

    def floor(self, epochs: np.ndarray) -> dict:
        """The random-pick floor at a strategy's epochs of every run."""
        values = at(self.test_eff, epochs)
        p05, p95 = np.percentile(values, [5, 95])
        return {"floor_mean": values.mean(), "floor_p05": p05, "floor_p95": p95}


def select_lattice(
    out_dir: Path,
    selection: dict,
    scores_dir: Path,
    lattice: str,
    partner: str | None,
    drop_unpaired: bool = False,
) -> None:
    """Compute the lattice criteria, apply every strategy and write the tables.

    :param out_dir: The stage's output directory of the lattice, whose ``runs/``
        holds the per-run outputs of ``Evaluator.evaluate_lattice``.
    :param selection: The ``selection`` node of the eval domain config: ``criteria``,
        ``q_sec``, ``labelled``, ``efficiency``, ``clusters`` (a CSV with columns
        intervention and target_cluster, or None), ``lattice_criteria`` (name: mode)
        and ``limits`` (strategy: note).
    :param scores_dir: The root of the dumps, ``<scores_dir>/<lattice>/<run>/``.
    :param lattice: The lattice's name, its directory under ``scores_dir``.
    :param partner: The lattice whose runs, paired by hyperpoint, give UDR; None
        leaves UDR out.
    :param drop_unpaired: Whether a run without a partner run of equal hyperpoint is
        left out of UDR (its UDR NaN) instead of raising ValueError.
    """
    # Imported here, since both modules import this one.
    from src.evaluation.harvest import Harvest
    from src.evaluation.report import Report

    out_dir, scores_dir = Path(out_dir), Path(scores_dir)
    lat = Lattice.read(out_dir, selection, scores_dir, lattice, partner, drop_unpaired)
    Report(lat).write(out_dir)
    Harvest(lat, lattice).write(out_dir)
