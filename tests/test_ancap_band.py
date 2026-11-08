import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.callbacks.components import selection as sel
from src.evaluation import dump, lattice
from src.evaluation.components.statistics import at
from src.evaluation.report import Report

NAN = math.nan
BAND = {
    "level": "ancap/ancapw_f1",
    "level_se": "ancap/ancapw_f1_se",
    "se": "ancap/ancapw_se_g",
    "band_se": 2,
}
SPEC = {
    "column": "ancap/ancapw_g",
    "mode": "max",
    "key": "val/summary/ancapw_g_ema",
    "skip_nonfinite": True,
    "band": BAND,
}
SELECTION = {
    "criteria": {"ancap3_g": SPEC},
    "q_sec": {"column": "fit/normal", "se": "fit/normal_se", "key": "val/fit_ema"},
    "labelled": {},
    "lattice_criteria": {},
}
RUNS = ["r0", "r1"]
EPOCHS = 3


def test_the_band_excludes_an_untrained_level_beyond_two_standard_errors():
    level, se = [0.20, 0.80, 0.82, 0.81, 0.79], [0.01] * 5
    assert sel.band(level, se, 2).tolist() == [1, 2, 3]
    assert sel.band(level, se, 4).tolist() == [1, 2, 3, 4]
    assert sel.band(level, se, 70).tolist() == [0, 1, 2, 3, 4]


def test_the_best_level_is_the_first_of_the_maximal_and_nan_is_never_eligible():
    level, se = [NAN, 0.5, 0.9, 0.9, NAN], [0.0] * 5
    assert sel.band(level, se, 2).tolist() == [2, 3]


@pytest.mark.parametrize(
    "level, se, expected",
    [
        ([NAN, NAN, NAN], [0.1, 0.1, 0.1], [0]),
        ([0.5, 0.9, 0.89], [0.1, NAN, 0.1], [1]),
        ([0.9], [0.1], [0]),
    ],
)
def test_the_band_is_never_empty(level, se, expected):
    assert sel.band(level, se, 2).tolist() == expected


def test_every_pick_lies_in_a_non_empty_band():
    rng = np.random.default_rng(0)
    for _ in range(300):
        n = int(rng.integers(1, 12))
        values = [rng.uniform(size=n) for _ in range(5)]
        for v in values[:2]:
            v[rng.uniform(size=n) < 0.3] = NAN
        eligible = sel.band(values[0], values[1], 2)
        assert len(eligible) > 0
        assert sel.best_epoch(values[0], "max") in eligible
        for k in (0, 5, sel.W_STEPS):
            chosen = sel.banded_pick(*values, k, 2)
            assert chosen.index in eligible and chosen.eligible == len(eligible)


def test_a_flat_agreement_hands_the_pick_to_the_level():
    level, level_se = [0.90, 0.85, 0.95], [0.1] * 3
    q_main, q_sec = [0.500, 0.510, 0.505], [1.0, 1.0, 1.0]
    flat = sel.banded_pick(level, level_se, q_main, [0.01] * 3, q_sec, 10, 2)
    assert (flat.index, flat.eligible, flat.top, flat.level_decided) == (2, 3, 3, True)
    sharp = sel.banded_pick(level, level_se, q_main, [1e-4] * 3, q_sec, 10, 2)
    assert (sharp.index, sharp.top, sharp.level_decided) == (1, 1, False)


def test_the_top_group_breaks_ties_by_the_agreement_then_by_the_index():
    level, se = [0.9, 0.9, 0.9, 0.9], [0.1] * 4
    q_main, q_sec = [0.50, 0.52, 0.52, 0.51], [1.0] * 4
    assert sel.banded_pick(level, se, q_main, se, q_sec, 10, 2).index == 1


def test_below_w_steps_the_rank_sum_picks_among_the_eligible():
    # Epoch 0 has the best Q' and Q'' but a level far below the band.
    level, level_se = [0.1, 0.8, 0.82, 0.81, 0.83], [0.02] * 5
    q_main, q_main_se = [0.9, 0.2, 0.5, 0.4, 0.3], [0.01] * 5
    q_sec = [0.1, 0.5, 0.9, 0.3, 0.7]
    eligible = sel.band(level, level_se, 2)
    assert eligible.tolist() == [1, 2, 3, 4]
    for k in range(sel.W_STEPS):
        chosen = sel.banded_pick(level, level_se, q_main, q_main_se, q_sec, k, 2)
        subset = [np.take(v, eligible) for v in (q_main, q_sec)]
        assert chosen.index == eligible[sel.pick(*subset, k, "max")]
        assert chosen.top is None and chosen.level_decided is None
        assert sel.pick(q_main, q_sec, k, "max") == 0
    assert sel.banded_pick(level, level_se, q_main, q_main_se, q_sec, 0, 2).index == 3


def test_the_epoch_stage_smooths_the_level_and_the_agreement_alone():
    # A NaN prefix, as before the direction check passes.
    level = [NAN, NAN, 0.3, 0.7, 0.8, 0.82, 0.81, 0.6]
    q_main = [NAN, NAN, 0.2, 0.4, 0.45, 0.44, 0.5, 0.1]
    level_se, q_main_se = [0.02] * 8, [0.03] * 8
    q_sec = [3.0, 2.5, 2.0, 1.5, 1.4, 1.3, 1.35, 1.2]
    for k in (0, 5, sel.W_STEPS):
        found = sel.epoch_band_pick(
            level, level_se, q_main, q_main_se, q_sec, k, 2, "max", True
        )
        smoothed = [sel.smoothed(v, skip_nonfinite=True) for v in (level, q_main)]
        expected = sel.banded_pick(
            smoothed[0], level_se, smoothed[1], q_main_se, sel.smoothed(q_sec), k, 2
        )
        assert found == expected
        assert found.index >= 2


def _trained(level, level_se, q_main, q_main_se, q_sec) -> list[dict]:
    columns = [BAND["level"], BAND["level_se"], SPEC["column"], BAND["se"]]
    columns.append(SELECTION["q_sec"]["column"])
    rows = zip(level, level_se, q_main, q_main_se, q_sec)
    return [{"epoch": e, **dict(zip(columns, r))} for e, r in enumerate(rows)]


@pytest.mark.parametrize("k", [0, 4, sel.W_STEPS])
def test_the_dump_stage_picks_a_banded_criterion_by_its_band_rule(k):
    run = object.__new__(dump.RunEvaluation)
    run.strategies = dump._strategies(SELECTION)
    values = ([NAN, 0.3, 0.8, 0.81], [0.01] * 4, [NAN, 0.9, 0.4, 0.5], [0.01] * 4)
    q_sec = [2.0, 0.5, 1.0, 1.1]
    expected = sel.epoch_band_pick(*values, q_sec, k, 2, "max", True).index
    assert run._pick("ancap3_g", _trained(*values, q_sec), k) == expected


def _arrays(level_r0: float) -> dict[str, np.ndarray]:
    # Run r0 is barely trained, unless level_r0 lifts it into the band; its agreement
    # is the higher.
    nan = np.full((2, EPOCHS + 1), NAN)
    return {
        BAND["level"]: np.array([[NAN, 0.2, 0.25, level_r0], [NAN, 0.5, 0.8, 0.79]]),
        BAND["level_se"]: np.full((2, EPOCHS + 1), 0.01),
        SPEC["column"]: np.array([[NAN, 0.5, 0.6, 0.9], [NAN, 0.3, 0.4, 0.41]]),
        BAND["se"]: np.full((2, EPOCHS + 1), 0.01),
        "fit/normal": np.array([[2.0, 1.5, 1.2, 1.0], [2.0, 1.6, 1.3, 1.1]]),
        "fit/normal_se": np.full((2, EPOCHS + 1), 0.01),
        "ema/ancap3_g": nan,
        "ema/q_sec": nan,
    }


def _lattice(level_r0: float) -> lattice.Lattice:
    index = pd.MultiIndex.from_product(
        [RUNS, range(-1, EPOCHS)], names=["run", "epoch"]
    )
    columns = {("val", c): v.ravel() for c, v in _arrays(level_r0).items()}
    rows = {("ancap3_g", k): [2, 2] for k in range(sel.W_STEPS + 1)}
    rows[("q_sec", NAN)] = [2, 2]
    picks = pd.DataFrame(list(rows.values()), columns=RUNS)
    picks.index = pd.MultiIndex.from_tuples(list(rows), names=["strategy", "k"])
    values = pd.DataFrame(columns, index=index)
    return lattice.Lattice([], values, picks, SELECTION, None, None, [])


@pytest.mark.parametrize("level_r0, point", [(0.3, 1), (0.79, 0)])
def test_the_run_stage_bands_the_runs_at_their_epochs(level_r0, point):
    lat = _lattice(level_r0)
    strategy = lat.strategies[0]
    assert isinstance(strategy, lattice.BandedStrategy)
    assert [s.name for s in lat.strategies] == ["ancap3_g", "q_sec"]
    epochs = np.array([2, 2])
    arrays = _arrays(level_r0)
    names = [BAND["level"], BAND["level_se"], SPEC["column"], BAND["se"]]
    values = [at(arrays[n], epochs) for n in [*names, "fit/normal"]]
    for k in range(sel.W_STEPS + 1):
        expected = sel.banded_pick(*values, k, 2)
        assert strategy.choose(k) == (expected.index, 2)
        assert expected.index == point
    assert strategy.choose(5, sel.W_STEPS) == (point, 2)


def test_the_band_table_reports_every_pick():
    lat = _lattice(0.3)
    table = Report(lat).band()
    assert list(table.columns) == [
        "strategy",
        "k",
        "point",
        "epoch",
        "eligible_epochs",
        "eligible_runs",
        "top_group",
        "level_decided",
    ]
    assert list(table["k"]) == list(range(sel.W_STEPS + 1))
    assert set(table["point"]) == {"r1"} and set(table["eligible_runs"]) == {1}
    level = sel.smoothed(_arrays(0.3)[BAND["level"]][1, 1:], skip_nonfinite=True)
    expected = len(sel.band(level, [0.01] * EPOCHS, 2))
    assert set(table["eligible_epochs"]) == {expected}
    last = table.iloc[-1]
    assert last["top_group"] == 1 and not last["level_decided"]
    assert table.iloc[:-1]["top_group"].isna().all()


def test_a_registry_without_a_banded_criterion_writes_no_band_table(tmp_path):
    lat = _lattice(0.3)
    spec = {key: value for key, value in SPEC.items() if key != "band"}
    plain = {**SELECTION, "criteria": {"ancap3_g": spec}}
    unbanded = lattice.Lattice([], lat.values, lat.picks, plain, None, None, [])
    assert type(unbanded.strategies[0]) is lattice.Strategy
    assert not Report(unbanded).banded
    assert Report(lat).banded


PLAIN = {"column": "x/plain", "mode": "min", "key": "val/summary/plain_ema"}
DUMP_SELECTION = SELECTION | {
    "criteria": {"plain": PLAIN, "ancap3_g": SPEC},
    "efficiency": "eff/",
}


def _run_evaluation() -> dump.RunEvaluation:
    run = object.__new__(dump.RunEvaluation)
    run.selection, run.strategies = DUMP_SELECTION, dump._strategies(DUMP_SELECTION)
    run.signals, run.nulls, run.run_dir = {"val": [], "test": ["s"]}, [], Path("r0")
    return run


def _dump_rows(working_point: bool) -> dict[str, list[dict]]:
    rng = np.random.default_rng(3)
    val, test = [], []
    for epoch in range(-1, 6):
        row = {"epoch": epoch, "x/plain": rng.uniform(), "fit/normal": rng.uniform()}
        banded = {column: rng.uniform() for column in sel.band_columns(SPEC)}
        val.append(row | banded if working_point else row)
        test.append({"epoch": epoch, "eff/s": rng.uniform(), "eff/normal": 0.01})
    return {"val": val, "test": test}


def _without_band(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row["strategy"] != "ancap3_g"]


def test_the_dump_stage_skips_a_banded_criterion_without_its_columns(caplog):
    present = _run_evaluation()._tables(_dump_rows(True))
    with caplog.at_level(logging.INFO, logger="src.evaluation.dump"):
        absent = _run_evaluation()._tables(_dump_rows(False))
    notes = [r.message for r in caplog.records if "ancap3_g" in r.message]
    assert len(notes) == 1 and notes[0].startswith("r0: ")
    assert {r["strategy"] for r in present["picks"]} == {"plain", "ancap3_g", "q_sec"}
    assert {r["strategy"] for r in absent["picks"]} == {"plain", "q_sec"}
    assert _without_band(present["picks"]) == absent["picks"]
    assert "ema/ancap3_g" in present["val"][1]
    assert "ema/ancap3_g" not in absent["val"][1]
    emas = [[row["ema/plain"] for row in t["val"]] for t in (present, absent)]
    assert emas[0] == emas[1]


@pytest.mark.parametrize("lacking", ["columns", "picks"])
def test_the_run_stage_skips_a_banded_criterion_the_tables_lack(lacking, caplog):
    lat = _lattice(0.3)
    values, picks = lat.values, lat.picks
    if lacking == "columns":
        values = values.drop(columns=[("val", c) for c in sel.band_columns(SPEC)])
    else:
        picks = picks[picks.index.get_level_values("strategy") != "ancap3_g"]
    with caplog.at_level(logging.INFO, logger="src.evaluation.lattice"):
        skipped = lattice.Lattice([], values, picks, SELECTION, None, None, [])
        assert [s.name for s in skipped.strategies] == ["q_sec"]
    assert [r.message for r in caplog.records if "ancap3_g" in r.message]
    assert not Report(skipped).banded


def _random_lattice(criteria: dict, runs: int = 6, epochs: int = 4):
    rng = np.random.default_rng(11)
    names, size = [f"r{i}" for i in range(runs)], runs * (epochs + 1)
    columns = [*sel.band_columns(SPEC), "x/plain", "fit/normal", "fit/normal_se"]
    columns += ["ema/ancap3_g", "ema/plain", "ema/q_sec"]
    data = {("val", c): rng.uniform(size=size) for c in columns}
    data[("test", "test_mean_eff")] = rng.uniform(size=size)
    index = pd.MultiIndex.from_product([names, range(-1, epochs)])
    index.names = ["run", "epoch"]
    keys = [(s, k) for s in ("ancap3_g", "plain") for k in range(sel.W_STEPS + 1)]
    rows = {key: rng.integers(epochs, size=runs) for key in keys}
    rows[("q_sec", NAN)] = rng.integers(epochs, size=runs)
    picks = pd.DataFrame(list(rows.values()), columns=names)
    picks.index = pd.MultiIndex.from_tuples(list(rows), names=["strategy", "k"])
    selection = SELECTION | {"criteria": criteria}
    values = pd.DataFrame(data, index=index)
    return lattice.Lattice([], values, picks, selection, None, None, [])


def test_a_banded_strategy_leaves_the_holm_family_of_the_others():
    with_band = Report(_random_lattice({"plain": PLAIN, "ancap3_g": SPEC})).validity()
    without = Report(_random_lattice({"plain": PLAIN})).validity()
    banded = with_band["strategy"] == "ancap3_g"
    assert banded.sum() == sel.W_STEPS + 1
    assert with_band["p"][banded].notna().any()
    assert with_band["p_holm"][banded].isna().all()
    pd.testing.assert_frame_equal(with_band[~banded].reset_index(drop=True), without)
    assert without["p_holm"].notna().any()
