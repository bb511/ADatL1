import functools
import math

import numpy as np
import pandas as pd
import pytest

from src.callbacks import ancap8 as a8
from src.evaluation.components.ancap8 import BOUND_CONST, COLUMNS, factor, tables
from tests.test_ancap8 import (
    COND_LATTICE,
    DEEP,
    FAMILY_P,
    LOW,
    S1,
    TOL,
    cond_families,
    grid,
    kinds4,
    level,
    points,
    tied,
)

# The planted picks of scratchpad/adl1t/theory/theory_xxiv/step4/FAB.md, FAB_A1.md and
# FAB_A2.md and theory_xxvii/step4/FAB_A3.md, under the design's reading (U2 tenths, U4
# top), which FAB_A3.md equates with `strict` on every planted lattice but its own.
CELL = "c"
FILLERS = [f"z{i:02d}" for i in range(12)]
INF = -math.inf


def _with_nan(arrays: tuple, which: int) -> tuple:
    out = [x.copy() for x in arrays]
    out[which][7] = math.nan
    return tuple(out)


def _g_tilt() -> dict[str, tuple]:
    rng = np.random.default_rng([107, 0])
    # The three uniform draws of FAB.md's generic rule, made and discarded.
    rng.uniform(1, 5), rng.uniform(-0.2, 1.0), rng.uniform(0.5, 2.0)
    s1 = rng.standard_normal(1000)
    s2 = 1.0 + rng.standard_normal(5400)
    p = 3.0 + rng.standard_normal(4456)
    s1, s2, p = (x.astype(np.float32).astype(np.float64) for x in (s1, s2, p))
    v = (np.sort(s1)[-9] + np.sort(s1)[-10]) / 2
    moved = p.copy()
    moved[np.argsort(p, kind="stable")[:10]] = v
    filler = (grid(0, 1, 1000), grid(0, 1, 5400) + 0.00001, points((4456, 0.2)))
    return {"x": (s1, s2, p), "y": (s1, s2, moved), "tilt_filler": filler, "v": v}


# FAB_A3.md's lattice `tenths`: the ranks tied in s1 and s2, and the points at 5.0.
TENTHS = {
    "full": (None, None, 500),
    "edge18": ((19, 21), (19, 21), 600),
    "under17": ((18, 21), (18, 21), 700),
    "one_side": (None, (18, 21), 800),
    "tenths_filler": (None, None, 0),
}


def _tenths(tie_1, tie_2, deep: int) -> tuple:
    s1, s2 = grid(0, 1, 2050), grid(0, 1, 2050) + 0.0001
    s1 = s1 if tie_1 is None else tied(s1, *tie_1)
    s2 = s2 if tie_2 is None else tied(s2, *tie_2)
    return s1, s2, points((deep, 5.0), (1000 - deep, 0.2))


# FAB.md's lattice `guard`: the points at 5.0 of each candidate.
GUARD_POINTS = {"u": 300, "e": 700, "t": 600, "w": 500}
TILT = _g_tilt()
ARRAYS = dict(FAMILY_P) | {k: _tenths(*v) for k, v in TENTHS.items()}
ARRAYS |= {
    k: (S1, DEEP, points((n, 5.0), (1000 - n, 0.2))) for k, n in GUARD_POINTS.items()
}
ARRAYS |= {"bad": _with_nan(FAMILY_P["cap"], 1), "n": _with_nan(FAMILY_P["cap"], 0)}
ARRAYS |= {k: TILT[k] for k in ("x", "y", "tilt_filler")} | cond_families()
ARRAYS["edge_filler"] = ARRAYS["edge_read"][:2] + (points((100, LOW)),)


@functools.cache
def values(name: str) -> dict:
    return a8.candidate(*ARRAYS[name])


def lattice(members: list[tuple[str, int, str]]) -> dict[str, pd.DataFrame]:
    # One cell; a member is (run, epoch, the name of its arrays).
    rows = [
        {"run": run, "epoch": epoch} | {f"{k}_{CELL}": v for k, v in values(a).items()}
        for run, epoch, a in members
    ]
    return tables(pd.DataFrame(rows), [CELL], 0)


def fillers(epoch: int, arrays: str = "dull", names=FILLERS) -> list[tuple]:
    return [(name, epoch, arrays) for name in names]


def picks(frames: dict) -> dict:
    return frames["picks"].iloc[0].to_dict()


def empty(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def assert_row(got: dict, want: dict) -> None:
    for key, value in want.items():
        if value is None:
            assert empty(got[key]), key
        elif isinstance(value, float) and math.isfinite(value):
            assert abs(got[key] - value) <= TOL, key
        else:
            assert got[key] == value, key


SEP = [("b_deep", 10, "b_deep"), ("a_shallow", 10, "a_shallow")]
SEP_H = SEP + [("d_disputed", 10, "d_disputed")] + fillers(10)
GUARD_H = [("u", -1, "u"), ("e", 3, "e"), ("t", 50, "t"), ("w", 60, "w")]
GUARD_H += fillers(70)
ALL_PICK = {f"top_{g}": "pick" for g in "abcde"}


def test_separation_sep_h():
    got = picks(lattice(SEP_H))
    want = {"n_trained": 15, "n_not_eligible": 0, "n_unusable": 0, "near_size": 1}
    want |= {"top": "b_deep@10", "pick_top": "b_deep@10", "abstain_top": 0}
    want |= {"ell": "d_disputed@10", "c2_pick": "d_disputed@10", "same_top": 0}
    want |= {"count_pick_top": "d_disputed@10", "count_top": "d_disputed@10"}
    want |= {"B": INF, "B0": INF, "B1": INF, "J": INF, "early_top": 0} | ALL_PICK
    assert_row(got, want | {"given_up_top": 0.35})
    assert values("d_disputed")["Cbar2"] == 0.65 and values("b_deep")["Cbar2"] == 0.6
    # Cbar2 = 0.6 exactly, and every flagged point lies deep: the charge vanishes.
    assert_row(got, {"pick_ceiling": level(0.6), "pick_charge": 0.0})


def test_separation_sep2_h():
    got = picks(lattice(SEP + fillers(10)))
    want = {"top": "b_deep@10", "pick_top": "b_deep@10", "c2_pick": "a_shallow@10"}
    want |= {"count_pick_top": "b_deep@10", "same_top": 1} | ALL_PICK
    assert_row(got, want)
    assert values("a_shallow")["Cbar2"] < 0.55


def test_unusable_candidate_takes_no_part():
    plain = lattice(SEP_H)
    with_bad = lattice(SEP_H + [("bad", 10, "bad")])
    assert picks(with_bad)["n_unusable"] == 1
    for key, value in picks(plain).items():
        other = picks(with_bad)[key]
        if key != "n_unusable":
            assert other == value or (empty(value) and empty(other)), key
    row = with_bad["candidates"].set_index("run").loc["bad"].to_dict()
    assert (row["usable"], row["trained"], row["n1"], row["nP"]) == (0, 0, 2000, 1000)
    assert all(empty(row[k]) for k in ("I", "L", "k1", "elig_tenths"))


def test_guard_h():
    got = picks(lattice(GUARD_H))
    want = {"n_trained": 15, "top": "e@3", "pick_top": "e@3", "early_top": 1}
    want |= {"B": 700, "B0": 300, "B1": 300, "J": round(values("e")["I"], 9)}
    want |= {"top_a": "guard", "top_b": "pick", "top_c": "pick", "top_d": "pick"}
    want |= {"top_e": "guard", "count_pick_top": "e@3", "same_top": 1}
    assert_row(got, want)


def test_guard2_h():
    got = picks(lattice([m for m in GUARD_H if m[0] != "e"]))
    want = {"pick_top": "t@50", "B": 300, "B0": 300, "B1": 300} | ALL_PICK
    want |= {"J": 1.388586290894, "count_pick_top": "t@50", "same_top": 1}
    assert_row(got, want)


def test_badearly_equals_guard_h():
    plain, bad = picks(lattice(GUARD_H)), picks(lattice(GUARD_H + [("n", 2, "n")]))
    assert bad["n_unusable"] == 1
    assert {k: v for k, v in bad.items() if k != "n_unusable"} == {
        k: v for k, v in plain.items() if k != "n_unusable"
    }


def test_merge_is_not_at_rate():
    members = [("d_disputed", 10, "d_disputed"), ("d_merged", 10, "d_merged")]
    got = picks(lattice(members + fillers(10)))
    want = {"n_not_eligible": 1, "top": "d_disputed@10", "pick_top": "d_disputed@10"}
    assert_row(got, want | {"n_trained": 13})


def test_early():
    members = [("x", 0, "cap"), ("t", 50, "b_deep")] + fillers(70)
    got = picks(lattice(members))
    want = {"n_trained": 13, "top": "t@50", "pick_top": "t@50"}
    want |= {"B": 1000, "B1": 1000, "B0": INF, "J": 4.605170185988}
    want |= {"top_a": "guard", "top_b": "pick", "top_c": "pick", "top_d": "guard"}
    assert_row(got, want | {"top_e": "guard"})


def test_empty():
    frames = lattice([("u", -1, "dull"), ("v", 0, "b_deep")])
    got = picks(frames)
    want = {"n_trained": 0, "top": None, "pick_top": None, "abstain_top": 0}
    want |= {f"top_{g}": "empty" for g in "abcde"} | {"ell": None, "same_top": None}
    assert_row(got, want | {"pick_ceiling": None, "given_up_top": None})
    assert frames["condition"].iloc[0]["cell"] == CELL
    assert all(empty(v) for v in frames["condition"].iloc[0].drop("cell"))
    assert frames["beside"]["role"].tolist() == ["pick", "top", "ell"]
    assert frames["beside"]["candidate"].map(empty).all()


def test_ten_abstains_and_eleven_picks():
    names = [f"r{i:02d}" for i in range(1, 10)]
    ten = [("r00", 10, "b_deep")] + fillers(10, names=names)
    got = picks(lattice(ten))
    want = {"n_trained": 10, "top": "r00@10", "pick_top": None, "abstain_top": 1}
    assert_row(got, want | {f"top_{g}": "abstain" for g in "abcde"})
    eleven = picks(lattice(ten + [("r10", 10, "dull")]))
    assert_row(eleven, {"pick_top": "r00@10", "abstain_top": 0})


def test_exact_ties():
    names = [f"r{i:02d}" for i in range(30)]
    got = picks(lattice([(n, 10, "b_deep") for n in names]))
    assert_row(got, {"top": "r00@10", "pick_top": None, "abstain_top": 1})
    two = [("r00", 20, "b_deep"), ("r01", 10, "b_deep")]
    two += [(n, 10, "d_disputed") for n in names[2:]]
    got = picks(lattice(two))
    assert_row(got, {"pick_top": "r01@10", "near_size": 2, "top_a": "pick"})


def test_tenths_of_fab_a3():
    runs = ["full", "edge18", "under17", "one_side"]
    members = [(r, 10, r) for r in runs] + fillers(10, "tenths_filler")
    frames = lattice(members)
    want = {"n_trained": 14, "n_not_eligible": 2, "top": "edge18@10"}
    assert_row(picks(frames), want | {"pick_top": "edge18@10", "abstain_top": 0})
    eligible = frames["candidates"].set_index("run")["elig_tenths"]
    assert eligible[runs].tolist() == [1, 1, 0, 0]


def test_near_top_of_two_g_tilt():
    x, y, v = values("x"), values("y"), TILT["v"]
    s1, s2, p = ARRAYS["x"]
    # FAB_A2.md's premises; the cuts are the 10th largest of s1 and the 54th of s2.
    assert v < np.sort(s2)[-54]
    assert np.sort(p)[9] <= min(np.sort(s1)[-10], np.sort(s2)[-54])
    _, k2, _ = a8.pair_counts(s1, s2, np.array([v]))
    assert not a8.flagged(k2, len(s2) // 2).any()
    gap = round(x["I"], 9) - round(y["I"], 9)
    assert 0 < gap <= 2 * round(x["spread"], 9)
    assert (y["L"], y["L2"], y["C2"]) == (x["L"] + 10, x["L2"], x["C2"])
    assert y["ND"] == x["ND"] + 10 and y["Cbar2"] == x["Cbar2"]
    names = [f"z{i:02d}" for i in range(19)]
    members = [("x", 20, "x"), ("y", 20, "y")] + fillers(20, "tilt_filler", names)
    want = {"n_trained": 21, "top": "x@20", "near_size": 2, "pick_top": "x@20"}
    want |= {"abstain_top": 0, "count_top": "y@20", "count_pick_top": "y@20"}
    assert_row(picks(lattice(members)), want | {"count_near_size": 2, "same_top": 1})


def test_beside_rows():
    frames = lattice(SEP_H)
    beside = frames["beside"].set_index("role")
    assert beside.loc["pick", "candidate"] == "b_deep@10"
    assert beside.loc["ell", "candidate"] == "d_disputed@10"
    deep = values("b_deep")
    row = beside.loc["pick"].to_dict()
    want = {"I": deep["I"], "I0_minus_I": deep["I0"] - deep["I"], "LnP": 0.6}
    want |= {"C2nP": 0.6, "NDnP": 0.0, "N11k": 1.0, "N22k": 1.0}
    assert_row(row, want | {"r2a": 100 * deep["N21"] / 2000})


def test_candidates_table_columns():
    frames = lattice(SEP_H)
    assert tuple(frames["candidates"].columns) == COLUMNS["candidates"]
    assert tuple(frames["picks"].columns) == COLUMNS["picks"]
    assert frames["kinds"].empty and tuple(frames["kinds"].columns) == COLUMNS["kinds"]
    runs = frames["candidates"]["run"].tolist()
    assert runs == sorted(runs)
    # Integers stay integers in the object frame, so that a table writes them so.
    first = frames["candidates"].iloc[0]
    assert isinstance(first["L"], (int, np.integer)) and isinstance(first["I"], float)


def test_kinds_table():
    s1, s2, p, kinds = kinds4()
    main = a8.candidate(s1, s2, p, kinds)
    filler = a8.candidate(s1, s2, points((400, LOW)), kinds)
    cells = {f"{k}_{CELL}": v for k, v in main.items()}
    rows = [{"run": "k", "epoch": 10} | cells]
    cells = {f"{k}_{CELL}": v for k, v in filler.items()}
    rows += [{"run": f"z{i:02d}", "epoch": 10} | cells for i in range(10)]
    frames = tables(pd.DataFrame(rows), [CELL], 4)
    kinds_table = frames["kinds"]
    assert kinds_table["role"].tolist() == ["pick"] * 4 + ["ell"] * 4
    assert kinds_table["kind"].tolist() == [1, 2, 3, 4] * 2
    pick_rows = kinds_table[kinds_table["role"] == "pick"]
    assert pick_rows["L_k"].tolist() == [100, 0, 100, 0]
    assert pick_rows["n_k"].tolist() == [100] * 4
    assert abs(sum(pick_rows["I_k"]) - main["I"]) <= TOL
    assert (pick_rows["I"] == main["I"]).all()


@pytest.mark.parametrize("family", list(COND_LATTICE))
def test_condition_on_ell(family):
    marker, k1, k2, level_1, interval = COND_LATTICE[family]
    row = lattice([("r", 10, family)])["condition"].iloc[0].to_dict()
    if family == "no_threshold":
        # N22 = 67 < 0.9 k(n2): the detector is not at rate, and the cell has no ell.
        assert all(empty(v) for k, v in row.items() if k != "cell")
        return
    want = {"ell": "r@10", "k1": k1, "k2": k2, "L": level_1, "marker": marker}
    assert_row(row, want | {"pick": None, "pick_meets_bound": None})
    if interval is None:
        assert empty(row["h"]) and empty(row["bound"])
        return
    assert interval[0] <= row["h"] <= interval[1]
    want_bound = row["h"] * level_1 / row["n_p"] - BOUND_CONST
    assert abs(row["bound"] - want_bound) <= TOL
    assert row["h"] == factor(values(family)["r_hat"])


def test_pick_meets_bound():
    members = [("r", 10, "edge_read")] + fillers(10, "edge_filler")
    frames = lattice(members)
    row = frames["condition"].iloc[0].to_dict()
    assert row["ell"] == "r@10" and row["pick"] == "r@10"
    assert row["pick_Cbar2"] == 1.0 and row["pick_meets_bound"] == 1
    assert abs(row["bound"] - (row["h"] - BOUND_CONST)) <= TOL
