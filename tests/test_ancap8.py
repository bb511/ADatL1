import math

import numpy as np
import pytest

from src.callbacks import ancap8 as a8
from src.evaluation.components.ancap8 import BOUND_CONST, factor

# The planted answers of the fabricated families of
# scratchpad/adl1t/theory/theory_xxiv/step4/FAB.md, FAB_A1.md and FAB_A2.md,
# theory_xxvii/step4/FAB_A3.md, theory_xxxiii/step5/FAB_R4.md,
# theory_xxxi/step4/cond/FAB_COND.md and, for the condition's markers,
# test_bed/v8_repo/agents/markers/MARKERS.txt; floats are compared within 1e-9.
TOL = 1e-9
LOG_A, LOG_1A = math.log(1 / 0.01), math.log(1 / 0.99)
BASE, TOP, LOW = 3_000_000, 3_100_000, 1
NAN = math.nan


def level(f: float) -> float:
    # Proposition 4 of the design: every flagged point lies deep, the rest far below.
    return f * LOG_A + (1 - f) * LOG_1A


def grid(a: float, b: float, n: int) -> np.ndarray:
    return a + (b - a) * (np.arange(n) + 0.5) / n


def regular(n: int, step: int, offset: int = 0) -> np.ndarray:
    return (BASE - (step * np.arange(n) + offset)).astype(np.float64)


def points(*blocks: tuple[int, float]) -> np.ndarray:
    return np.concatenate([np.full(n, float(v)) for n, v in blocks])


def tied(x: np.ndarray, a: int, b: int) -> np.ndarray:
    # FAB_A3.md: the a-th to b-th largest values set to the b-th largest.
    out = np.sort(x)
    out[len(x) - b : len(x) - a + 1] = out[len(x) - b]
    return out


def same(got: float, want) -> bool:
    if want is None or (isinstance(want, float) and math.isnan(want)):
        return math.isnan(got)
    return abs(got - want) <= TOL


def assert_values(got: dict, want: dict) -> None:
    bad = {k: (got[k], v) for k, v in want.items() if not same(got[k], v)}
    assert not bad


S1 = grid(0, 1, 2000)
DEEP = S1 + 0.0001
FAMILY_P = {
    "const": (np.ones(2000), np.ones(2000), np.ones(1000)),
    "cap": (S1, S1, points((1000, 5.0))),
    "dispute": (S1, grid(10, 11, 2000), points((1000, 5.0))),
    "b_deep": (S1, DEEP, points((600, 5.0), (400, 0.2))),
    "a_shallow": (S1, DEEP, points((600, 0.9926), (400, 0.2))),
    "d_disputed": (
        S1,
        grid(0.5, 1.5, 2000),
        points((650, 5.0), (300, 1.2), (50, 0.2)),
    ),
    "dull": (S1, DEEP, points((1000, 0.2))),
}


def merged(x: np.ndarray) -> np.ndarray:
    return np.where((x >= 0.9) & (x <= 1.5), 1.2, x)


FAMILY_P["d_merged"] = tuple(merged(x) for x in FAMILY_P["d_disputed"])


@pytest.fixture(scope="module")
def fab() -> dict[str, dict]:
    return {name: a8.candidate(*arrays) for name, arrays in FAMILY_P.items()}


def test_bounds_of_fab(fab):
    assert_values(fab["const"], {"I": LOG_1A, "L": 0, "C2": 0, "N11": 0})
    assert_values(fab["cap"], {"I": LOG_A, "L": 1000, "C2": 1000})
    assert round(fab["cap"]["spread"], 9) == 0
    want = {"I": 0.0, "beta_star": 0.0, "L": 1000, "C2": 0, "ND": 1000}
    assert_values(fab["dispute"], want)


def test_separation_of_fab(fab):
    deep = fab["b_deep"]
    assert_values(deep, {"L": 600, "C2": 600, "I": 2.767122245934})
    assert round(deep["spread"], 9) == 0
    assert_values(fab["a_shallow"], {"L": 600, "C2": 600})
    assert fab["a_shallow"]["I"] < deep["I"] - 0.5
    assert_values(fab["d_disputed"], {"L": 950, "C2": 650, "ND": 300})
    assert fab["d_disputed"]["I"] < deep["I"] - 0.2


def test_dull_and_merged_of_fab_a1(fab):
    want = {"N11": 19, "N22": 19, "L": 0, "C2": 0, "I": 0.010050335853501}
    assert_values(fab["dull"], want)
    want = {"L": 650, "L2": 650, "C2": 650, "ND": 0, "N11": 0, "N22": 0}
    assert_values(fab["d_merged"], want | {"I": 2.996878238441})


def test_odd_sizes_of_fab_a1():
    s1 = grid(0, 1, 2001)
    got = a8.candidate(s1, s1 + 0.0001, points((1000, 5.0)))
    assert_values(got, {"I": LOG_A, "I0": LOG_A, "N11": 19, "N22": 19, "k1": 19})


def test_ties_of_fab():
    s1 = np.concatenate([grid(0, 0.9, 48800), np.ones(1200)])
    s2 = np.concatenate([grid(0, 0.9, 49500), np.ones(500)])
    p = points((4000, 2.0), (3000, 1.0), (3000, 0.5))
    got = a8.candidate(s1, s2, p)
    want = {"L": 4000, "L2": 4000, "C2": 4000, "N11": 0, "N21": 0, "N12": 0}
    assert_values(got, want)


# FAB_A3.md, the lattice `tenths`: N11, N22, L = C2 and the number of points at 5.0.
TENTHS = {
    "full": (None, None, 20, 20, 500),
    "edge18": ((19, 21), (19, 21), 18, 18, 600),
    "under17": ((18, 21), (18, 21), 17, 17, 700),
    "one_side": (None, (18, 21), 20, 17, 800),
}


@pytest.mark.parametrize("run", list(TENTHS))
def test_tenths_of_fab_a3(run):
    tie_1, tie_2, n11, n22, deep = TENTHS[run]
    s1, s2 = grid(0, 1, 2050), grid(0, 1, 2050) + 0.0001
    s1 = s1 if tie_1 is None else tied(s1, *tie_1)
    s2 = s2 if tie_2 is None else tied(s2, *tie_2)
    got = a8.candidate(s1, s2, points((deep, 5.0), (1000 - deep, 0.2)))
    want = {"N11": n11, "N22": n22, "L": deep, "C2": deep, "k1": 20, "k2": 20}
    assert_values(got, want | {"I": level(deep / 1000)})


def test_curve_of_cap_at_beta_one():
    k1, k2, _ = a8.pair_counts(S1, S1, points((1000, 5.0)))
    curve = a8.mean_curve(k1, k2, 1000, 1000)
    m0 = math.log(0.01 * 1001 / 0.5)
    p0 = 1 / (1 + math.exp(-(m0 + math.log(0.01 / 0.99))))
    derived = math.log(1 + (p0 - 0.01) ** 2 / (0.01 * 0.99))
    assert abs(m0 - 2.996731774) <= TOL and abs(derived - 1.260795912) <= TOL
    assert abs(curve[a8.BETAS == 1.0][0] - derived) <= TOL


def kinds4() -> tuple:
    p = points((100, TOP), (100, LOW), (100, BASE - 150), (100, BASE - 10_000))
    kinds = np.repeat(np.arange(1, 5), 100)
    return regular(2000, 10), regular(2000, 5, 2), p, kinds


KINDS4_PARTS = (
    (100, 100, 1.0, 1.150100444716),
    (0, 0, 0.0, 0.002512583558),
    (100, 0, 0.0078125, -0.003845151942),
    (0, 0, 0.0, 0.002512581387),
)
KINDS4_CANDIDATE = {"I": 1.151280457719, "beta_star": 0.01 * 10 ** (102 / 40)}
KINDS4_CANDIDATE |= {"Cbar2": 0.251953125, "L": 200, "L2": 100, "C2": 100, "ND": 100}


def test_parts_of_kinds4():
    got = a8.candidate(*kinds4())
    assert_values(got, KINDS4_CANDIDATE)
    for kind, (level_k, c2_k, cbar2_k, i_k) in enumerate(KINDS4_PARTS, 1):
        want = {f"n_k{kind}": 100, f"L_k{kind}": level_k, f"C2_k{kind}": c2_k}
        want |= {f"Cbar2_k{kind}": cbar2_k, f"I_k{kind}": i_k}
        assert_values(got, want | {f"share_k{kind}": i_k / got["I"]})
    assert got["I_k3"] < 0


def test_parts_of_one_kind():
    s1, s2, p, _ = kinds4()
    got = a8.candidate(s1, s2, p, np.ones(400, dtype=np.int64))
    want = {"n_k1": 400, "L_k1": 200, "C2_k1": 100, "Cbar2_k1": 0.251953125}
    assert_values(got, want | {"I_k1": KINDS4_CANDIDATE["I"], "share_k1": 1.0})
    assert "n_k2" not in got


def test_an_empty_kind_keeps_its_keys():
    s1, s2, p, _ = kinds4()
    got = a8.ancap8(s1, {"c": s2}, [p[:200], p[200:], np.empty(0)], kinds=True)
    want = {"usable_c": 1, "n_k3_c": 0, "L_k3_c": 0, "C2_k3_c": 0, "I_k3_c": 0.0}
    assert_values(got, want | {"Cbar2_k3_c": NAN, "share_k3_c": 0.0})
    assert "n_k4_c" not in got
    one = a8.candidate(s1, s2, p, np.ones(400, dtype=np.int64), n_kinds=2)
    assert_values(one, {"n_k1": 400, "n_k2": 0, "I_k1": KINDS4_CANDIDATE["I"]})


# The first code's output on FAB_R4.md's lhc_size (a_p/out/fab_r4/lhc_size.json),
# without its seconds.
LHC_SIZE = {
    "I": 2.705560177557403,
    "beta_star": 84.13951416451948,
    "spread": 0.00514388910987304,
    "Cbar2": 0.5968619140625,
    "spread_c": 0.0007391915139160419,
    "L": 24439,
    "L2": 23936,
    "C2": 23936,
    "ND": 503,
    "N21": 534,
    "N12": 453,
    "N11": 499,
    "N22": 499,
}
LHC_SIZE_PARTS = (
    (4483, 4339, 0.43195234375, 0.4859303622709109, 0.1796043445278722),
    (6428, 6305, 0.62916796875, 0.7138142922393135, 0.2638323472382527),
    (8058, 7969, 0.79601796875, 0.9086356294053126, 0.3358401106515526),
    (5470, 5323, 0.530309375, 0.5971798936419536, 0.22072319758235479),
)


def test_lhc_size_against_the_first_code():
    rng = np.random.default_rng(20261112)
    s1 = rng.standard_normal(50_000)
    s2 = 1.02 * rng.standard_normal(50_000) + 0.01
    p = np.concatenate([rng.normal(m, 0.8, 10_000) for m in (2.2, 2.6, 3.0, 2.4)])
    s1, s2, p = (x.astype(np.float32) for x in (s1, s2, p))
    kinds = np.repeat(np.arange(1, 5), 10_000)
    got = a8.candidate(s1, s2, p, kinds, {"n1": 50_000, "n2": 50_000, "n_p": 40_000})
    assert_values(got, LHC_SIZE)
    for kind, values in enumerate(LHC_SIZE_PARTS, 1):
        names = [f"{k}{kind}" for k in ("L_k", "C2_k", "Cbar2_k", "I_k", "share_k")]
        assert_values(got, dict(zip(names, values)) | {f"n_k{kind}": 10_000})


def test_unusable_candidates():
    s1, s2, p = FAMILY_P["b_deep"]
    with_nan = s1.copy()
    with_nan[3] = math.nan
    got = a8.candidate(with_nan, s2, p, np.ones(1000, dtype=np.int64))
    assert got["usable"] == 0
    assert (got["n1"], got["n2"], got["n_p"]) == (2000, 2000, 1000)
    assert all(math.isnan(v) for k, v in got.items() if k not in a8.STATISTIC[:4])
    assert "I_k1" in got
    sized = a8.candidate(s1, s2, p, sizes={"n1": 2000, "n_p": 999})
    assert sized["usable"] == 0 and sized["n1"] == 2000 and math.isnan(sized["n_p"])
    absent = a8.candidate(s1, None, p)
    assert absent["usable"] == 0 and math.isnan(absent["n2"]) and absent["n_p"] == 1000
    assert a8.candidate(s1, s2, np.empty(0))["usable"] == 0


def test_keys_and_floats(fab):
    got = fab["b_deep"]
    assert tuple(got) == a8.keys(0)
    assert all(type(v) is float for v in got.values())
    assert got["usable"] == 1.0 and got["n_p"] == 1000.0


def test_pairs_are_cached_and_read_only():
    first, second = a8._permutations(2000, 2000)
    assert a8._permutations(2000, 2000)[0] is first
    rng = np.random.default_rng([a8.SEED, 3])
    assert np.array_equal(first[3], rng.permutation(2000))
    assert np.array_equal(second[3], rng.permutation(2000)[:1000])
    assert not first.flags.writeable


# FAB_COND.md's families and the three of MARKERS.txt.
def _grow() -> np.ndarray:
    head = BASE + 10 * np.arange(1, 61)
    return np.concatenate([head, regular(49_940, 30, 15)]).astype(np.float64)


def _ties() -> np.ndarray:
    tail = BASE - (14_400 + 30 * np.arange(1, 49_421))
    block = np.full(100, BASE - 14_400)
    return np.concatenate([regular(480, 30), block, tail]).astype(np.float64)


def _floor() -> np.ndarray:
    j = np.arange(1, 501)
    top = BASE - (14_970 - 15 * (500 - j))
    k = np.arange(501, 50_001)
    return np.concatenate([top, BASE - (14_970 + 30 * (k - 500))]).astype(np.float64)


def _tied_above_cut() -> np.ndarray:
    tail = BASE - (2_010 + 30 * np.arange(1, 49_401))
    block = np.full(533, BASE - 2_010)
    return np.concatenate([regular(67, 30), block, tail]).astype(np.float64)


def cond_families() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    a, b = regular(50_000, 30), regular(50_000, 20, 5)
    mixed = points((600, TOP), (200, BASE - 12_000), (200, LOW))
    pushed = points((300, TOP), (200, BASE - 1_500), (300, BASE - 2_010), (200, LOW))
    # fmt: off
    return {
        "const15": (a, b, mixed),
        "const15_rev": (b, a, mixed),
        "grow": (a, _grow(), points((500, TOP), (500, BASE - 14_000))),
        "ties": (_ties(), regular(50_000, 30, 15), points((300, TOP), (400, BASE - 14_400), (300, LOW))),
        "floor": (a, _floor(), points((1000, TOP))),
        "unequal": (regular(50_000, 40), regular(40_000, 50, 5), points((700, TOP), (300, LOW))),
        "unflagged": (a, b, points((1000, LOW))),
        "small": (regular(1000, 30), regular(1000, 30, 5), points((60, TOP), (40, LOW))),
        "edge_read": (regular(19_000, 30), regular(19_000, 30, 5), points((100, TOP))),
        "edge_unread": (regular(18_800, 30), regular(18_800, 30, 5), points((100, TOP))),
        "no_threshold": (regular(50_000, 2), _tied_above_cut(), pushed),
        "tiny_s1": (regular(49, 30), regular(50_000, 30, 5), points((60, TOP), (40, LOW))),
        "tiny_s2": (regular(50_000, 30), regular(49, 30, 5), points((60, TOP), (40, LOW))),
    }
    # fmt: on


COND_KEYS = ("looser", "read", "n_thresholds", "ratio_max", "c_a_at_max", "c_b_at_max")
COND_KEYS += ("r_at_cut", "flagged_a", "flagged_both", "rows_a_flagged")
COND_KEYS += ("rows_a_flagged_both", "consequence_holds")
U = None
# FAB_COND.md's second table and MARKERS.txt, section 4, in COND_KEYS' order. Beside
# them, in COND_LATTICE: the marker, the condition's k1, k2 and L, the interval of h.
# fmt: off
COND = {
    "const15": (1, 1, 432, 1493 / 997, 498, 746, 1499 / 1001, 800, 600, 499, 333, 1),
    "const15_rev": (2, 1, 432, 1493 / 997, 498, 746, 1499 / 1001, 800, 600, 499, 333, 1),
    "grow": (1, 1, 432, 255 / 137, 68, 127, 1119 / 1001, 1000, 500, 499, 440, 0),
    "ties": (2, 1, 432, 129 / 107, 481, 580, 109 / 91, 700, 300, 499, 480, 0),
    "floor": (1, 1, 432, 997 / 999, 499, 498, 1, 1000, 1000, 499, 499, 1),
    "unequal": (1, 1, 432, 1883371 / 1880047, 70, 56, 5721543 / 5720143, 700, 700, 499, 499, 1),
    "unflagged": (1, 1, 432, 1493 / 997, 498, 746, 1499 / 1001, 0, 0, 499, 333, U),
    "small": (2, 0, U, U, U, U, U, 60, 60, 9, 9, 1),
    "edge_read": (2, 1, 164, 1, 189, 189, 1, 100, 100, 189, 189, 1),
    "edge_unread": (2, 0, U, U, U, U, U, 100, 100, 187, 187, 1),
    "no_threshold": (2, 1, 0, U, U, U, U, 500, 300, 67, 34, 1),
    "tiny_s1": (U, 0, U, U, U, U, U, U, U, U, U, U),
    "tiny_s2": (U, 0, U, U, U, U, U, U, U, U, U, U),
}
COND_LATTICE = {
    "const15": (0, 500, 500, 800, (0.545, 0.575)),
    "const15_rev": (0, 500, 500, 600, (0.545, 0.575)),
    "grow": (0, 500, 500, 1000, (0.38, 0.47)),
    "ties": (0, 500, 500, 300, (0.74, 0.81)),
    "floor": (0, 500, 500, 1000, (0.98, 1.003)),
    "unequal": (0, 500, 400, 700, (0.97, 1.003)),
    "unflagged": (0, 500, 500, 0, (0.545, 0.575)),
    "small": (1, 10, 10, 60, None),
    "edge_read": (0, 190, 190, 100, (0.98, 1.003)),
    "edge_unread": (1, 188, 188, 100, None),
    "no_threshold": (2, 500, 500, 300, None),
    "tiny_s1": (1, 0, 500, None, None),
    "tiny_s2": (1, 500, 0, 60, None),
}
# fmt: on
# The first code's h on reading 7's float64 arrays (a_p/out/fab_r4/<family>.json).
FIRST_CODE_H = {
    "const15": 0.5578933224094419,
    "const15_rev": 0.5578933224094419,
    "grow": 0.40383884035383755,
    "ties": 0.7658123910015097,
    "floor": 1.0017892651911613,
    "unequal": 0.9996622897731907,
    "unflagged": 0.5578933224094419,
    "edge_read": 1.0017892651911613,
}
# Cuts and the depth beside the planted counts, from MARKERS.txt and the first code.
COND_CUTS = {
    "const15": (2985030, 2990015, 67.16899497113872, 0.003024508820301334),
    "no_threshold": (2999002, 2997990, 67.16899497113872, U),
    "tiny_s1": (U, 2985025, U, U),
    "tiny_s2": (2985030, U, U, U),
    "small": (2999730, 2999725, 0.8547061851984932, U),
}


@pytest.fixture(scope="module")
def conditions() -> dict[str, dict]:
    return {name: a8.condition(*arrays) for name, arrays in cond_families().items()}


@pytest.mark.parametrize("family", list(COND))
def test_condition_of_fab_cond_and_markers(conditions, family):
    got = conditions[family]
    want = dict(zip(COND_KEYS, COND[family])) | {"marker": COND_LATTICE[family][0]}
    ratio = want["ratio_max"]
    want["r_hat"] = U if ratio is None else max(1.0, ratio)
    if ratio is None:
        want["depth_of_max"] = U
    assert_values(got, want)
    if family in COND_CUTS:
        names = ("cut1", "cut2", "rows_at_depth", "depth_of_max")
        assert_values(got, dict(zip(names, COND_CUTS[family])))


def test_share_between_of_the_markers(conditions):
    assert_values(conditions["no_threshold"], {"share_between": 0.4})
    assert_values(conditions["const15"], {"share_between": 0.25})
    for family in ("tiny_s1", "tiny_s2", "unflagged"):
        assert math.isnan(conditions[family]["share_between"])


@pytest.mark.parametrize("family", list(FIRST_CODE_H))
def test_factor_against_the_first_code(conditions, family):
    h = factor(conditions[family]["r_hat"])
    low, high = COND_LATTICE[family][4]
    assert low <= h <= high
    # A line-by-line port of implementation A: equal to the last bit on olgpu.
    assert abs(h - FIRST_CODE_H[family]) <= 1e-12


def test_bound_constant():
    # The first code's bound on `unflagged`, whose L is 0.
    assert abs(BOUND_CONST - 0.0021871760000355886) <= 1e-15
