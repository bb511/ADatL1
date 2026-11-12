"""ANCAP v8's lattice step: from every candidate's values, the pick of each cell under
the design's reading and what is reported beside it.

The design is ``scratchpad/adl1t/theory/theory_xxxi/step4/V8.md`` (sections 2 and 3),
which keeps the selection of ``theory_xxiv/step4/V7_r3.md`` (section 2) under the user's
readings U2 ``tenths`` and U4 ``top`` (``theory_xxvii/step4/V7_r3_CHOICES.md``). The
tables are those of the frozen first code of step 5
(``theory_xxxiii/step5/a_p/code/run5a_r4.py``, with ``ancap7r3.py`` and ``cond.py``),
whose columns are declared in ``theory_xxvii/step5/DECLARATION_5A.md`` and
``theory_xxxiii/step5/DECLARATION_5A.md`` ("Tables"), restricted to that reading. The
values of a candidate are those of ``src.callbacks.ancap8.candidate``, whose docstring
defines them.

Per cell, among its candidates (one per run and epoch, epoch -1 the untrained weights)::

    eligible   usable, 10 N11 >= 9 k1 and 10 N22 >= 9 k2              (U2 tenths)
    trained    eligible with epoch >= 1
    top        the trained candidate of largest I; ties go to the larger L, then the
               earlier epoch, then the run name
    near-top   the trained candidates with I >= round(I(top) - 2 spread(top), 9);
               n_edge counts those within 1e-9 of that threshold
    pick       the top, unless ten times the number of trained candidates whose I
               equals the top's is at least the number of trained ones: then the cell
               abstains (U4 top); a cell without a trained candidate is empty
    B, B0, B1  the largest L over the eligible candidates of epochs -1 to 5, of epoch
               -1, of epochs -1 and 0; J the largest I of epochs -1 to 5; -inf for none
    guard      the pick stands under a when L > B, b when L > B0, c always, d when
               L > B1, e when I > J; its status is pick, guard, abstain or empty
    ell        the trained candidate of largest L; ties go to the earlier epoch, then
               the run name
    count      the same rules with Cbar2 and spread_c in place of I and spread
    same       the count's pick lies in the capacity's near-top set and the capacity's
               pick in the count's; empty where either has no pick
    given up   (L(ell) - L(pick)) / n_P
    c2_pick    the trained candidate of largest C2, then the order of ell
    ceiling    phi(Cbar2) = Cbar2 log(1/alpha) + (1 - Cbar2) log(1/(1 - alpha)) of the
               pick, and its charge phi(Cbar2) - I (V8.md, section 3)

I, I0, Cbar2 and the spreads are rounded with round(x, 9) before every rank and
comparison; counts are compared as integers and run names as Python strings.

For ell, the condition's factor and bound (V8.md, section 2), where its r_hat is read::

    h     = C(r_hat, M) / log((1 - alpha) / alpha)
    C     = max(0, max over the grid of beta of E_e k_beta(e, min(e, M) - log r_hat))
    bound = h L(ell) / n_P - log(1/(1 - alpha)) / log((1 - alpha) / alpha)

with e exponential of mean 1, M = 2 nats and k_beta(m, m') = log(p p'/alpha + (1 - p)
(1 - p')/(1 - alpha)), p = sigmoid(beta m + logit alpha). The expectation is computed by
the condition's implementation A (``theory_xxxi/step4/cond/impl_a/cond.py``, frozen in
step 5 as ``a_p/code/cond.py``): adaptive Simpson rules with Richardson's correction
on pieces cut at e = 0, log r_hat, M and 1/beta-wide neighbourhoods of the points where
p or p' crosses 1/2, to e = 64. ``pick_meets_bound`` is round(Cbar2(pick), 9) >=
round(bound, 9). The condition's k_i are its cut ranks (n_i + 50) // 100, and its L is
ell's, empty where S_1 has no cut.

``tables`` returns pandas frames of dtype object holding Python ints, floats and
strings, with None or NaN where a field is empty, so that ``DataFrame.to_csv`` writes
them as the first code writes its tables: a flag ``1`` or ``0``, a candidate
``run@epoch``, an empty maximum ``-inf``.
"""

import functools
import math
from collections.abc import Callable

import pandas as pd

from src.callbacks.ancap8 import ALPHA, DIGITS, M_CAP, PARTS, TIE, keys

# The guard reads the epochs -1 to 5, and five readings of it (V7_r3.md, section 2).
GUARD_EPOCHS = (-1, 5)
GUARDS = "abcde"
# A cell abstains when ten times its candidates at the top's value reach its trained
# ones (V7_r3_CHOICES.md, U4).
ABSTENTION = 10
LOG_ODDS = math.log((1 - ALPHA) / ALPHA)
# The exact constant of the bound, not its rounded 0.0022 (FAB_COND.md, reading 8).
BOUND_CONST = math.log(1 / (1 - ALPHA)) / LOG_ODDS
# The grid of beta in Python floats, as implementation A computes it, so that h is that
# code's to the last bit; numpy's power may differ from Python's in the last bit.
FACTOR_BETAS = [0.0] + [0.01 * 10 ** (j / 40) for j in range(241)]
# exp(-64) < 1e-27 and |k| stays below 5 where e > M, so the dropped tail is negligible.
E_MAX = 64.0
# Offsets, in units of 1/beta, of the breakpoints around each point where p or p'
# crosses 1/2: at beta = 1e4 the integrand changes within a few 1e-4 of these points.
WIDTHS = (-30, -10, -3, -1, 0, 1, 3, 10, 30)
QUADRATURE_TOL = 1e-10
MAX_DEPTH = 50
INTEGER_KEYS = {"usable", "n1", "n2", "n_p", "k1", "k2", "L", "L2", "C2", "ND"}
INTEGER_KEYS |= {"N21", "N12", "N11", "N22", "looser", "read", "n_thresholds"}
INTEGER_KEYS |= {"c_a_at_max", "c_b_at_max", "flagged_a", "flagged_both", "marker"}
INTEGER_KEYS |= {"rows_a_flagged", "rows_a_flagged_both", "consequence_holds"}
INTEGER_PARTS = ("n_k", "L_k", "C2_k")
STATISTIC_COLUMNS = ("I", "beta_star", "spread", "I0", "beta_star0", "spread0", "L")
STATISTIC_COLUMNS += ("L2", "C2", "ND", "N21", "N12", "N11", "N22", "k1", "k2")
STATISTIC_COLUMNS += ("Cbar2", "spread_c")
RATIO_COLUMNS = ("cut1", "cut2", "looser", "rows_at_depth", "read", "n_thresholds")
RATIO_COLUMNS += ("ratio_max", "r_hat", "r_at_cut", "c_a_at_max", "c_b_at_max")
RATIO_COLUMNS += ("depth_of_max",)
FLAG_COLUMNS = ("flagged_a", "flagged_both", "share_between", "rows_a_flagged")
FLAG_COLUMNS += ("rows_a_flagged_both", "consequence_holds")
COLUMNS = {
    "candidates": (
        ("cell", "run", "epoch", "usable", "trained", "n1", "n2", "nP")
        + STATISTIC_COLUMNS
        + ("elig_tenths",)
    ),
    "picks": (
        ("cell", "n_trained", "n_not_eligible", "n_unusable", "top", "near_size")
        + ("n_edge", "pick_top", "abstain_top", "B", "B0", "B1", "J")
        + tuple(f"top_{g}" for g in GUARDS)
        + ("early_top", "ell", "c2_pick", "count_top", "count_near_size")
        + ("count_pick_top", "same_top", "given_up_top", "pick_ceiling", "pick_charge")
    ),
    "beside": (
        ("cell", "role", "candidate", "I", "spread", "beta_star", "I0", "I0_minus_I")
        + ("LnP", "C2nP", "NDnP", "r2a", "r1a", "N11k", "N22k")
    ),
    "kinds": ("cell", "role", "run", "epoch", "kind", *PARTS, "beta_star", "I"),
    "condition": (
        ("cell", "ell", "n1", "n2", "n_p", "k1", "k2", *RATIO_COLUMNS, "h", "bound")
        + ("L", *FLAG_COLUMNS, "pick", "pick_Cbar2", "pick_meets_bound", "marker")
    ),
}


def _r9(x: float) -> float:
    return round(x, DIGITS)


def _name(row: dict | None) -> str | None:
    """Return a candidate as ``run@epoch``, None for no candidate."""
    return None if row is None else f"{row['run']}@{row['epoch']}"


def _value(key: str, x: float | None) -> int | float | None:
    """Return a logged value as a Python int or float, None where it is NaN."""
    if x is None or math.isnan(x):
        return None
    integer = key in INTEGER_KEYS or key.rstrip("0123456789") in INTEGER_PARTS
    return int(x) if integer else float(x)


def _typed(record: dict) -> dict:
    """Return a candidate of a cell with its usability as a bool."""
    row = {k: _value(k, v) for k, v in record.items() if k not in ("run", "epoch")}
    row |= {"run": str(record["run"]), "epoch": int(record["epoch"])}
    return row | {"usable": row["usable"] == 1}


def _cell_candidates(rows: pd.DataFrame, cell: str, n_kinds: int) -> list[dict]:
    """Return the candidates of a cell, ordered by run and epoch, keyed without the
    cell; a column that rows lacks reads as NaN."""
    named = {f"{key}_{cell}": key for key in keys(n_kinds)}
    table = rows.reindex(columns=["run", "epoch", *named]).rename(columns=named)
    found = [_typed(r) for r in table.to_dict("records")]
    return sorted(found, key=lambda r: (r["run"], r["epoch"]))


def is_eligible(row: dict) -> bool:
    """Return whether a candidate is usable and at rate under U2's ``tenths``."""
    if not row["usable"]:
        return False
    return 10 * row["N11"] >= 9 * row["k1"] and 10 * row["N22"] >= 9 * row["k2"]


def _by_level(row: dict) -> tuple:
    return (-row["L"], row["epoch"], row["run"])


def _ranked(key: str):
    def order(row: dict) -> tuple:
        return (-_r9(row[key]),) + _by_level(row)

    return order


def select(trained: list[dict], key: str, spread: str) -> dict:
    """Return the top, the near-top set, n_edge and the pick under U4's ``top``.

    key and spread are ("I", "spread") for the capacity and ("Cbar2", "spread_c") for
    the count. The pick is None where the cell abstains or has no candidate.
    """
    if not trained:
        return {"top": None, "near": [], "n_edge": 0, "pick": None}
    top = min(trained, key=_ranked(key))
    threshold = _r9(_r9(top[key]) - 2 * _r9(top[spread]))
    near = [r for r in trained if _r9(r[key]) >= threshold]
    edge = sum(_r9(abs(_r9(r[key]) - threshold)) <= TIE for r in trained)
    equal = sum(_r9(r[key]) == _r9(top[key]) for r in trained)
    choice = None if ABSTENTION * equal >= len(trained) else top
    return {"top": top, "near": near, "n_edge": edge, "pick": choice}


def _best(rows: list[dict], epochs: tuple[int, int], key: str) -> float:
    """Return the largest key over the rows of epochs, I rounded; -inf for none."""
    values = [_r9(r[key]) if key == "I" else r[key] for r in rows]
    picked = [v for r, v in zip(rows, values) if epochs[0] <= r["epoch"] <= epochs[1]]
    return max(picked, default=-math.inf)


def guard_bars(eligible: list[dict]) -> dict[str, float]:
    """Return B, B0, B1 (L) and J (rounded I) over the eligible rows; -inf if none."""
    low, high = GUARD_EPOCHS
    bars = {"B": _best(eligible, (low, high), "L")}
    bars |= {"B0": _best(eligible, (low, low), "L")}
    bars |= {"B1": _best(eligible, (low, 0), "L")}
    return bars | {"J": _best(eligible, (low, high), "I")}


def guard_status(choice: dict | None, bars: dict, empty: bool) -> dict[str, str]:
    """Return the status of a pick under the five readings of the guard."""
    if choice is None:
        return dict.fromkeys(GUARDS, "empty" if empty else "abstain")
    level, cap = choice["L"], _r9(choice["I"])
    stands = {"a": level > bars["B"], "b": level > bars["B0"], "c": True}
    stands |= {"d": level > bars["B1"], "e": cap > bars["J"]}
    return {k: "pick" if v else "guard" for k, v in stands.items()}


def _same(cap: dict, count: dict) -> int | None:
    """Return 1 where the cell is same, 0 where different, None where either has no
    pick."""
    a, b = cap["pick"], count["pick"]
    if a is None or b is None:
        return None
    return int(any(r is b for r in cap["near"]) and any(r is a for r in count["near"]))


def pick(cands: list[dict]) -> dict:
    """Apply the selection to the candidates of one cell.

    Return the capacity's and the count's ``select``, the guard's bars and statuses,
    same or different, ell, c2_pick and the counts of trained and not eligible ones.
    """
    eligible = [r for r in cands if is_eligible(r)]
    trained = [r for r in eligible if r["epoch"] >= 1]
    cap, count = select(trained, "I", "spread"), select(trained, "Cbar2", "spread_c")
    bars = guard_bars(eligible)
    out = {"n_trained": len(trained), "cap": cap, "count": count, "bars": bars}
    out["n_not_eligible"] = sum(r["usable"] for r in cands) - len(eligible)
    out |= {"status": guard_status(cap["pick"], bars, not trained)}
    out |= {"same": _same(cap, count), "ell": min(trained, key=_by_level, default=None)}
    by_c2 = min(trained, key=lambda r: (-r["C2"],) + _by_level(r), default=None)
    return out | {"c2_pick": by_c2}


def ceiling(cbar2: float) -> float:
    """Return phi(Cbar2), the capacity of a pick whose agreed points all lie deep."""
    return cbar2 * math.log(1 / ALPHA) + (1 - cbar2) * math.log(1 / (1 - ALPHA))


def _log_sigmoid(x: float) -> float:
    """Return log(sigmoid(x)) without overflow."""
    if x >= 0:
        return -math.log1p(math.exp(-x))
    return x - math.log1p(math.exp(x))


def _kernel(beta: float, m: float, m2: float) -> float:
    """Return k_beta(m, m'), its two terms summed in log space: at beta = 1e4 either can
    underflow a double."""
    x, x2 = beta * m - LOG_ODDS, beta * m2 - LOG_ODDS
    both_in = _log_sigmoid(x) + _log_sigmoid(x2) - math.log(ALPHA)
    both_out = _log_sigmoid(-x) + _log_sigmoid(-x2) - math.log1p(-ALPHA)
    hi, lo = max(both_in, both_out), min(both_in, both_out)
    return hi + math.log1p(math.exp(lo - hi))


def _breakpoints(beta: float, log_r: float) -> list[float]:
    """Return the ends of the pieces of the quadrature in e, ascending."""
    points = {0.0, log_r, M_CAP, 1.0, 4.0, 8.0, 16.0, 32.0, E_MAX}
    if beta > 0:
        for centre in (LOG_ODDS / beta, log_r + LOG_ODDS / beta):
            points.update(centre + w / beta for w in WIDTHS)
    return sorted(x for x in points if 0.0 <= x <= E_MAX)


def _refine(
    f: Callable[[float], float],
    a: float,
    b: float,
    fa: float,
    fm: float,
    fb: float,
    whole: float,
    tol: float,
    depth_left: int,
) -> float:
    """Return Simpson's rule on [a, b] halved until its two halves agree within 15 tol
    or depth_left is spent; whole is the rule on [a, b]."""
    m = (a + b) / 2
    flm, frm = f((a + m) / 2), f((m + b) / 2)
    left = (m - a) * (fa + 4 * flm + fm) / 6
    right = (b - m) * (fm + 4 * frm + fb) / 6
    if depth_left == 0 or abs(left + right - whole) <= 15 * tol:
        return left + right + (left + right - whole) / 15
    return _refine(f, a, m, fa, flm, fm, left, tol / 2, depth_left - 1) + _refine(
        f, m, b, fm, frm, fb, right, tol / 2, depth_left - 1
    )


def _simpson(f: Callable[[float], float], a: float, b: float, tol: float) -> float:
    """Integrate f over [a, b] by adaptive Simpson with Richardson's correction."""
    fa, fm, fb = f(a), f((a + b) / 2), f(b)
    whole = (b - a) * (fa + 4 * fm + fb) / 6
    return _refine(f, a, b, fa, fm, fb, whole, tol, MAX_DEPTH)


def _expected_kernel(beta: float, log_r: float) -> float:
    """Return E_e k_beta(e, min(e, M) - log r) for e exponential with mean 1."""

    def integrand(e: float) -> float:
        return _kernel(beta, e, min(e, M_CAP) - log_r) * math.exp(-e)

    pts = _breakpoints(beta, log_r)
    return sum(
        _simpson(integrand, lo, hi, QUADRATURE_TOL) for lo, hi in zip(pts, pts[1:])
    )


@functools.cache
def factor(r_hat: float) -> float:
    """Return h(r_hat, M), unclipped: 1.0018 at r_hat = 1 on this grid."""
    log_r = math.log(r_hat)
    best = max(_expected_kernel(beta, log_r) for beta in FACTOR_BETAS)
    return max(0.0, best) / LOG_ODDS


def bound(h: float, level: int, n_p: int) -> float:
    """Return the population bound on the pick's jointly flagged share."""
    return h * level / n_p - BOUND_CONST


def _candidate_row(r: dict) -> dict:
    """Return a candidate's row of the candidates table; an unusable one keeps its sizes
    alone."""
    out = {"run": r["run"], "epoch": r["epoch"], "usable": int(r["usable"])}
    out["trained"] = int(r["usable"] and r["epoch"] >= 1)
    out |= {"n1": r["n1"], "n2": r["n2"], "nP": r["n_p"]}
    if not r["usable"]:
        return out
    out |= {k: r[k] for k in STATISTIC_COLUMNS}
    return out | {"elig_tenths": int(is_eligible(r))}


def _pick_head(cands: list[dict], found: dict) -> dict:
    """Return the picks table's columns up to early_top."""
    cap, choice, status = found["cap"], found["cap"]["pick"], found["status"]
    out = {"n_trained": found["n_trained"], "n_not_eligible": found["n_not_eligible"]}
    out |= {"n_unusable": sum(not r["usable"] for r in cands), "top": _name(cap["top"])}
    out |= {"near_size": len(cap["near"]), "n_edge": cap["n_edge"]}
    out |= {"pick_top": _name(choice), "abstain_top": int(status["a"] == "abstain")}
    out |= found["bars"] | {f"top_{g}": s for g, s in status.items()}
    early = None if choice is None else int(choice["epoch"] <= GUARD_EPOCHS[1])
    return out | {"early_top": early}


def _pick_row(cands: list[dict], found: dict) -> dict:
    """Return the cell's row of the picks table."""
    choice, count, ell = found["cap"]["pick"], found["count"], found["ell"]
    out = _pick_head(cands, found) | {"ell": _name(ell)}
    out |= {"c2_pick": _name(found["c2_pick"]), "count_top": _name(count["top"])}
    out |= {"count_near_size": len(count["near"])}
    out |= {"count_pick_top": _name(count["pick"]), "same_top": found["same"]}
    if choice is None:
        return out
    out["given_up_top"] = (ell["L"] - choice["L"]) / choice["n_p"]
    phi = ceiling(choice["Cbar2"])
    return out | {"pick_ceiling": phi, "pick_charge": phi - choice["I"]}


def _per(count: int, k: int) -> float | None:
    """Return count / k, None where k is 0: k(n) is 0 below 150 rows, where a
    candidate's own flags have no nominal share."""
    return count / k if k else None


def _beside_row(role: str, r: dict | None) -> dict:
    """Return a role's row of the beside table, the role alone where it is vacant."""
    if r is None:
        return {"role": role}
    out = {"role": role, "candidate": _name(r), "I": r["I"], "spread": r["spread"]}
    out |= {"beta_star": r["beta_star"], "I0": r["I0"], "I0_minus_I": r["I0"] - r["I"]}
    out |= {"LnP": r["L"] / r["n_p"], "C2nP": r["C2"] / r["n_p"]}
    out |= {"NDnP": r["ND"] / r["n_p"], "r2a": 100 * r["N21"] / r["n2"]}
    out |= {"r1a": 100 * r["N12"] / r["n1"], "N11k": _per(r["N11"], r["k1"])}
    return out | {"N22k": _per(r["N22"], r["k2"])}


def _beside_rows(found: dict) -> list[dict]:
    """Return the rows of the pick, the top and ell, in that order."""
    roles = {"pick": found["cap"]["pick"], "top": found["cap"]["top"]}
    roles["ell"] = found["ell"]
    return [_beside_row(role, r) for role, r in roles.items()]


def _kind_rows(found: dict, n_kinds: int) -> list[dict]:
    """Return the parts of the pick, then of ell, one row per kind; none for a vacant
    role."""
    out = []
    for role, r in (("pick", found["cap"]["pick"]), ("ell", found["ell"])):
        if r is None:
            continue
        head = {"role": role, "run": r["run"], "epoch": r["epoch"]}
        tail = {"beta_star": r["beta_star"], "I": r["I"]}
        for kind in range(1, n_kinds + 1):
            part = {p: r.get(f"{p}{kind}") for p in PARTS}
            out.append(head | {"kind": kind} | part | tail)
    return out


def _factor_fields(ell: dict) -> dict:
    """Return h and the bound of ell, None where its r_hat is not read."""
    if ell["r_hat"] is None:
        return {"h": None, "bound": None}
    h = factor(ell["r_hat"])
    return {"h": h, "bound": bound(h, ell["L"], ell["n_p"])}


def _ell_fields(ell: dict) -> dict:
    """Return ell's fields of the condition table, h and the bound included."""
    out = {"ell": _name(ell)} | {k: ell[k] for k in ("n1", "n2", "n_p", "marker")}
    out |= {"k1": (ell["n1"] + 50) // 100, "k2": (ell["n2"] + 50) // 100}
    out |= {k: ell[k] for k in RATIO_COLUMNS + FLAG_COLUMNS} | _factor_fields(ell)
    return out | {"L": None if ell["cut1"] is None else ell["L"]}


def _pick_fields(choice: dict, bound: float | None) -> dict:
    """Return the pick's fields of the condition table; meeting the bound is empty where
    the bound is."""
    meets = None if bound is None else int(_r9(choice["Cbar2"]) >= _r9(bound))
    out = {"pick": _name(choice), "pick_Cbar2": choice["Cbar2"]}
    return out | {"pick_meets_bound": meets}


def _condition_row(found: dict) -> dict:
    """Return the cell's row of the condition table, empty after the cell where it has
    no ell; the pick's fields are empty where it has no pick."""
    ell, choice = found["ell"], found["cap"]["pick"]
    if ell is None:
        return {}
    out = _ell_fields(ell)
    if choice is None:
        return out
    return out | _pick_fields(choice, out["bound"])


def _cell_records(rows: pd.DataFrame, cell: str, n_kinds: int) -> dict[str, list]:
    """Return the rows of every table for one cell, each led by the cell."""
    cands = _cell_candidates(rows, cell, n_kinds)
    found, head = pick(cands), {"cell": cell}
    return {
        "candidates": [head | _candidate_row(r) for r in cands],
        "picks": [head | _pick_row(cands, found)],
        "beside": [head | r for r in _beside_rows(found)],
        "kinds": [head | r for r in _kind_rows(found, n_kinds)],
        "condition": [head | _condition_row(found)],
    }


def tables(
    rows: pd.DataFrame, cells: list[str], n_kinds: int
) -> dict[str, pd.DataFrame]:
    """Return the tables of one lattice: candidates, picks, beside, kinds, condition.

    Args:
        rows: One row per candidate: ``run``, ``epoch`` and ``<key>_<cell>`` for every
            key of ``src.callbacks.ancap8.keys(n_kinds)`` and every cell.
        cells: The cells, in the order of the tables' rows.
        n_kinds: The number of kinds of the points; 0 gives an empty kinds table.
    """
    records = {name: [] for name in COLUMNS}
    for cell in cells:
        for name, found in _cell_records(rows, cell, n_kinds).items():
            records[name] += found
    return {
        name: pd.DataFrame(found, columns=COLUMNS[name], dtype=object)
        for name, found in records.items()
    }
