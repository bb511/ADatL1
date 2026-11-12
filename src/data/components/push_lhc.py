"""The LHC's pushed points of ANCAP v8: held-out zero-bias rows moved in four kinds.

The definition is the frozen page
``scratchpad/adl1t/theory/theory_xxxi/step4/push/LHC.md`` (md5 75fe4d63...), sections
"Constants", "The possible events" and "The four kinds of move", with the readings of
its last section. The scales are those read there and fixed in
``theory_xxxi/step4/V8.md``, section 2. At these scales the rows built here are the
frozen reference rows of ``theory_xxxi/step4/FREEZE.md`` byte for byte, which the page's
code A, ``push/lhc_a/lhc_push2.py``, wrote.

A row is (39, 3) over (slot, feature), features (Et, eta, phi), and every move acts on
its hardware integers: FET in slot 0 (Et and phi, eta masked), e/gamma in slots 1 to 12,
jets in 13 to 22, muons in 23 to 26 and taus in 27 to 38. The present slots of a
collection come first, Et never rising, and an object is present where its Et is. In
float64, a normalised x has the hardware value rint(x · scale + shift) and a hardware hw
the normalised value (hw - shift) / scale, cast to float32; a masked input is +0.0.

T₁ is one million training rows and H ten thousand test rows, each drawn without
replacement and sorted. Over T₁, for each collection c, low_c is the smallest Et of a
present object, and V_eta[c] and V_phi[c] are the sorted distinct eta and phi of present
objects. Each row of H is pushed by each kind at its scale s::

    1  each present object: Et' = min(cap_c, low_c + (s max(0, Et - low_c) + 50) // 100)
    2  FET: Et' = min(4094, (s Et + 50) // 100)
    3  min(s, free slots) objects in free slots drawn without replacement, each with Et
       uniform from low_c to min(cap_c, E_max), E_max the row's largest Et, and (eta,
       phi) drawn from T₁'s grid; FET less the added calorimeter objects' vectors
    4  s splits of a present object of Et >= 2 low_c in a collection with a free slot,
       the piece's Et uniform from low_c to Et - low_c and its (eta, phi) one grid step
       from its parent's; each collection's total Et and FET kept

Row i of H draws from ``numpy.random.default_rng([20261114, kind, i])`` in the page's
order, and after a move every collection is sorted by Et, stably. Scale 0 (s = 100 for
kinds 1 and 2, s = 0 for kinds 3 and 4) returns the row. ``count_impossible`` counts the
rows that break any of the page's five conditions on the possible events.
"""

import numpy as np

FEATURES = ("Et", "eta", "phi")
N_SLOTS = 39
# name: (first slot, number of slots, Et cap, phi period), in the order the moves visit.
COLLECTIONS = {
    "egammas": (1, 12, 510, 144),
    "jets": (13, 10, 2046, 144),
    "muons": (23, 4, 510, 576),
    "taus": (27, 12, 510, 144),
}
SLOT_COLLECTION = {
    s: c for c, (first, n, _, _) in COLLECTIONS.items() for s in range(first, first + n)
}
FET_CAP, FET_PERIOD = 4094, 144
OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
T1_SEED, H_SEED, MOVE_SEED = 20261112, 20261113, 20261114
N_T1, N_H = 1_000_000, 10_000
# F₁₀₀ of kinds 1 and 2 and k of kinds 3 and 4, as read and frozen (V8.md, section 2).
SCALES = {1: 283, 2: 673, 3: 6, 4: 6}

Norm = tuple[np.ndarray, np.ndarray]
Constants = dict[str, dict]
Object = tuple[int, int, int]
Objects = dict[str, list[Object]]


def norm_arrays(params: dict[str, dict]) -> Norm:
    """Return the scale and the shift of every (slot, feature), each (39, 3) float64.

    ``params`` maps FET and each collection to the content of its
    ``<name>_norm_params.pkl``, ``{feature: {"shift": ..., "scale": ...}}``. FET's eta,
    always masked, has none and keeps scale 1 and shift 0.
    """
    scale, shift = np.ones((N_SLOTS, 3)), np.zeros((N_SLOTS, 3))
    blocks = {"FET": (0, 1), **{c: v[:2] for c, v in COLLECTIONS.items()}}
    for name, (first, n) in blocks.items():
        for feature, p in params[name].items():
            j = FEATURES.index(feature)
            scale[first : first + n, j] = float(p["scale"])
            shift[first : first + n, j] = float(p["shift"])
    return scale, shift


def decode(x: np.ndarray, mask: np.ndarray, norm: Norm) -> np.ndarray:
    """Return the hardware integers of normalised rows, int64, 0 where masked."""
    scale, shift = norm
    hw = np.rint(x.astype(np.float64) * scale + shift).astype(np.int64)
    return np.where(mask, hw, 0)


def encode(hw: np.ndarray, mask: np.ndarray, norm: Norm) -> np.ndarray:
    """Return the normalised float32 rows of hardware integers, +0.0 where masked."""
    scale, shift = norm
    x = ((hw - shift) / scale).astype(np.float32)
    return np.where(mask, x, np.float32(0.0))


def draw_rows(n: int, size: int, seed: int) -> np.ndarray:
    """Return ``size`` distinct rows of ``n``, sorted, int64: T₁ or H by its seed."""
    rows = np.random.default_rng(seed).choice(n, size, replace=False)
    return np.sort(rows).astype(np.int64)


def constants(hw: np.ndarray, mask: np.ndarray) -> Constants:
    """Return low_c, V_eta[c] and V_phi[c] of each collection over the rows of T₁."""
    consts = {}
    for c, (first, n, _, _) in COLLECTIONS.items():
        objects = hw[:, first : first + n][mask[:, first : first + n, 0]]
        consts[c] = {
            "low": int(objects[:, 0].min()),
            "v_eta": np.unique(objects[:, 1]),
            "v_phi": np.unique(objects[:, 2]),
        }
    return consts


def push_objects_energy(
    hw: np.ndarray, mask: np.ndarray, f100: int, consts: Constants
) -> np.ndarray:
    """Return kind 1's rows: each present object's Et above low_c times F₁₀₀ / 100."""
    new = hw.copy()
    for c, (first, n, cap, _) in COLLECTIONS.items():
        low, present = consts[c]["low"], mask[:, first : first + n, 0]
        et = hw[:, first : first + n, 0]
        raw = low + (f100 * np.maximum(0, et - low) + 50) // 100
        new[:, first : first + n, 0] = np.where(present, np.minimum(cap, raw), et)
    return new


def push_missing_energy(hw: np.ndarray, mask: np.ndarray, f100: int) -> np.ndarray:
    """Return kind 2's rows: FET's Et times F₁₀₀ / 100, nothing else moved."""
    new, et = hw.copy(), hw[:, 0, 0]
    raw = (f100 * et + 50) // 100
    new[:, 0, 0] = np.where(mask[:, 0, 0], np.minimum(FET_CAP, raw), et)
    return new


def _by_et(objs: list[Object]) -> list[Object]:
    """Sort by Et, largest first; equal Et keep their order, since sorted is stable."""
    return sorted(objs, key=lambda o: -o[0])


def _new_object(
    rng: np.random.Generator, c: str, e_max: int, consts: Constants
) -> Object:
    """Draw an object of c: Et uniform from low_c to min(cap_c, e_max), T₁'s grid."""
    low, cap = consts[c]["low"], COLLECTIONS[c][2]
    v_eta, v_phi = consts[c]["v_eta"], consts[c]["v_phi"]
    u = rng.random()
    span = max(0, min(cap, e_max) - low)
    et = low + min(span, int(u * (span + 1)))
    eta = int(v_eta[rng.integers(0, len(v_eta))])
    phi = int(v_phi[rng.integers(0, len(v_phi))])
    return et, eta, phi


def _calorimeter_sum(made: list[tuple[str, Object]]) -> tuple[float, float]:
    """Return the summed (Et cos, Et sin) of the e/gammas, jets and taus made."""
    sx = sy = 0.0
    for c, (et, _, phi) in made:
        # FET is a calorimeter sum (kMissingEtHF): an added muon leaves it unchanged.
        if c == "muons":
            continue
        theta = 2 * np.pi * phi / FET_PERIOD
        sx += et * np.cos(theta)
        sy += et * np.sin(theta)
    return sx, sy


def recompute_fet(
    fet: tuple[int, int], made: list[tuple[str, Object]]
) -> tuple[int, int]:
    """Return FET's (Et, phi) less the vectors of the calorimeter objects made.

    Dillon et al. recompute MET so (arXiv:2301.04660, section 4); where the new Et is 0,
    the old phi is kept. The float64 operations and their order are the page's, on which
    the frozen rows depend.
    """
    sx, sy = _calorimeter_sum(made)
    theta_f = 2 * np.pi * fet[1] / FET_PERIOD
    vx, vy = fet[0] * np.cos(theta_f) - sx, fet[0] * np.sin(theta_f) - sy
    et_f = min(FET_CAP, int(np.rint(np.hypot(vx, vy))))
    if et_f == 0:
        return 0, fet[1]
    phi_f = np.rint(np.arctan2(vy, vx) * FET_PERIOD / (2 * np.pi))
    return et_f, int(phi_f) % FET_PERIOD


def add_objects(
    objs: Objects,
    fet: tuple[int, int],
    free: list[int],
    i: int,
    k: int,
    consts: Constants,
) -> tuple[int, int]:
    """Add min(k, len(free)) objects to ``objs`` in place and return the new FET.

    ``free`` lists the row's empty slots in slot order, and row i of H seeds the draws.
    """
    rng = np.random.default_rng([MOVE_SEED, 3, i])
    perm = rng.permutation(len(free))
    e_max = max([fet[0]] + [o[0] for c in objs for o in objs[c]])
    made = []
    for j in range(min(k, len(free))):
        c = SLOT_COLLECTION[free[perm[j]]]
        made.append((c, _new_object(rng, c, e_max, consts)))
        objs[c].append(made[-1][1])
    objs.update({c: _by_et(v) for c, v in objs.items()})
    return recompute_fet(fet, made)


def _grid_index(v: np.ndarray, value: int) -> int:
    """Return the index of value on the sorted grid v, or of the next point above."""
    return min(int(np.searchsorted(v, value)), len(v) - 1)


def neighbour(const: dict, eta: int, phi: int, d: tuple[int, int]) -> tuple[int, int]:
    """Return the grid point d steps from (eta, phi), eta clipped and phi wrapped.

    ``const`` holds one collection's grids ``v_eta`` and ``v_phi``. A value off its grid
    steps from the next point above it, or from the last point beyond the grid's end.
    """
    v_eta, v_phi = const["v_eta"], const["v_phi"]
    i_eta = min(max(_grid_index(v_eta, eta) + d[0], 0), len(v_eta) - 1)
    i_phi = (_grid_index(v_phi, phi) + d[1]) % len(v_phi)
    return int(v_eta[i_eta]), int(v_phi[i_phi])


def _split_candidates(objs: Objects, consts: Constants) -> list[tuple[str, int]]:
    """Return (collection, position) of each object that kind 4 can split."""
    return [
        (c, p)
        for c in objs
        for p in range(len(objs[c]))
        if len(objs[c]) < COLLECTIONS[c][1] and objs[c][p][0] >= 2 * consts[c]["low"]
    ]


def _split(
    rng: np.random.Generator, objs: Objects, c: str, p: int, const: dict
) -> None:
    """Split object p of c in two, the new piece's Et from low_c to Et - low_c."""
    (et, eta, phi), low = objs[c][p], const["low"]
    u = rng.random()
    a = low + min(et - 2 * low, int(u * (et - 2 * low + 1)))
    d = OFFSETS[rng.integers(0, 8)]
    objs[c][p] = (et - a, eta, phi)
    objs[c] = _by_et(objs[c] + [(a, *neighbour(const, eta, phi, d))])


def split_objects(objs: Objects, i: int, k: int, consts: Constants) -> None:
    """Split up to k times, in place; row i of H seeds the draws.

    Fewer than k splits are made when no object can split: each collection with a free
    slot then holds only objects below 2 low_c.
    """
    rng = np.random.default_rng([MOVE_SEED, 4, i])
    for _ in range(k):
        candidates = _split_candidates(objs, consts)
        if not candidates:
            return
        c, p = candidates[rng.integers(0, len(candidates))]
        _split(rng, objs, c, p, consts[c])


def _row_objects(hw_row: list, mask_row: list) -> Objects:
    """Return the present objects of a row per collection, in slot order."""
    return {
        c: [tuple(hw_row[s]) for s in range(first, first + n) if mask_row[s][0]]
        for c, (first, n, _, _) in COLLECTIONS.items()
    }


def push_row(
    hw_row: list, mask_row: list, i: int, kind: int, k: int, consts: Constants
) -> tuple[Objects, tuple[int, int]]:
    """Return the objects and FET's (Et, phi) of row i of H after kind 3 or 4 at k.

    ``hw_row`` and ``mask_row`` are the row's hardware integers and mask, as lists.
    """
    objs = _row_objects(hw_row, mask_row)
    fet = (hw_row[0][0], hw_row[0][2])
    if kind == 4:
        split_objects(objs, i, k, consts)
        return objs, fet
    free = [s for s in range(1, N_SLOTS) if not mask_row[s][0]]
    return objs, add_objects(objs, fet, free, i, k, consts)


def _write_row(
    objs: Objects, fet: tuple[int, int], hw: np.ndarray, mask: np.ndarray
) -> None:
    """Write a row's FET and objects into its (39, 3) hw and mask, collections first."""
    hw[0] = (fet[0], 0, fet[1])
    for c, (first, _, _, _) in COLLECTIONS.items():
        for j, obj in enumerate(objs[c]):
            hw[first + j] = obj
            mask[first + j] = True


def _push_rows(
    hw: np.ndarray, mask: np.ndarray, kind: int, k: int, consts: Constants
) -> tuple[np.ndarray, np.ndarray]:
    """Return the rows after kind 3 or 4, rebuilt with FET's stored mask."""
    hw_out, mask_out = np.zeros_like(hw), np.zeros_like(mask)
    mask_out[:, 0] = mask[:, 0]
    hw_l, mask_l = hw.tolist(), mask.tolist()
    for i in range(len(hw)):
        objs, fet = push_row(hw_l[i], mask_l[i], i, kind, k, consts)
        _write_row(objs, fet, hw_out[i], mask_out[i])
    return hw_out, mask_out


def push(
    kind: int, hw: np.ndarray, mask: np.ndarray, scale: int, consts: Constants
) -> tuple[np.ndarray, np.ndarray]:
    """Return the hardware integers and mask of rows pushed by one kind at one scale.

    Row i of ``hw`` is row i of H, which seeds the draws of kinds 3 and 4. Kinds 1 and 2
    keep the mask; kinds 3 and 4 write every collection as a prefix of its slots.
    """
    if kind == 1:
        return push_objects_energy(hw, mask, scale, consts), mask.copy()
    if kind == 2:
        return push_missing_energy(hw, mask, scale), mask.copy()
    if kind in (3, 4):
        return _push_rows(hw, mask, kind, scale, consts)
    raise ValueError(f"no kind of move {kind}")


def _fet_impossible(hw: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Flag the rows whose FET is not (Et, masked eta, phi) within its ranges."""
    shape_ok = mask[:, 0, 0] & ~mask[:, 0, 1] & mask[:, 0, 2]
    et, phi = hw[:, 0, 0], hw[:, 0, 2]
    return ~shape_ok | (et < 0) | (et > FET_CAP) | (phi < 0) | (phi >= FET_PERIOD)


def _slots_impossible(m: np.ndarray) -> np.ndarray:
    """Flag the rows of one collection's mask (n, slots, 3) with a gap or a torn slot.

    A slot is torn when its three inputs are not present together.
    """
    together = ((m[..., 0] == m[..., 1]) & (m[..., 1] == m[..., 2])).all(1)
    return ~together | (m[:, 1:, 0] & ~m[:, :-1, 0]).any(1)


def _values_impossible(
    h: np.ndarray, present: np.ndarray, c: str, consts: Constants
) -> np.ndarray:
    """Flag the rows of collection c with a value out of range or a rising Et."""
    cap, period = COLLECTIONS[c][2:]
    low, v_eta = consts[c]["low"], consts[c]["v_eta"]
    et, eta, phi = h[..., 0], h[..., 1], h[..., 2]
    outside = (et < low) | (et > cap) | (eta < v_eta[0]) | (eta > v_eta[-1])
    outside |= (phi < 0) | (phi >= period)
    # With the prefix rule, consecutive present slots are the slots s and s + 1.
    rising = present[:, 1:] & present[:, :-1] & (et[:, 1:] > et[:, :-1])
    return (present & outside).any(1) | rising.any(1)


def count_impossible(
    x: np.ndarray, mask: np.ndarray, norm: Norm, consts: Constants
) -> int:
    """Return the number of rows outside the possible events, read from float32 rows."""
    hw = decode(x, mask, norm)
    bad = (~mask & (x != 0)).any((1, 2)) | _fet_impossible(hw, mask)
    # A present float32 holds a hardware integer when it is that integer's encoding.
    bad |= (mask & (encode(hw, mask, norm) != x)).any((1, 2))
    for c, (first, n, _, _) in COLLECTIONS.items():
        m, h = mask[:, first : first + n], hw[:, first : first + n]
        bad |= _slots_impossible(m) | _values_impossible(h, m[..., 0], c, consts)
    return int(bad.sum())
