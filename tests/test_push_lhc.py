import numpy as np
import pytest

from src.data.components import push_lhc as pl

# Every input encodes as (hw - 3) / 8, exact in float32 for these integers.
NORM = (np.full((39, 3), 8.0), np.full((39, 3), 3.0))
LOWS = {"egammas": 4, "jets": 1, "muons": 3, "taus": 6}
V_ETA, V_PHI = np.arange(-6, 7, 3), np.arange(0, 144, 2)
CONSTS = {c: {"low": low, "v_eta": V_ETA, "v_phi": V_PHI} for c, low in LOWS.items()}
# Every low_c is 2 and every grid one point: in a row whose largest Et is 2, each object
# drawn is (2, 5, 36), whichever slot it takes.
PINNED = {c: {"low": 2, "v_eta": np.array([5]), "v_phi": np.array([36])} for c in LOWS}


def _row(fet=(40, 3), **collections):
    hw, mask = np.zeros((39, 3), np.int64), np.zeros((39, 3), bool)
    hw[0], mask[0] = (fet[0], 0, fet[1]), (True, False, True)
    for c, objects in collections.items():
        first = pl.COLLECTIONS[c][0]
        for j, obj in enumerate(objects):
            hw[first + j], mask[first + j] = obj, True
    return hw, mask


def _rows(*rows):
    return np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows])


def _full(obj=(2, 5, 36), **free):
    return {
        c: [obj] * (n - free.get(c, 0)) for c, (_, n, _, _) in pl.COLLECTIONS.items()
    }


GOOD = _row(egammas=[(9, 3, 10), (5, -3, 20)], muons=[(7, 0, 500)])


def test_norm_arrays_fill_each_block_and_leave_fet_eta_neutral():
    fet = {"Et": {"shift": 1.0, "scale": 2.0}, "phi": {"shift": 3.0, "scale": 4.0}}
    params = {"FET": fet}
    for k, c in enumerate(pl.COLLECTIONS, start=1):
        params[c] = dict.fromkeys(pl.FEATURES, {"shift": float(k), "scale": 10.0 * k})
    scale, shift = pl.norm_arrays(params)
    assert scale[0].tolist() == [2.0, 1.0, 4.0] and shift[0].tolist() == [1.0, 0.0, 3.0]
    blocks = np.repeat([10.0, 20.0, 30.0, 40.0], [12, 10, 4, 12])[:, None]
    assert (scale[1:] == blocks).all() and (shift[1:] == blocks / 10).all()


def test_encode_writes_masked_inputs_as_positive_zero_and_decode_inverts_it():
    hw, mask = _rows(GOOD)
    # A value under a mask is not encoded.
    hw[0, 3] = (5, 5, 5)
    x = pl.encode(hw, mask, NORM)
    assert x.dtype == np.float32 and (x[~mask] == 0).all()
    assert not np.signbit(x[~mask]).any()
    assert np.array_equal(pl.decode(x, mask, NORM), np.where(mask, hw, 0))


def test_constants_read_the_present_objects_alone():
    hw, mask = _rows(
        _row(egammas=[(9, 3, 10), (5, -3, 20)], jets=[(40, 0, 4)], taus=[(8, 2, 2)]),
        _row(egammas=[(4, 3, 10)], jets=[(2, -1, 6)], muons=[(7, 1, 500), (3, 1, 0)]),
    )
    # An absent slot holds values below every present one.
    hw[1, 2] = (1, -9, 99)
    consts = pl.constants(hw, mask)
    assert [consts[c]["low"] for c in pl.COLLECTIONS] == [4, 2, 3, 8]
    assert consts["egammas"]["v_eta"].tolist() == [-3, 3]
    assert consts["egammas"]["v_phi"].tolist() == [10, 20]
    assert consts["muons"]["v_phi"].tolist() == [0, 500]


def test_objects_energy_scales_above_low_and_holds_at_the_caps():
    row = _row(
        fet=(50, 10),
        egammas=[(14, 0, 0), (4, 0, 0)],
        jets=[(1000, 0, 0), (11, 0, 0)],
        muons=[(200, 0, 0), (103, 0, 0)],
        taus=[(6, 0, 0)],
    )
    hw, mask = _rows(row)
    new, new_mask = pl.push(1, hw, mask, 283, CONSTS)
    # low_c + (283 (Et - low_c) + 50) // 100; the jet holds at 2046, the muon at 510.
    expected = hw.copy()
    expected[0, [1, 2, 13, 14, 23, 24, 27], 0] = [32, 4, 2046, 29, 510, 286, 6]
    assert np.array_equal(new, expected) and np.array_equal(new_mask, mask)


def test_missing_energy_scales_fet_alone_and_holds_at_its_cap():
    hw, mask = _rows(
        _row(fet=(100, 7), jets=[(30, 0, 0)]), _row(fet=(1000, 7)), _row(fet=(0, 7))
    )
    new, new_mask = pl.push(2, hw, mask, 673, CONSTS)
    expected = hw.copy()
    expected[:, 0, 0] = [673, 4094, 0]
    assert np.array_equal(new, expected) and np.array_equal(new_mask, mask)


def test_added_calorimeter_objects_move_fet_and_added_muons_do_not():
    # One free e/gamma slot and four free muon slots, every Et at low_c.
    hw, mask = _rows(_row(fet=(2, 0), **_full((2, 0, 0), egammas=1, muons=4)))
    new, new_mask = pl.push(3, hw, mask, 5, PINNED)
    assert new_mask[0, 1:].all() and (new[0, 23:27] == (2, 5, 36)).all()
    # The new e/gamma enters after the old ones of equal Et.
    assert (new[0, 1:12] == (2, 0, 0)).all() and tuple(new[0, 12]) == (2, 5, 36)
    # FET (2, 0) less an Et of 2 at phi 36 (90 degrees) is (2, -2): Et 2.83, phi -18.
    assert tuple(new[0, 0]) == (3, 0, 126)


def test_added_muons_alone_leave_fet_unchanged():
    hw, mask = _rows(_row(fet=(50, 30), **_full((20, 5, 36), muons=4)))
    new, new_mask = pl.push(3, hw, mask, 4, PINNED)
    et = new[0, 23:27, 0]
    assert new_mask[0, 23:27].all() and tuple(new[0, 0]) == (50, 0, 30)
    assert (et[:-1] >= et[1:]).all() and (et >= 2).all() and (et <= 50).all()


def test_more_objects_than_free_slots_fill_every_slot():
    hw, mask = _rows(_row(fet=(40, 3), **_full(jets=2, taus=1)))
    new, new_mask = pl.push(3, hw, mask, 6, PINNED)
    x = pl.encode(new, new_mask, NORM)
    assert new_mask[0, 1:].all()
    assert pl.count_impossible(x, new_mask, NORM, PINNED) == 0


@pytest.mark.parametrize(
    "eta, phi, d, planted",
    [
        (6, 142, (1, 1), (6, 0)),  # eta clipped at the top, phi wrapped past 142
        (-6, 0, (-1, -1), (-6, 142)),  # eta clipped at the bottom, phi wrapped below 0
        (0, 10, (1, -1), (3, 8)),
        (1, 10, (-1, 0), (0, 10)),  # eta 1 lies off the grid and steps from 3
        (9, 10, (0, 1), (6, 12)),  # eta 9 lies beyond the grid and steps from 6
    ],
)
def test_split_offsets_clip_eta_and_wrap_phi(eta, phi, d, planted):
    assert pl.neighbour(CONSTS["jets"], eta, phi, d) == planted


def test_split_at_twice_low_gives_two_pieces_at_low_and_then_stops():
    # The jet, below 2 low_c, cannot split.
    hw, mask = _rows(_row(fet=(9, 4), egammas=[(4, 5, 36)], jets=[(3, 5, 36)]))
    new, new_mask = pl.push(4, hw, mask, 6, PINNED)
    two = _row(fet=(9, 4), egammas=[(2, 5, 36), (2, 5, 36)], jets=[(3, 5, 36)])
    assert np.array_equal(new[0], two[0]) and np.array_equal(new_mask[0], two[1])


def test_row_that_cannot_be_split_is_unchanged():
    # The e/gammas are full; the jet and the muon lie below 2 low_c.
    row = _row(
        fet=(9, 4), egammas=[(10, 5, 36)] * 12, jets=[(3, 5, 36)], muons=[(2, 5, 36)]
    )
    hw, mask = _rows(row)
    new, new_mask = pl.push(4, hw, mask, 6, PINNED)
    assert np.array_equal(new, hw) and np.array_equal(new_mask, mask)


def test_splits_keep_every_total_and_the_row_possible():
    hw, mask = _rows(_row(fet=(70, 9), egammas=[(60, 0, 10)], jets=[(90, 3, 142)]))
    new, new_mask = pl.push(4, hw, mask, 6, CONSTS)
    for first, n, _, _ in pl.COLLECTIONS.values():
        assert new[0, first : first + n, 0].sum() == hw[0, first : first + n, 0].sum()
    assert tuple(new[0, 0]) == (70, 0, 9) and new_mask[0, 1:, 0].sum() == 2 + 6
    x = pl.encode(new, new_mask, NORM)
    assert pl.count_impossible(x, new_mask, NORM, CONSTS) == 0


@pytest.mark.parametrize("kind, scale", [(1, 100), (2, 100), (3, 0), (4, 0)])
def test_scale_zero_returns_the_row(kind, scale):
    hw, mask = _rows(
        _row(fet=(4094, 143), egammas=[(30, 3, 70)], muons=[(9, 0, 575)]),
        _row(fet=(0, 72), egammas=[(4, -6, 0)], jets=[(2046, 6, 142)]),
        _row(fet=(37, 72), taus=[(6, -3, 2)] * 12),
    )
    new, new_mask = pl.push(kind, hw, mask, scale, CONSTS)
    x, new_x = pl.encode(hw, mask, NORM), pl.encode(new, new_mask, NORM)
    assert new_x.tobytes() == x.tobytes() and np.array_equal(new_mask, mask)


def test_count_impossible_passes_a_possible_row():
    hw, mask = _rows(GOOD)
    assert pl.count_impossible(pl.encode(hw, mask, NORM), mask, NORM, CONSTS) == 0


@pytest.mark.parametrize(
    "hw_edit, mask_edit, x_edit",
    [
        ({(1, 0): 511}, {}, {}),  # an e/gamma Et above its cap
        ({(2, 0): 3}, {}, {}),  # an e/gamma Et below low_c
        ({(1, 1): 7}, {}, {}),  # an eta beyond the ends of V_eta
        ({(23, 2): 576}, {}, {}),  # a muon phi beyond its period
        ({(2, 0): 10}, {}, {}),  # Et rising along the e/gammas
        ({3: (5, -3, 20)}, {2: False, 3: True}, {}),  # a gap before a present slot
        ({}, {(2, 1): False}, {}),  # a slot whose three inputs disagree
        ({}, {(0, 1): True}, {}),  # FET's eta present
        ({(0, 0): 4095}, {}, {}),  # FET above its cap
        ({}, {}, {(30, 1): 0.5}),  # a masked input not 0
        ({}, {}, {(1, 0): 0.78125}),  # an Et of 9.25, off the hardware grid
    ],
)
def test_count_impossible_flags_a_planted_row(hw_edit, mask_edit, x_edit):
    hw, mask = (a.copy() for a in GOOD)
    for index, value in hw_edit.items():
        hw[index] = value
    for index, value in mask_edit.items():
        mask[index] = value
    x = pl.encode(hw, mask, NORM)
    for index, value in x_edit.items():
        x[index] = value
    assert pl.count_impossible(x[None], mask[None], NORM, CONSTS) == 1
