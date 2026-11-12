from dataclasses import replace

import numpy as np
import pytest

from src.data.components import push_plain as pp

NAMES = ["count", "rate", "shift", "ht", "tau21_j1", "tau32_j1"]
MEDIAN = np.array([10.5, 2, 0, 5, 0.5, 0.5], dtype=np.float32)
SPREAD = np.array([4, 1, 2, 1, 0.25, 0.25], dtype=np.float32)
# Raw values, each exact as an input in float32: count holds integers, ht holds
# integers but is logged, shift goes below zero, and tau32 rises to 1.25.
RAW = np.array(
    [
        [2, 0.5, -3, 1, 0.25, 0.25],
        [10, 2, 0.5, 2, 0.5, 0.5],
        [14, 3.5, 1, 3, 0.75, 0.75],
        [30, 4, 2, 4, 1, 1.25],
    ]
)
X_TRAIN = ((RAW - MEDIAN) / SPREAD).astype(np.float32)
TABULAR = pp.tabular_events(X_TRAIN, MEDIAN, SPREAD, NAMES, ["ht"])

MEAN = np.array([0.5, 0.25, 0.75], dtype=np.float32)
STD = np.array([0.25, 0.2, 0.3], dtype=np.float32)
PIXELS = np.random.default_rng(0).integers(0, 256, (6, 3, 2, 2))
# Encoded in float32 as the image datamodules encode them: most are inexact.
X_IMAGES = (PIXELS.astype(np.float32) / 255 - MEAN[:, None, None]) / STD[:, None, None]
IMAGES = pp.image_events(X_IMAGES, MEAN, STD)


def _unbounded(d: int, tabular: bool) -> pp.Possible:
    return pp.Possible(
        a=np.ones(d),
        b=np.zeros(d),
        tol=np.full(d, 1e-5),
        lo=np.full(d, -np.inf),
        hi=np.full(d, np.inf),
        integer=np.zeros(d, dtype=bool),
        never_negative=np.zeros(d, dtype=bool),
        tabular=tabular,
    )


def test_tabular_events_read_flags_and_bounds_from_the_training_rows():
    assert TABULAR.integer.tolist() == [True, False, False, False, False, False]
    assert TABULAR.never_negative.tolist() == [True, True, False, False, True, True]
    # count's bound 10.5 + 40 is floored; tau21 stops at 1, tau32 at its largest value.
    assert TABULAR.lo.tolist() == [0, 0, -20, -5, 0, 0]
    assert TABULAR.hi.tolist() == [50, 12, 20, 15, 1, 1.25]
    assert pp.n_violations(X_TRAIN, TABULAR) == 0


def test_image_events_are_integer_pixels_from_0_to_255_per_channel():
    assert IMAGES.integer.all() and IMAGES.never_negative.all()
    assert (IMAGES.lo == 0).all() and (IMAGES.hi == 255).all()
    # Flattened (channel, row, column): each channel's scale covers four pixels.
    a = IMAGES.a.reshape(3, 4)
    assert (a == 255 * STD.astype(np.float64)[:, None]).all()
    assert pp.n_violations(X_IMAGES.reshape(len(X_IMAGES), -1), IMAGES) == 0


def test_projection_keeps_values_inside_their_bounds_bit_for_bit():
    p = np.array([[0.1, 0.123456789012345, -7.77777777, 9.87654321, 1.999, 2.5]])
    assert np.array_equal(pp.project(p, TABULAR)[:, 1:], p[:, 1:])


def test_projection_rounds_integer_inputs_and_clips_at_the_bounds():
    p = np.array(
        [[0.3, -5, 0, 0, 3, 2.9], [-5.1, 0, 0, 0, 0, 3.5], [20, 0, 0, 0, 0, 0]]
    )
    x = pp.project(p, TABULAR)
    # count: raw 11.7 rounds to 12, raw -9.9 to the floor 0, raw 90.5 to the bound 50.
    assert x[:, 0].tolist() == [0.375, -2.625, 9.875]
    # rate's floor 0, tau21's ceiling 1 and tau32's 1.25, in input units.
    assert x[0, [1, 4, 5]].tolist() == [-2, 2, 2.9] and x[1, 5] == 3


def test_tabular_points_are_clipped_at_ten_and_images_are_not():
    p = np.array([[-12.0, 15.0, 3.0]])
    assert pp.project(p, _unbounded(3, True)).tolist() == [[-10, 10, 3]]
    assert pp.project(p, _unbounded(3, False)).tolist() == [[-12, 15, 3]]


@pytest.mark.parametrize("events", [TABULAR, IMAGES], ids=["tabular", "images"])
def test_projected_points_violate_no_bound(events):
    p = 30 * np.random.default_rng(2).standard_normal((500, len(events.a)))
    assert pp.n_violations(p.astype(np.float32), events) > 0
    assert pp.n_violations(pp.project(p, events).astype(np.float32), events) == 0


@pytest.mark.parametrize(
    "j, value",
    [
        (0, 0.1),  # count off the integer grid
        (1, -2.5),  # rate below its floor 0
        (2, -10.5),  # shift below its bound 0 - 20
        (4, 2.5),  # tau21 above 1
        (5, 3.5),  # tau32 above its largest training value
    ],
)
def test_n_violations_flags_a_planted_row(j, value):
    x = X_TRAIN.copy()
    x[1, j] = value
    assert pp.n_violations(x, TABULAR) == 1


def test_first_rows_finds_the_first_equal_row_and_reads_minus_zero_as_zero():
    x = np.array([[1, 2], [0, 3], [1, 2], [-0.0, 3], [5, 5]], dtype=np.float32)
    rows = np.array([[-0.0, 3], [1, 2], [5, 5]], dtype=np.float32)
    index = pp.first_rows(x, rows)
    assert index.tolist() == [1, 0, 4] and index.dtype == np.int64
    with pytest.raises(KeyError):
        pp.first_rows(x, np.array([[9, 9]], dtype=np.float32))


def test_push_at_scale_zero_returns_its_start_rows():
    rows = np.random.default_rng(3).standard_normal((200, 3)).astype(np.float32)
    x, start = pp.push(rows.astype(np.float64), 0.0, _unbounded(3, False))
    assert len(x) > 0 and x.dtype == np.float32
    assert x.tobytes() == rows[start].tobytes()


def test_points_pushed_beyond_a_floor_are_projected_onto_it():
    rows = np.random.default_rng(4).uniform(0, 1, (200, 2))
    floor = replace(_unbounded(2, True), lo=np.zeros(2))
    x, _ = pp.push(rows, 4.0, floor)
    assert (x == 0).any() and pp.n_violations(x, floor) == 0
