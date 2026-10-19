"""The extractor must order objects the way the published record's loader does."""

import awkward as ak

from src.data.components.extraction import _et_ordered


def test_objects_come_out_hardest_first():
    # The torch stage clips by position, so the leading object must lead.
    data = ak.zip({"Et": [[3, 9, 1], [], [5]], "eta": [[0, 1, 2], [], [7]]})
    ordered = _et_ordered(data)

    assert ordered.Et.tolist() == [[9, 3, 1], [], [5]]
    assert ordered.eta.tolist() == [[1, 0, 2], [], [7]]


def test_ties_keep_the_readout_order():
    # A stable sort leaves the collections that already arrive ordered untouched.
    data = ak.zip({"Et": [[9, 9, 9]], "eta": [[0, 1, 2]]})

    assert _et_ordered(data).eta.tolist() == [[0, 1, 2]]
