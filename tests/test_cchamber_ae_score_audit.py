from pathlib import Path

import numpy as np

from scripts import cchamber_ae_score_audit as audit


def test_score_families_are_complete_and_finite() -> None:
    """Every frozen AE score is a finite event-level vector."""
    generator = np.random.default_rng(7)
    residual = generator.normal(size=(128, 4))
    latent = generator.normal(size=(128, 3))
    state = audit._fit_score_state(residual, latent, delta=1.0)
    scores = audit.score_arrays(residual, latent, state)

    assert tuple(scores) == audit.SCORE_NAMES
    assert all(value.shape == (128,) for value in scores.values())
    assert all(np.isfinite(value).all() for value in scores.values())
    np.testing.assert_allclose(scores["mse"], np.mean(residual**2, axis=1))


def test_pairing_controls_are_deterministic() -> None:
    """Metadata, CDF, and seeded-random pairings have frozen behavior."""
    left = np.asarray([3.0, 1.0, 2.0])
    right = np.asarray([20.0, 30.0, 10.0])
    x = np.arange(6, dtype=float).reshape(3, 2)

    metadata = audit._pair_indices(
        "metadata", left, right, encoder_table=None, split="valid", x_1=x, x_2=x
    )
    cdf = audit._pair_indices("cdf", left, right, encoder_table=None, split="valid", x_1=x, x_2=x)
    random_first = audit._pair_indices(
        "random", left, right, encoder_table=None, split="valid", x_1=x, x_2=x
    )
    random_second = audit._pair_indices(
        "random", left, right, encoder_table=None, split="valid", x_1=x, x_2=x
    )

    np.testing.assert_array_equal(metadata[0], np.arange(3))
    np.testing.assert_array_equal(metadata[1], np.arange(3))
    np.testing.assert_array_equal(left[cdf[0]], np.sort(left))
    np.testing.assert_array_equal(right[cdf[1]], np.sort(right))
    np.testing.assert_array_equal(random_first[0], random_second[0])
    np.testing.assert_array_equal(random_first[1], random_second[1])


def test_holm_adjustment_preserves_original_order() -> None:
    """Multiplicity adjustment is monotone in ordered raw p-values."""
    adjusted = audit._holm([0.01, 0.04, 0.03])
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])


def test_parser_exposes_outcome_gate_stages() -> None:
    """The CLI keeps proxy freezing separate from intervention evaluation."""
    parser = audit._parser()
    args = parser.parse_args(
        [
            "extract-normal",
            "--audit-root",
            str(Path("audit")),
            "--output-root",
            str(Path("output")),
            "--trajectory-index",
            "3",
        ]
    )
    assert args.command == "extract-normal"
    assert args.trajectory_index == 3
