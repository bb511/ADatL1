"""Synthetic acceptance tests for the Phase 4 figures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.pareto_plots import (
    FIGURE_FILENAMES,
    GAMMA_RAMP,
    ParetoPlotError,
    front_parallel_coordinates,
    write_pareto_figures,
)
from src.evaluation.pareto_selection import select_pareto_front


def _row(
    configuration_id: str,
    *,
    gamma: float,
    bins: int,
    architecture: str,
    leakage: float,
    correlation: float,
    efficiency: float,
    valid: bool = True,
    feasible: bool = True,
) -> dict[str, object]:
    row: dict[str, object] = {
        "study_id": "study",
        "protocol_version": "fet-et-four-probe-v10",
        "configuration_id": configuration_id,
        "mi_gamma": gamma,
        "mi_sensitive_num_bins": bins,
        "architecture_id": architecture,
        "configuration_valid": valid,
        "feasible": feasible,
        "rejection_reasons": "" if valid and feasible else "paired_constraints_failed",
    }
    for column, value in (
        ("leakage_worst", leakage),
        ("residual_correlation", correlation),
        ("median_efficiency", efficiency),
    ):
        row[f"{column}_mean"] = value
        row[f"{column}_n_seeds"] = 2
        row[f"{column}_sample_std"] = 0.008
        row[f"{column}_standard_error"] = 0.008 / 2 ** 0.5
        row[f"{column}_seed_min"] = value - 0.004
        row[f"{column}_seed_max"] = value + 0.004
        row[f"{column}_ci95_low"] = value - 0.01
        row[f"{column}_ci95_high"] = value + 0.01
    return row


def _study() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for index, architecture in enumerate(("h64_32", "h128_64", "h64_64_32")):
        for step, gamma in enumerate((0.0, 0.05, 0.1, 0.15, 0.2, 0.25)):
            rows.append(
                _row(
                    f"{architecture}-g{gamma}",
                    gamma=gamma,
                    bins=(40, 50, 60)[step % 3],
                    architecture=architecture,
                    leakage=0.12 - 0.02 * step + 0.004 * index,
                    correlation=0.30 - 0.03 * step,
                    efficiency=0.90 - 0.05 * step,
                )
            )
    rows.append(
        _row("collapsed", gamma=0.25, bins=60, architecture="h64_32",
             leakage=0.01, correlation=0.01, efficiency=0.99, feasible=False)
    )
    candidates, front, _ = select_pareto_front(pd.DataFrame(rows))
    return candidates, front


def test_every_figure_is_written(tmp_path: Path) -> None:
    candidates, front = _study()
    written = write_pareto_figures(candidates, front, output_dir=tmp_path)

    assert set(written) == set(FIGURE_FILENAMES)
    for name, path in written.items():
        assert path.name == FIGURE_FILENAMES[name]
        assert path.is_file() and path.stat().st_size > 0


def test_infeasible_configuration_is_not_treated_as_a_competitor(tmp_path: Path) -> None:
    """A latent-collapsed run must never be ringed as front material."""

    candidates, front = _study()
    collapsed = candidates.loc[candidates["configuration_id"] == "collapsed"].iloc[0]

    assert collapsed["selection_status"] == "infeasible"
    assert not bool(collapsed["is_pareto_front"])
    assert "collapsed" not in set(front["configuration_id"])
    write_pareto_figures(candidates, front, output_dir=tmp_path)


def test_more_gamma_levels_than_ramp_steps_is_refused(tmp_path: Path) -> None:
    """Never cycle or generate a hue past the validated ramp."""

    rows = [
        _row(f"g{index}", gamma=0.05 * (index + 1), bins=50, architecture="h64_32",
             leakage=0.1 - 0.001 * index, correlation=0.2, efficiency=0.8)
        for index in range(len(GAMMA_RAMP) + 1)
    ]
    candidates, front, _ = select_pareto_front(pd.DataFrame(rows))
    with pytest.raises(ParetoPlotError, match="ordinal ramp"):
        write_pareto_figures(candidates, front, output_dir=tmp_path)


def test_empty_front_refuses_the_parallel_coordinates_figure(tmp_path: Path) -> None:
    candidates, _ = _study()
    with pytest.raises(ParetoPlotError, match="empty"):
        front_parallel_coordinates(
            candidates.iloc[0:0], output_path=tmp_path / "front.png"
        )


def test_missing_column_is_reported_rather_than_crashing_matplotlib(tmp_path: Path) -> None:
    candidates, front = _study()
    with pytest.raises(ParetoPlotError, match="missing columns"):
        write_pareto_figures(
            candidates.drop(columns=["architecture_id"]), front, output_dir=tmp_path
        )


def test_bars_are_the_seed_spread_not_a_confidence_interval(tmp_path: Path) -> None:
    """The drawn bar must come from sample_std, never from the ci95 columns.

    Two seeds cannot support an inferential claim, so a bar sized from a 95%
    interval would overstate what the study knows.
    """

    from src.evaluation.pareto_plots import _spread_bounds

    candidates, _ = _study()
    bounds = _spread_bounds(candidates, "leakage_worst_mean")

    assert bounds.shape == (2, len(candidates))
    assert (bounds[0] == bounds[1]).all()
    assert (bounds[0] == candidates["leakage_worst_sample_std"].to_numpy()).all()
