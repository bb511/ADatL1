"""Synthetic acceptance tests for the lean Phase 3 Pareto selector."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.pareto.selection import (
    ParetoSelectionError,
    read_phase2_table,
    select_and_write_pareto_front,
    select_pareto_front,
)


def _row(
    configuration_id: str,
    *,
    leakage: float = 0.2,
    correlation: float = 0.2,
    efficiency: float = 0.7,
    valid: bool = True,
    feasible: bool = True,
    rejection_reasons: str = "",
) -> dict[str, object]:
    values = {
        "leakage_worst": leakage,
        "residual_correlation": correlation,
        "median_efficiency": efficiency,
    }
    row: dict[str, object] = {
        "study_id": "synthetic-study",
        "protocol_version": "fet-et-pareto-v3",
        "configuration_id": configuration_id,
        "configuration_valid": valid,
        "feasible": feasible,
        "rejection_reasons": rejection_reasons,
        **values,
    }
    return row


def _table(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_pareto_dominance_honours_minimize_and_maximize_directions() -> None:
    candidates, front, selection = select_pareto_front(
        _table(
            _row("a", leakage=0.1, correlation=0.1, efficiency=0.8),
            _row("b", leakage=0.2, correlation=0.2, efficiency=0.7),
            _row("c", leakage=0.05, correlation=0.3, efficiency=0.7),
        )
    )

    assert list(front["configuration_id"]) == ["a", "c"]
    assert candidates.loc[
        candidates["configuration_id"] == "b", "selection_status"
    ].item() == "dominated"
    assert selection["selected_configuration_id"] == "a"


def test_equal_objective_ties_stay_on_front_and_rank_by_configuration_id() -> None:
    _, front, selection = select_pareto_front(
        _table(
            _row("z", leakage=0.1, correlation=0.1, efficiency=0.9),
            _row("a", leakage=0.1, correlation=0.1, efficiency=0.9),
        )
    )

    assert list(front["configuration_id"]) == ["a", "z"]
    assert list(front["pareto_rank"]) == [1, 2]
    assert selection["selected_configuration_id"] == "a"


def test_invalid_and_infeasible_configurations_are_retained_but_not_ranked() -> None:
    invalid = _row(
        "invalid",
        valid=False,
        feasible=False,
        rejection_reasons="invalid_probe",
    )
    invalid["leakage_worst"] = np.nan
    candidates, front, _ = select_pareto_front(
        _table(
            _row("selected", leakage=0.1, correlation=0.1, efficiency=0.9),
            _row(
                "collapsed",
                feasible=False,
                rejection_reasons="paired_constraints_failed:['relative_entropy_pass']",
            ),
            invalid,
        )
    )

    statuses = candidates.set_index("configuration_id")["selection_status"].to_dict()
    assert statuses == {
        "collapsed": "infeasible",
        "invalid": "invalid",
        "selected": "pareto_front",
    }
    assert list(front["configuration_id"]) == ["selected"]
    assert "invalid_probe" in candidates.loc[
        candidates["configuration_id"] == "invalid", "rejection_reasons"
    ].item()


def test_eligible_nonfinite_objective_is_rejected() -> None:
    broken = _row("broken")
    broken["median_efficiency"] = np.nan

    with pytest.raises(ParetoSelectionError, match="finite median_efficiency"):
        select_pareto_front(_table(broken))


def test_csv_cli_path_writes_all_lean_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "pareto_metrics.csv"
    _table(
        _row("a", leakage=0.1, correlation=0.1, efficiency=0.9),
        _row("b", leakage=0.2, correlation=0.2, efficiency=0.8),
    ).to_csv(input_path, index=False)

    result = select_and_write_pareto_front(input_path, output_dir=tmp_path / "selection")
    selection_dir = tmp_path / "selection"
    assert result["selected_configuration_id"] == "a"
    assert read_phase2_table(input_path).equals(pd.read_csv(input_path))
    assert (selection_dir / "pareto_candidates.csv").is_file()
    assert (selection_dir / "pareto_front.csv").is_file()
    report = json.loads((selection_dir / "pareto_selection.json").read_text())
    assert report["selected_configuration_id"] == "a"
    assert report["counts"]["pareto_front_configurations"] == 1


def test_report_records_paths_relative_to_itself(tmp_path: Path) -> None:
    """The report must survive the study tree being copied off EOS.

    Phase 2 and phase 3 are siblings under the study root, so the recorded
    provenance is ../phase2/pareto_metrics.csv and the outputs are bare
    filenames. An absolute path here would be correct only on the machine that
    produced it.
    """

    phase2 = tmp_path / "phase2"
    phase3 = tmp_path / "phase3"
    phase2.mkdir()
    input_path = phase2 / "pareto_metrics.csv"
    _table(
        _row("a", leakage=0.1, correlation=0.1, efficiency=0.9),
        _row("b", leakage=0.2, correlation=0.2, efficiency=0.8),
    ).to_csv(input_path, index=False)

    select_and_write_pareto_front(input_path, output_dir=phase3)
    report = json.loads((phase3 / "pareto_selection.json").read_text())

    assert report["input_table"] == str(Path("..") / "phase2" / "pareto_metrics.csv")
    assert report["output_paths"] == {
        "candidates_csv": "pareto_candidates.csv",
        "front_csv": "pareto_front.csv",
        "selection_json": "pareto_selection.json",
    }
    for relative in report["output_paths"].values():
        assert not Path(relative).is_absolute()
        assert (phase3 / relative).is_file()
    assert (phase3 / report["input_table"]).resolve() == input_path.resolve()


def test_selection_writes_no_figures(tmp_path: Path) -> None:
    """Phase 3 computes; phase 4 draws. Selection must emit no images."""

    input_path = tmp_path / "pareto_metrics.csv"
    _table(
        _row("a", leakage=0.1, correlation=0.1, efficiency=0.9),
        _row("b", leakage=0.2, correlation=0.2, efficiency=0.8),
    ).to_csv(input_path, index=False)

    select_and_write_pareto_front(input_path, output_dir=tmp_path / "phase3")

    assert not list((tmp_path / "phase3").glob("*.png"))
