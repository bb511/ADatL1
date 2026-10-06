"""Tests for the Phase 4c γ / effective-bin sweeps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from src.evaluation.pareto_matrices import METRICS  # noqa: E402
from src.evaluation.pareto_mi_changes import (  # noqa: E402
    EFFECTIVE_BINS,
    ParetoMiChangeError,
    bin_mapping_text,
    build_sweeps,
    effective_bins_by_configuration,
    write_mi_changes,
)

GAMMAS = (0.0, 0.1, 0.5, 0.9)
BINS = (10, 50, 100)
EFFECTIVE = {10: 10, 50: 48, 100: 64}


def _cid(gamma: float, bins: int) -> str:
    return f"study__gamma-{gamma}__bins-{bins}__arch-h64_32"


def _candidates() -> pd.DataFrame:
    rows = []
    for gamma in GAMMAS:
        for bins in BINS:
            rows.append({
                "configuration_id": _cid(gamma, bins), "autoencoder_seed": 1,
                "architecture_id": "h64_32", "mi_gamma": gamma, "mi_sensitive_num_bins": bins,
                "feasible": gamma != 0.9, "is_pareto_front": gamma == 0.1 and bins == 50,
                "pareto_rank": 1.0 if gamma == 0.1 and bins == 50 else float("nan"),
                **{col: 0.1 + gamma + bins / 1000 for col, *_ in METRICS if col != "redundancy_bits"},
            })
    return pd.DataFrame(rows)


def _study(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Candidates table, study map with paths of another machine, local checkpoints."""
    root = tmp_path / "Pareto-Front-Test"
    (root / "phase3").mkdir(parents=True)
    candidates = root / "phase3" / "pareto_candidates.csv"
    _candidates().to_csv(candidates, index=False)
    checkpoints = tmp_path / "checkpoints"
    lines = ["runs:"]
    for gamma in GAMMAS:
        for bins in BINS:
            run = f"Seed1_Gamma_{gamma}_Bins_{bins}_architecture_h64_32_Run01"
            data = checkpoints / "Pareto-Front-Test" / run / "plots/mi_diagnostics/data"
            for epoch in ("epoch_0000", "epoch_0199"):
                (data / epoch).mkdir(parents=True)
                pd.DataFrame({"bin_id": np.arange(EFFECTIVE[bins])}).to_csv(
                    data / epoch / f"mi_bin_widths_{epoch.replace('_', '')}.csv", index=False)
            lines += [f"- configuration_id: {_cid(gamma, bins)}",
                      f"  checkpoint_run_dir: /eos/x/checkpoints/Pareto-Front-Test/{run}"]
    study_map = root / "study_map.yaml"
    study_map.write_text("\n".join(lines) + "\n")
    return candidates, study_map, checkpoints


def test_effective_bins_come_from_the_local_run_checkpoints(tmp_path):
    _, study_map, checkpoints = _study(tmp_path)
    ids = [_cid(0.1, b) for b in BINS]
    assert effective_bins_by_configuration(study_map, checkpoints, ids) == {
        _cid(0.1, b): float(EFFECTIVE[b]) for b in BINS
    }


def test_missing_effective_bins_fail_loudly(tmp_path):
    _, study_map, checkpoints = _study(tmp_path)
    with pytest.raises(ParetoMiChangeError, match="No effective bin count"):
        effective_bins_by_configuration(study_map, checkpoints / "nowhere", [_cid(0.1, 10)])


def test_sweeps_hold_the_other_parameter_fixed():
    table = _candidates()
    effective = {_cid(g, b): EFFECTIVE[b] for g in GAMMAS for b in BINS}
    gamma_sweep, bin_sweep = build_sweeps(table, bins_for_gamma=50, gamma_for_bins=0.1,
                                          effective_bins=effective)
    assert list(gamma_sweep.table["mi_gamma"]) == list(GAMMAS)
    assert set(gamma_sweep.table["mi_sensitive_num_bins"]) == {50}
    assert "48 effective" in gamma_sweep.fixed_text
    assert list(bin_sweep.table[EFFECTIVE_BINS]) == [10, 48, 64]
    assert set(bin_sweep.table["mi_gamma"]) == {0.1}
    assert list(bin_sweep.baseline["mi_gamma"]) == [0.0, 0.0, 0.0]
    assert bin_mapping_text(bin_sweep) == "Nominal → effective bins: 10→10, 50→48, 100→64"


def test_writes_one_png_per_metric_per_sweep_overviews_and_csvs(tmp_path):
    candidates, study_map, checkpoints = _study(tmp_path)
    out = tmp_path / "mi-changes"
    written = write_mi_changes(candidates, out, study_map=study_map, checkpoints_root=checkpoints)

    n_metrics = len([m for m in METRICS])
    assert len(written) == 2 * (n_metrics + 1)
    assert {p.parent.name for p in written if p.parent != out} == {"gamma", "effective_bins"}
    assert (out / "00_overview_gamma.png").is_file()
    assert (out / "00_overview_effective_bins.png").is_file()
    assert all(p.stat().st_size > 10_000 for p in written)
    gamma_csv = pd.read_csv(out / "sweep_gamma.csv")
    bins_csv = pd.read_csv(out / "sweep_effective_bins.csv")
    assert list(gamma_csv["mi_gamma"]) == list(GAMMAS)
    assert list(bins_csv[EFFECTIVE_BINS]) == [10, 48, 64]


def test_without_checkpoints_only_the_gamma_sweep_is_drawn(tmp_path):
    candidates, _, _ = _study(tmp_path)
    written = write_mi_changes(candidates, tmp_path / "out")
    assert {p.parent.name for p in written} == {"gamma", "out"}


def test_rejected_runs_break_the_line_and_are_open_red(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    from src.evaluation.pareto_matrices import CRITICAL
    from src.evaluation.pareto_mi_changes import SERIES, draw_metric

    (gamma_sweep,) = build_sweeps(_candidates(), bins_for_gamma=50)
    fig, ax = plt.subplots()
    draw_metric(ax, gamma_sweep, "leakage_worst", "lower", "{:.3f}")
    line = next(l for l in ax.lines if to_rgba(l.get_color()) == to_rgba(SERIES) and l.get_linestyle() == "-")
    ydata = np.asarray(line.get_ydata(), dtype=float)
    assert np.isnan(ydata[-1]) and np.isfinite(ydata[:-1]).all()  # γ 0.9 is rejected
    rejected = [l for l in ax.lines if to_rgba(l.get_markeredgecolor()) == to_rgba(CRITICAL)]
    assert len(rejected) == 1 and list(rejected[0].get_xdata()) == [0.9]
    plt.close(fig)
