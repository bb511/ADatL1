"""src/analysis/run_vs_gamma0_comparison.py: test outputs of a run vs its γ = 0 / 50-bin run."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.analysis import run_vs_gamma0_comparison as cmp  # noqa: E402

SIGNALS = ("sigA", "sigB", "sigC")
EDGES = np.array([0.0, 1.0, 2.0, 3.0])


def _hist(input_counts, reco_counts, edges=EDGES) -> pd.DataFrame:
    """As src/evaluation/callbacks/reco.py writes it: underflow, bins, overflow."""
    return pd.DataFrame({"bin_low": np.concatenate([[-np.inf], edges]),
                         "bin_high": np.concatenate([edges, [np.inf]]),
                         "input": input_counts, "reco": reco_counts})


def _run(experiment: Path, name: str, *, gamma: float, bins: int = 50, seed: int = 1,
         eff=(0.1, 0.02, 0.0), ascore=(10.0, 50.0), drift=0.2, w1=0.005,
         reco=(0, 50, 30, 20, 0), split="test", outputs=True) -> Path:
    run = experiment / name
    run.mkdir(parents=True)
    (run / "resolved_config.yaml").write_text(
        f"seed: {seed}\ntrainer:\n  max_epochs: 200\nalgorithm:\n  mi_gamma: {gamma}\n"
        f"  mi_sensitive_num_bins: {bins}\n  encoder:\n    nodes: [64, 32, 8]\n")
    (run / "plots").mkdir()
    if not outputs:
        return run
    split_dir = run / "plots" / split
    ckpt = split_dir / "loss_total"
    (ckpt / "eff").mkdir(parents=True)
    per_signal = dict(zip(SIGNALS, eff))
    (ckpt / "eff" / "eff_summary.json").write_text(json.dumps({
        "signal_efficiencies": per_signal, "median_efficiency": float(np.median(eff)),
        "min_efficiency": min(eff), "mean_efficiency": float(np.mean(eff)),
        "cvar25_efficiency": min(eff)}))
    for folder, payload in (("ascore_operational_summary",
                             {"loss_total": {"normal": ascore[0], "sigA": ascore[1]}}),
                            ("thres_drift_summary", {"operational": {"loss_total": drift}}),
                            ("wasserstein_summary", {"loss_total": w1})):
        (split_dir / folder).mkdir(parents=True)
        with open(split_dir / folder / "summary.pkl", "wb") as handle:
            pickle.dump(payload, handle)
    data = ckpt / "reco" / "normal" / "data"
    data.mkdir(parents=True)
    _hist([1, 40, 40, 19, 0], list(reco)).to_csv(data / "jets_Et.csv", index=False)
    return run


@pytest.fixture
def experiment(tmp_path):
    exp = tmp_path / "Exp"
    _run(exp, "G0_bins50", gamma=0.0, eff=(0.1, 0.02, 0.0), ascore=(10.0, 50.0), drift=0.2, w1=0.005,
         reco=(0, 50, 30, 20, 0))
    # γ = 0 at another bin count, with different outputs: must never be the reference.
    _run(exp, "G0_bins10", gamma=0.0, bins=10, eff=(0.9, 0.9, 0.9), drift=9.0, w1=9.0)
    _run(exp, "RunA", gamma=0.1, bins=40, eff=(0.12, 0.01, 0.0), ascore=(11.0, 60.0), drift=0.1,
         w1=0.006, reco=(0, 40, 40, 19, 1))
    _run(exp, "RunOtherSeed", gamma=0.1, seed=2)
    _run(exp, "RunUntested", gamma=0.2, outputs=False)
    return exp


def _summary(experiment, run="RunA"):
    out = experiment / run / "plots" / "test" / "loss_total" / "comparison"
    return out, pd.read_csv(out / "summary.csv").set_index("quantity")


def test_quantities_are_compared_with_the_50_bin_gamma0_run(experiment):
    report = cmp.process_experiment(experiment, run_names=["RunA"])
    assert report.written == [("RunA", "G0_bins50", [])]
    out, summary = _summary(experiment)

    median = summary.loc["median_efficiency"]
    assert (median["run"], median["gamma0"]) == (pytest.approx(0.01), pytest.approx(0.02))
    assert median["change_percent"] == pytest.approx(-50.0)
    assert str(median["better"]) == "False"  # higher is better
    drift = summary.loc["drift_operational"]
    assert drift["change_percent"] == pytest.approx(-50.0) and str(drift["better"]) == "True"
    w1 = summary.loc["W1(normal, SingleNeutrino_E-10-gun)"]
    assert w1["change_percent"] == pytest.approx(20.0) and str(w1["better"]) == "False"
    assert np.isnan(summary.loc["min_efficiency", "change_percent"])  # 0 vs 0: undefined

    per_signal = pd.read_csv(out / "efficiency" / "efficiency_per_signal.csv").set_index("quantity")
    assert per_signal.loc["sigA", "change_percent"] == pytest.approx(20.0)
    assert str(per_signal.loc["sigA", "better"]) == "True"
    ascore = pd.read_csv(out / "ascore_operational" / "mean_ascore_per_dataset.csv").set_index("quantity")
    assert ascore.loc["sigA", "change_percent"] == pytest.approx(20.0)
    assert ascore["direction"].isna().all() and ascore["better"].isna().all()  # no better/worse

    for png in ("summary.png", "efficiency/efficiency_per_signal.png",
                "ascore_operational/mean_ascore_per_dataset.png", "reco/normal/jets_Et.png"):
        assert (out / png).stat().st_size > 0, png
    reference = json.loads((out / "reference.json").read_text())
    assert reference["reference_run"] == "G0_bins50"
    assert reference["matched_on"]["mi_sensitive_num_bins"] == 50
    # Correlation matrices are not part of this comparison.
    assert not any("correlation" in str(p) for p in out.rglob("*"))


def test_reconstruction_histograms_are_differenced_bin_by_bin(experiment):
    cmp.process_experiment(experiment, run_names=["RunA"])
    out, _ = _summary(experiment)
    table = pd.read_csv(out / "reco" / "normal" / "jets_Et.csv")
    assert list(table["reco_run"]) == [0, 40, 40, 19, 1]
    assert list(table["reco_gamma0"]) == [0, 50, 30, 20, 0]
    assert table["difference_fraction"].tolist() == pytest.approx([0, -0.1, 0.1, -0.01, 0.01])
    reco = pd.read_csv(out / "reco" / "reco_summary.csv").iloc[0]
    assert reco["same_bins"]
    assert reco["tvd_reco_run_vs_reco_gamma0"] == pytest.approx(0.11)
    assert reco["tvd_reco_run_vs_input"] == pytest.approx(0.01)
    assert reco["tvd_reco_gamma0_vs_input"] == pytest.approx(0.11)


def test_different_bins_are_overlaid_without_a_difference(experiment):
    data = experiment / "RunA" / "plots" / "test" / "loss_total" / "reco" / "normal" / "data"
    _hist([1, 40, 40, 19, 0], [0, 40, 40, 19, 1], edges=EDGES * 2).to_csv(data / "jets_Et.csv", index=False)
    cmp.process_experiment(experiment, run_names=["RunA"])
    out, _ = _summary(experiment)
    assert not (out / "reco" / "normal" / "jets_Et.csv").exists()
    assert (out / "reco" / "normal" / "jets_Et.png").is_file()
    assert not pd.read_csv(out / "reco" / "reco_summary.csv").iloc[0]["same_bins"]


def test_runs_that_cannot_be_compared_are_reported(experiment):
    report = cmp.process_experiment(
        experiment, run_names=["G0_bins50", "RunOtherSeed", "RunUntested", "Nope"])
    assert report.written == []
    reasons = dict(report.skipped)
    assert reasons["G0_bins50"] == "γ = 0 run (the reference)"
    assert reasons["RunOtherSeed"].startswith("no γ = 0 / 50-bin run")
    assert reasons["RunUntested"] == "no test outputs"
    assert reasons["Nope"] == "no such run in Exp"


def test_untested_gamma0_run_is_named(tmp_path):
    exp = tmp_path / "Exp"
    _run(exp, "G0_bins50", gamma=0.0, outputs=False)
    _run(exp, "RunA", gamma=0.1)
    report = cmp.process_experiment(exp)
    assert dict(report.skipped)["RunA"].startswith("its γ = 0 / 50-bin run has no test outputs")


def test_missing_quantities_are_listed(experiment):
    for folder in ("thres_drift_summary", "wasserstein_summary"):
        (experiment / "G0_bins50" / "plots" / "test" / folder / "summary.pkl").unlink()
    report = cmp.process_experiment(experiment, run_names=["RunA"])
    assert report.written == [("RunA", "G0_bins50", ["thres_drift", "wasserstein"])]
    out, summary = _summary(experiment)
    assert "drift_operational" not in summary.index
    assert json.loads((out / "summary.json").read_text())["missing"] == ["thres_drift", "wasserstein"]


def test_cli(experiment, capsys):
    assert cmp.main(["--experiment-dir", str(experiment), "--run-name", "RunA",
                     "--run-name", "G0_bins50"]) == 0
    printed = capsys.readouterr().out
    assert "1 run(s) compared" in printed and "RunA vs G0_bins50" in printed
    assert "skipped G0_bins50: γ = 0 run (the reference)" in printed
