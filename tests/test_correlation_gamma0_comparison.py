"""Layout split (self_improvement/) and the comparison with the γ = 0 run."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from src.analysis import correlation_gamma0_comparison as cmp  # noqa: E402

LABELS = ["jets.phi", "FET.Et", "jets.Et"]
METHOD_DIR = "plots/val/loss_total/correlation_matrix/normal/Pearson"


def _corr(fet_jets_phi: float, fet_jets_et: float) -> pd.DataFrame:
    return pd.DataFrame(
        [[1.0, fet_jets_phi, 0.3], [fet_jets_phi, 1.0, fet_jets_et], [0.3, fet_jets_et, 1.0]],
        index=LABELS, columns=LABELS,
    )


def _png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), "white").save(path)


def _run(experiment: Path, name: str, *, gamma: float, seed: int = 1, epochs: int = 50,
         bins: int = 50, reco: pd.DataFrame | None = None, old_layout: bool = False) -> Path:
    run = experiment / name
    run.mkdir(parents=True)
    (run / "resolved_config.yaml").write_text(
        f"seed: {seed}\ntrainer:\n  max_epochs: {epochs}\nalgorithm:\n  mi_gamma: {gamma}\n"
        f"  mi_sensitive_num_bins: {bins}\n  encoder:\n    nodes: [64, 32, 8]\n"
    )
    method_dir = run / METHOD_DIR
    method_dir.mkdir(parents=True)
    if reco is not None:
        _corr(0.5, 0.5).to_csv(method_dir / "input_pearson_correlation_matrix.csv")
        reco.to_csv(method_dir / "reconstruction_pearson_correlation_matrix.csv")
        _png(method_dir / "input_pearson_correlation_matrix.png")
        _png(method_dir / "reconstruction_pearson_correlation_matrix.png")
        stem = "abs_reconstruction_minus_input_pearson_correlation_matrix"
        target = method_dir if old_layout else method_dir / "self_improvement"
        for suffix in ("", "_et_only", "_sorted_by_increase", "_sorted_by_decrease_et_only"):
            _png(target / f"{stem}{suffix}.png")
    return run


@pytest.fixture
def plot_calls(monkeypatch):
    calls = []

    def fake_plot(**kwargs):
        calls.append(kwargs)
        _png(Path(kwargs["save_dir"]) / kwargs["filename"])

    monkeypatch.setattr("src.plot.matrix.plot", fake_plot)
    return calls


@pytest.fixture
def experiment(tmp_path):
    exp = tmp_path / "checkpoints" / "Exp"
    _run(exp, "G0_bins10", gamma=0.0, bins=10, reco=_corr(0.4, 0.2))
    _run(exp, "G0_bins50", gamma=0.0, bins=50, reco=_corr(0.4, 0.2))
    _run(exp, "G0_local3ep", gamma=0.0, epochs=3, reco=_corr(0.9, 0.9))
    _run(exp, "RunA", gamma=0.1, bins=80, reco=_corr(0.3, -0.6), old_layout=True)
    _run(exp, "RunOtherSeed", gamma=0.1, seed=2, reco=_corr(0.1, 0.1))
    return exp


def test_comparison_uses_a_gamma0_run_with_same_seed_architecture_and_epochs(experiment, plot_calls):
    report = cmp.process_experiment(experiment)

    out = experiment / "RunA" / METHOD_DIR / "comparison_gamma0"
    stem = "abs_reconstruction_minus_gamma0_reconstruction_pearson_correlation_matrix"
    assert sorted(p.name for p in out.glob("*.png")) == sorted(
        f"{stem}{d}{s}.png" for d in ("", "_sorted_by_increase", "_sorted_by_decrease")
        for s in ("", "_et_only")
    )
    reference = json.loads((out / "reference.json").read_text())
    assert reference["reference_run"] == "G0_bins10"  # bins ignored, first by name
    assert reference["reference_csv"] == f"G0_bins10/{METHOD_DIR}/reconstruction_pearson_correlation_matrix.csv"
    assert reference["matched_on"] == {"gamma": 0.0, "seed": 1, "encoder_nodes": [64, 32, 8],
                                       "max_epochs": 50}
    copy = pd.read_csv(out / "gamma0_reconstruction_pearson_correlation_matrix.csv", index_col=0)
    assert np.allclose(copy.to_numpy(), _corr(0.4, 0.2).to_numpy())

    # |r_reco(run)| - |r_reco(γ=0)|; green where |r_reco(run)| < |r_reco(γ=0)|:
    # jets.phi (0.3 < 0.4, although 0.3 > 0.1), not jets.Et (0.6 > 0.2), not the
    # diagonal (1 = 1, "closer" is strict).
    full = next(c for c in plot_calls if c["filename"] == f"{stem}.png")
    assert full["data"]["FET.Et"]["jets.phi"] == pytest.approx(0.3 - 0.4)
    assert full["data"]["FET.Et"]["jets.Et"] == pytest.approx(0.6 - 0.2)
    assert full["text_highlight_columns"] == ["jets.phi"]
    assert full["value_name"].startswith("Change vs γ = 0 in Pearson correlation")
    assert full["subtitle"] == "MI: γ = 0.1 · requested bins = 80 · effective bins = n/a"

    assert [run for _, run in report.written] == ["G0_bins10"]
    assert report.skipped["γ = 0 run"] == 3
    assert report.skipped["no γ = 0 run with the same seed, architecture and epochs"] == 1
    for name in ("G0_bins10", "G0_bins50", "RunOtherSeed"):
        assert not (experiment / name / METHOD_DIR / "comparison_gamma0").exists()


def test_existing_comparisons_are_kept_unless_forced(experiment, plot_calls):
    cmp.process_experiment(experiment)
    n = len(plot_calls)
    again = cmp.process_experiment(experiment)
    assert len(plot_calls) == n and again.skipped["comparison exists (use --force)"] == 1
    cmp.process_experiment(experiment, force=True)
    assert len(plot_calls) == 2 * n


def test_migration_moves_self_improvement_plots_out_of_the_method_folder(experiment, plot_calls):
    report = cmp.process_experiment(experiment, migrate=True)
    method_dir = experiment / "RunA" / METHOD_DIR
    assert sorted(p.name for p in method_dir.glob("*.png")) == [
        "input_pearson_correlation_matrix.png", "reconstruction_pearson_correlation_matrix.png"]
    assert len(list((method_dir / "self_improvement").glob("*.png"))) == 4
    assert report.moved == 4


def test_galleries_follow_the_evaluator_layout(experiment, plot_calls, tmp_path):
    mlruns = tmp_path / "mlruns"
    (mlruns / "7").mkdir(parents=True)
    (mlruns / "7" / "meta.yaml").write_text("name: Exp\n")
    for run_id, role in (("a" * 32, "primary training run of checkpoint"),
                         ("b" * 32, "other training attempt (checkpoint belongs to a)")):
        tags = mlruns / "7" / run_id / "tags"
        tags.mkdir(parents=True)
        (tags / "mlflow.runName").write_text("RunA")
        (tags / "link.role").write_text(role)
        (tags.parent / "meta.yaml").write_text("lifecycle_stage: active\n")

    cmp.process_experiment(experiment, migrate=True, mlruns_root=mlruns)

    artifacts = mlruns / "7" / ("a" * 32) / "artifacts" / "val" / "loss_total" / "correlation_matrix"
    main = (artifacts / "normal_correlation_matrix_pearson.html").read_text()
    assert "alt='reconstruction_pearson_correlation_matrix'" in main
    assert "abs_reconstruction" not in main
    assert (artifacts / "Pearson" / "normal_correlation_matrix_pearson_self_improvement.html").is_file()
    comparison = (artifacts / "Pearson" / "normal_correlation_matrix_pearson_comparison_gamma0.html")
    assert comparison.read_text().count("<div class='card'>") == 6
    assert not (mlruns / "7" / ("b" * 32) / "artifacts").exists()


def test_reference_disagreement_is_reported(experiment, plot_calls):
    (_corr(0.41, 0.2)).to_csv(
        experiment / "G0_bins50" / METHOD_DIR / "reconstruction_pearson_correlation_matrix.csv")
    report = cmp.process_experiment(experiment)
    assert any("γ = 0 runs disagree; used G0_bins10" in note for _, note in report.details)


def test_redo_redraws_comparison_plots_from_their_folder(experiment, plot_calls):
    from src.analysis.scripts import redo_correlation_matrix_plots as redo

    cmp.process_experiment(experiment)
    plot_calls.clear()
    out = experiment / "RunA" / METHOD_DIR / "comparison_gamma0"
    jobs = redo.discover_plot_jobs(out)
    results, _ = redo.redraw_all(jobs)
    assert [r.status for r in results] == ["redrawn"] * 6
    full = next(c for c in plot_calls if not c["filename"].endswith(("_et_only.png",))
                and "sorted" not in c["filename"])
    assert full["data"]["FET.Et"]["jets.phi"] == pytest.approx(0.3 - 0.4)
    assert {tuple(c["text_highlight_columns"]) for c in plot_calls} == {("jets.phi",)}


def test_redo_finds_sources_of_self_improvement_plots_in_the_method_folder(experiment, plot_calls):
    from src.analysis.scripts import redo_correlation_matrix_plots as redo

    self_dir = experiment / "G0_bins10" / METHOD_DIR / "self_improvement"
    results, _ = redo.redraw_all(redo.discover_plot_jobs(self_dir))
    assert [r.status for r in results] == ["redrawn"] * 4
    # Self-improvement plots keep the |r_reco| <= 0.1 rule (G0 reco: 0.4, 0.2).
    assert all(c["text_highlight_columns"] == [] for c in plot_calls)


def test_gallery_regex_maps_subfolder_galleries(tmp_path):
    from src.analysis.scripts import redo_correlation_matrix_plots as redo

    run_dir = tmp_path / "mlruns" / "7" / "abc"
    gallery = (run_dir / "artifacts/val/loss_total/correlation_matrix/Pearson"
               / "normal_correlation_matrix_pearson_comparison_gamma0.html")
    assert redo.gallery_plot_dir(gallery, run_dir, tmp_path / "ck") == (
        tmp_path / "ck/plots/val/loss_total/correlation_matrix/normal/Pearson/comparison_gamma0")


def test_closer_to_zero_columns_is_strict_and_sign_blind():
    from src.plot.correlation_matrix import closer_to_zero_columns

    labels = ["a", "FET.Et", "b", "c", "d", "e"]
    run = pd.DataFrame(0.0, index=labels, columns=labels)
    reference = run.copy()
    run.loc["FET.Et"] = [-0.2, 1.0, 0.5, 0.3, np.nan, 0.0]
    reference.loc["FET.Et"] = [0.3, 1.0, -0.4, -0.3, 0.9, 0.1]
    # a: 0.2 < 0.3; FET.Et and c equal; b: 0.5 > 0.4; d: NaN; e: 0 < 0.1.
    assert closer_to_zero_columns(run, reference) == ["a", "e"]
    assert closer_to_zero_columns(run, reference.drop(columns="a")) == ["e"]
    assert closer_to_zero_columns(run.rename(index={"FET.Et": "fet.et"}), reference) == ["a", "e"]
    assert closer_to_zero_columns(run, None) == []
    assert closer_to_zero_columns(run.drop(index="FET.Et"), reference) == []


def test_green_columns_override_the_decorrelation_reference(plot_calls, tmp_path):
    from src.plot import correlation_matrix as corr_plot

    corr = _corr(0.05, 0.6)
    corr_plot.plot_correlation_matrix(corr, tmp_path, "x.png", "t", decorrelation_reference=corr)
    corr_plot.plot_correlation_matrix(corr, tmp_path, "y.png", "t", decorrelation_reference=corr,
                                      green_columns=["jets.Et"])
    corr_plot.plot_correlation_matrix(corr, tmp_path, "z.png", "t", decorrelation_reference=corr,
                                      green_columns=[])
    assert [c["text_highlight_columns"] for c in plot_calls] == [["jets.phi"], ["jets.Et"], []]
