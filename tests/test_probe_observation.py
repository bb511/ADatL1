"""Loss plots, numerical histories, and live progress must be observation-only."""

import json
import logging
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image
from matplotlib import rc_context, rcParams
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

from src.evaluation.leakage_probe import (
    MLP_PROBE_CONFIG,
    ShuffledTargetMLPResult,
    evaluate_four_leakage_probes,
    extract_probe_split,
    fit_mlp_probe_candidate,
    four_probe_result_payload,
    make_probe_inner_partition,
    write_leakage_probe_results,
)
from src.evaluation.leakage_probe.diagnostics import enforce_shuffled_target_guardrail
from src.evaluation.leakage_probe.errors import ShuffledTargetGuardrailError
from src.evaluation.leakage_probe.persistence import write_invalid_leakage_probe_result
import src.evaluation.leakage_probe.plotting as plotting
from tests.test_linear_leakage_probes import make_representation_set
from tests.test_probe_representation_extraction import FakeProbeDataModule, RecordingProbeModel, make_batch


@pytest.fixture(scope="module")
def observed_result():
    rng = np.random.default_rng(80)

    def representations(split, n):
        features = rng.normal(size=(n, 3))
        return replace(
            make_representation_set(split, n),
            latent_logits=features,
            latent_sample=(features > 0).astype(float),
            reconstructed_data=np.column_stack([features, features**2]),
            sensitive_target=100.0 + 20 * features[:, 0] + rng.normal(scale=5, size=n),
        )

    train = representations("train", 200)
    valid = representations("valid", 80)
    result = evaluate_four_leakage_probes(train, valid, run_shuffled_target_controls=False)
    return result, train, valid


def _assert_pngs(payload, output_path, expected_count):
    entries = list(payload["probes"].values())
    controls = payload["diagnostics"]["shuffled_targets"]
    if controls["enabled"]:
        entries.extend([controls["z_logits"], controls["reconstruction"]])
    paths = []
    for probe in entries:
        assert probe["loss_plot"]["status"] == "created"
        path = output_path.parent / probe["loss_plot"]["path"]
        assert path.is_file()
        with Image.open(path) as image:
            assert image.format == "PNG"
            assert image.width >= 900 and image.height >= 600
        paths.append(path)
    assert len(set(paths)) == expected_count
    return paths


def test_each_of_four_probes_links_a_real_plot_and_preserves_histories(tmp_path, observed_result):
    result, train, valid = observed_result
    path = write_leakage_probe_results(result, tmp_path)
    payload = json.loads(path.read_text())
    summary_path = path.with_name("leakage_probes_summary.json")
    summary = json.loads(summary_path.read_text())
    _assert_pngs(payload, path, expected_count=4)
    assert payload["leakage_worst"] == result.leakage_worst
    assert summary == {
        "evaluation": {
            "development_event_manifest_hash": "train-manifest",
            "held_out_event_manifest_hash": "valid-manifest",
            "mode": "validation",
            "purpose": "scientific",
            "reporting_eligible": True,
        },
        "leakage_probe_protocol_version": "fet-et-four-probe-v6",
        "leakage_probe_summary_schema_version": 1,
        "leakage_worst": result.leakage_worst,
        "probe_valid": True,
        "probes": {
            name: {"r2_clipped": payload["probes"][name]["r2_clipped"]}
            for name in (
                "mlp/z_logits",
                "mlp/reconstruction",
                "linear/z_logits",
                "linear/reconstruction",
            )
        },
        "rejection_reason": None,
        "run": {
            "autoencoder_seed": None,
            "configuration_id": None,
        },
        "source_artifact": "leakage_probes.json",
        "worst_probe": result.worst_probe,
    }
    for key, probe in [("mlp/z_logits", result.mlp_latent_logits), ("mlp/reconstruction", result.mlp_reconstructed_data)]:
        history = payload["probes"][key]["training_history"]
        assert history["epochs"] == list(range(1, probe.outer_result.n_iter + 1))
        assert history["loss"] == probe.outer_result.estimator.loss_curve_
        assert history["early_stopping_validation_r2"] == probe.outer_result.estimator.validation_scores_
        assert history["loss_units"] == "dimensionless"
        assert "not the held-out split" in history["validation_scope"]
        for candidate_payload, candidate in zip(payload["probes"][key]["seed_selection"]["candidates"], probe.seed_selection.successful_candidates):
            assert candidate_payload["training_history"]["loss"] == candidate.estimator.loss_curve_
    for key, probe in [("linear/z_logits", result.linear_latent_logits), ("linear/reconstruction", result.linear_reconstructed_data)]:
        summary = payload["probes"][key]["loss_summary"]
        assert summary["method"] == "direct_least_squares"
        assert summary["epochs"] is None
        assert "training_history" not in payload["probes"][key]
        for pool, metric_key in [(train, "development_mse_gev2"), (valid, "held_out_mse_gev2")]:
            outer = probe.outer_result
            predictions = outer.estimator.predict(outer.feature_scaler.transform(getattr(pool, probe.representation_name)))
            assert summary[metric_key] == pytest.approx(mean_squared_error(pool.sensitive_target, predictions))


def test_smoke_plots_cannot_overwrite_scientific_plots(tmp_path, observed_result):
    result, _, _ = observed_result
    scientific_path = write_leakage_probe_results(result, tmp_path)
    scientific_payload = json.loads(scientific_path.read_text())
    scientific_summary = scientific_path.with_name("leakage_probes_summary.json")
    scientific_plots = _assert_pngs(scientific_payload, scientific_path, 4)
    original_bytes = {
        path: path.read_bytes()
        for path in [scientific_path, scientific_summary, *scientific_plots]
    }
    context = result.evaluation_context
    smoke = replace(result, evaluation_context=replace(
        context,
        development_data=replace(context.development_data, max_samples=200),
        held_out_data=replace(context.held_out_data, max_samples=80),
    ))
    smoke_path = write_leakage_probe_results(smoke, tmp_path)
    smoke_payload = json.loads(smoke_path.read_text())
    smoke_plots = _assert_pngs(smoke_payload, smoke_path, 4)
    assert smoke_path.name == "leakage_probes_smoke.json"
    smoke_summary_path = smoke_path.with_name("leakage_probes_smoke_summary.json")
    smoke_summary = json.loads(smoke_summary_path.read_text())
    assert smoke_summary["source_artifact"] == "leakage_probes_smoke.json"
    assert smoke_summary["evaluation"]["purpose"] == "smoke_test"
    assert smoke_summary["evaluation"]["reporting_eligible"] is False
    assert set(scientific_plots).isdisjoint(smoke_plots)
    assert all(path.read_bytes() == content for path, content in original_bytes.items())


def test_guardrail_rejection_keeps_all_probe_and_control_plots(tmp_path, observed_result):
    result, _, _ = observed_result
    control = replace(result.mlp_latent_logits, outer_result=replace(result.mlp_latent_logits.outer_result, outer_r2_raw=0.9, outer_r2_clipped=0.9))
    result = replace(result, shuffled_target_controls=ShuffledTargetMLPResult(
        latent_logits=control,
        reconstructed_data=result.mlp_reconstructed_data,
        inner_partition=result.inner_partition,
        shuffle_seed=12345,
        permutation_manifest_hash="test-permutation",
    ))
    with pytest.raises(ShuffledTargetGuardrailError) as error:
        enforce_shuffled_target_guardrail(result)
    path = write_invalid_leakage_probe_result(tmp_path, error.value)
    payload = json.loads(path.read_text())
    summary = json.loads(path.with_name("leakage_probes_summary.json").read_text())
    _assert_pngs(payload, path, expected_count=6)
    assert payload["probe_valid"] is False
    assert payload["leakage_worst"] is None
    assert summary["probe_valid"] is False
    assert summary["rejection_reason"] == "shuffled_target_guardrail_failed"
    assert summary["worst_probe"] is None
    assert summary["leakage_worst"] is None
    assert summary["probes"]["mlp/z_logits"]["r2_clipped"] == payload[
        "probes"
    ]["mlp/z_logits"]["r2_clipped"]


def test_plot_failure_is_visible_without_losing_numerical_result(tmp_path, monkeypatch, caplog, observed_result):
    def fail(*args):
        raise OSError("synthetic image write failure")
    monkeypatch.setattr(plotting, "_plot_mlp", fail)
    path = write_leakage_probe_results(observed_result[0], tmp_path)
    payload = json.loads(path.read_text())
    assert payload["probe_valid"] is True
    assert payload["probes"]["mlp/z_logits"]["loss_plot"] == {
        "status": "failed", "path": None, "error": "synthetic image write failure",
    }
    assert payload["probes"]["linear/z_logits"]["loss_plot"]["status"] == "created"
    assert "Could not save mlp/z_logits loss plot" in caplog.text


def test_verbose_mlp_fit_is_numerically_identical_and_reports_progress(caplog, capsys):
    rng = np.random.default_rng(42)
    features = rng.normal(size=(100, 3))
    target = 40 + 4 * features[:, 0]
    partition = make_probe_inner_partition(len(target))
    caplog.set_level(logging.INFO)
    result = fit_mlp_probe_candidate(features, target, partition, seed=10)
    quiet = MLPRegressor(**MLP_PROBE_CONFIG, random_state=10)
    quiet.fit(
        result.feature_scaler.transform(features[partition.fit_indices]),
        result.target_scaler.transform(target[partition.fit_indices, None]).reshape(-1),
    )
    np.testing.assert_array_equal(result.loss_curve, quiet.loss_curve_)
    for recorded_weights, quiet_weights in zip(result.estimator.coefs_, quiet.coefs_):
        np.testing.assert_array_equal(recorded_weights, quiet_weights)
    assert "MLP candidate seed=10 starting" in caplog.text
    assert "epochs=" in caplog.text and "inner score: R2=" in caplog.text
    stdout = capsys.readouterr().out
    assert "Iteration 1, loss =" in stdout
    assert "Validation score:" in stdout


def test_extraction_logs_split_caps_and_incremental_event_counts(caplog):
    caplog.set_level(logging.INFO)
    datamodule = FakeProbeDataModule([make_batch(offset=float(i)) for i in range(26)])
    extract_probe_split(RecordingProbeModel(), datamodule, "train", max_samples=100)
    assert "Extracting probe split=train" in caplog.text
    assert "cap=100" in caplog.text
    assert "batch=1," in caplog.text and "batch=25," in caplog.text
    assert "Probe extraction train finished" in caplog.text


def test_nonfinite_history_is_serialized_as_null_not_invalid_json(observed_result):
    result = observed_result[0]
    probe = result.mlp_latent_logits
    changed = replace(probe, outer_result=replace(probe.outer_result, loss_curve=(1.0, float("nan")), early_stopping_validation_scores=(float("nan"), 0.2)))
    payload = four_probe_result_payload(replace(result, mlp_latent_logits=changed))
    history = payload["probes"]["mlp/z_logits"]["training_history"]
    assert history["loss"] == [1.0, None]
    assert history["early_stopping_validation_r2"] == [None, 0.2]
    json.dumps(payload, allow_nan=False)


def test_plot_style_is_isolated_from_other_evaluation_callbacks(tmp_path, monkeypatch, observed_result):
    original = plotting._plot_mlp
    def check_style(*args):
        assert rcParams["font.size"] == 10
        assert rcParams["axes.labelsize"] != 40
        return original(*args)
    monkeypatch.setattr(plotting, "_plot_mlp", check_style)
    with rc_context({"font.size": 40, "axes.labelsize": 40}):
        path = write_leakage_probe_results(observed_result[0], tmp_path)
        assert rcParams["font.size"] == 40
        assert rcParams["axes.labelsize"] == 40
    _assert_pngs(json.loads(path.read_text()), path, 4)
