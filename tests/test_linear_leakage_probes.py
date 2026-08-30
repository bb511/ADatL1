import json
from unittest.mock import Mock

import numpy as np
import pytest

import src.evaluation.leakage_probe.evaluation as leakage_probe_evaluation
import src.evaluation.leakage_probe.persistence as leakage_probe_persistence
from src.evaluation.leakage_probe import (
    PROBE_INITIALIZATION_SEEDS,
    FourProbeEvaluationResult,
    LinearProbeOuterResult,
    MLPProbeCandidateResult,
    MLPProbeOuterResult,
    MLPProbeSeedSelection,
    NamedLinearProbeResult,
    NamedMLPProbeResult,
    PrimaryLinearProbeResult,
    PrimaryMLPLeakageResult,
    ProbeExtractionError,
    ProbeFitError,
    ProbeRepresentationSet,
    evaluate_and_write_loss_total_leakage_probes,
    evaluate_four_leakage_probes,
    evaluate_linear_probe_representation,
    evaluate_primary_linear_probes,
    fit_linear_probe,
    make_probe_inner_partition,
    write_leakage_probe_results,
    PROBE_TARGET_SHUFFLE_SEED,
    ShuffledTargetMLPResult,
)


def make_arrays():
    train_x = np.arange(
        20,
        dtype=np.float64,
    )
    validation_x = np.arange(
        20,
        26,
        dtype=np.float64,
    )

    train_features = np.column_stack(
        [train_x, train_x**2]
    )
    validation_features = np.column_stack(
        [validation_x, validation_x**2]
    )

    train_target = 100.0 + 10.0 * train_x
    validation_target = 100.0 + 10.0 * validation_x

    return (
        train_features,
        train_target,
        validation_features,
        validation_target,
    )


def make_representation_set(
    split: str,
    n_events: int,
) -> ProbeRepresentationSet:
    target = np.linspace(
        80.0,
        180.0,
        num=n_events,
    )

    latent = np.column_stack(
        [target, np.sin(target)]
    )
    reconstruction = np.column_stack(
        [target, target**2, np.cos(target)]
    )

    return ProbeRepresentationSet(
        split=split,
        latent_logits=latent,
        latent_sample=(latent > 0).astype(float),
        reconstructed_data=reconstruction,
        sensitive_target=target,
        n_events=n_events,
        sample_seed=12345,
        max_samples=None,
        manifest_hash=f"{split}-manifest",
    )


def make_selection(seed: int) -> MLPProbeSeedSelection:
    candidates = tuple(
        MLPProbeCandidateResult(
            seed=candidate_seed,
            inner_r2_raw=(
                0.5
                if candidate_seed == seed
                else 0.1
            ),
            inner_mae_gev=(
                5.0
                if candidate_seed == seed
                else 6.0
            ),
            convergence_warnings=(),
            n_iter=5,
            final_loss=0.1,
            feature_scaler=Mock(),
            target_scaler=Mock(),
            estimator=Mock(),
        )
        for candidate_seed in PROBE_INITIALIZATION_SEEDS
    )

    selected_candidate = next(
        candidate
        for candidate in candidates
        if candidate.seed == seed
    )

    return MLPProbeSeedSelection(
        selected_seed=seed,
        selected_candidate=selected_candidate,
        successful_candidates=candidates,
        failed_candidates=(),
    )


def make_named_mlp(
    representation_name: str,
    seed: int,
    clipped_r2: float,
) -> NamedMLPProbeResult:
    metric_names = {
        "latent_logits": "z_logits",
        "reconstructed_data": "reconstruction",
    }
    metric_name = metric_names[representation_name]

    outer = MLPProbeOuterResult(
        selected_seed=seed,
        outer_r2_raw=clipped_r2,
        outer_r2_clipped=clipped_r2,
        outer_mae_gev=5.0,
        convergence_warnings=(),
        n_iter=5,
        final_loss=0.1,
        n_train=20,
        n_validation=10,
        feature_scaler=Mock(),
        target_scaler=Mock(),
        estimator=Mock(),
    )

    return NamedMLPProbeResult(
        representation_name=representation_name,
        metric_name=metric_name,
        feature_dimension=2,
        seed_selection=make_selection(seed),
        outer_result=outer,
    )

def make_shuffled_controls() -> ShuffledTargetMLPResult:
    return ShuffledTargetMLPResult(
        latent_logits=make_named_mlp(
            "latent_logits",
            10,
            0.0,
        ),
        reconstructed_data=make_named_mlp(
            "reconstructed_data",
            123,
            0.0,
        ),
        inner_partition=make_probe_inner_partition(20),
        shuffle_seed=PROBE_TARGET_SHUFFLE_SEED,
        permutation_manifest_hash=(
            "test-shuffle-manifest"
        ),
    )

def make_named_linear(
    representation_name: str,
    clipped_r2: float,
) -> NamedLinearProbeResult:
    metric_name = (
        "z_logits"
        if representation_name == "latent_logits"
        else "reconstruction"
    )

    outer = LinearProbeOuterResult(
        outer_r2_raw=clipped_r2,
        outer_r2_clipped=clipped_r2,
        outer_mae_gev=6.0,
        n_train=20,
        n_validation=10,
        feature_scaler=Mock(),
        estimator=Mock(),
    )

    return NamedLinearProbeResult(
        representation_name=representation_name,
        metric_name=metric_name,
        feature_dimension=2,
        outer_result=outer,
    )


def test_linear_probe_fits_scaler_on_train_only() -> None:
    (
        train_features,
        train_target,
        validation_features,
        validation_target,
    ) = make_arrays()

    result = fit_linear_probe(
        train_features,
        train_target,
        validation_features,
        validation_target,
    )

    np.testing.assert_allclose(
        result.feature_scaler.mean_,
        train_features.mean(axis=0),
    )

    combined_mean = np.concatenate(
        [train_features, validation_features],
        axis=0,
    ).mean(axis=0)

    assert not np.allclose(
        result.feature_scaler.mean_,
        combined_mean,
    )


def test_linear_probe_reports_physical_outer_metrics() -> None:
    result = fit_linear_probe(*make_arrays())

    assert result.outer_r2_raw == pytest.approx(1.0)
    assert result.outer_r2_clipped == pytest.approx(1.0)
    assert result.outer_mae_gev == pytest.approx(
        0.0,
        abs=1e-10,
    )
    assert result.n_train == 20
    assert result.n_validation == 6


def test_negative_linear_r2_is_preserved_and_clipped() -> None:
    (
        train_features,
        train_target,
        validation_features,
        validation_target,
    ) = make_arrays()

    validation_target = validation_target[::-1].copy()

    result = fit_linear_probe(
        train_features,
        train_target,
        validation_features,
        validation_target,
    )

    assert result.outer_r2_raw < 0.0
    assert result.outer_r2_clipped == 0.0


def test_linear_feature_dimension_mismatch_is_rejected() -> None:
    (
        train_features,
        train_target,
        validation_features,
        validation_target,
    ) = make_arrays()

    with pytest.raises(ProbeFitError) as error:
        fit_linear_probe(
            train_features,
            train_target,
            validation_features[:, :1],
            validation_target,
        )

    assert (
        error.value.reason
        == "linear_feature_dimension_mismatch"
    )


def test_named_linear_probe_uses_requested_representation() -> None:
    train = make_representation_set("train", 20)
    validation = make_representation_set("valid", 10)

    result = evaluate_linear_probe_representation(
        train,
        validation,
        representation_name="latent_logits",
    )

    assert result.representation_name == "latent_logits"
    assert result.metric_name == "z_logits"
    assert result.feature_dimension == 2


def test_latent_sample_is_not_a_main_linear_probe() -> None:
    train = make_representation_set("train", 20)
    validation = make_representation_set("valid", 10)

    with pytest.raises(ProbeFitError) as error:
        evaluate_linear_probe_representation(
            train,
            validation,
            representation_name="latent_sample",
        )

    assert (
        error.value.reason
        == "unknown_linear_probe_representation"
    )


def test_two_linear_probes_have_independent_state() -> None:
    train = make_representation_set("train", 20)
    validation = make_representation_set("valid", 10)

    result = evaluate_primary_linear_probes(
        train,
        validation,
    )

    latent = result.latent_logits.outer_result
    reconstruction = (
        result.reconstructed_data.outer_result
    )

    assert latent.estimator is not reconstruction.estimator
    assert (
        latent.feature_scaler
        is not reconstruction.feature_scaler
    )


@pytest.mark.parametrize(
    ("train_split", "validation_split", "reason"),
    [
        (
            "valid",
            "valid",
            "invalid_linear_probe_training_split",
        ),
        (
            "train",
            "test",
            "invalid_linear_probe_outer_split",
        ),
    ],
)
def test_linear_pair_enforces_outer_split_protocol(
    train_split: str,
    validation_split: str,
    reason: str,
) -> None:
    train = make_representation_set(train_split, 20)
    validation = make_representation_set(
        validation_split,
        10,
    )

    with pytest.raises(ProbeFitError) as error:
        evaluate_primary_linear_probes(
            train,
            validation,
        )

    assert error.value.reason == reason


@pytest.mark.parametrize(
    ("scores", "expected_worst_probe"),
    [
        ((0.9, 0.2, 0.3, 0.4), "mlp/z_logits"),
        ((0.2, 0.9, 0.3, 0.4), "mlp/reconstruction"),
        ((0.2, 0.3, 0.9, 0.4), "linear/z_logits"),
        ((0.2, 0.3, 0.4, 0.9), "linear/reconstruction"),
    ],
)
def test_each_of_four_probes_can_determine_leakage_worst(
    monkeypatch,
    scores: tuple[float, float, float, float],
    expected_worst_probe: str,
) -> None:
    train = make_representation_set("train", 20)
    validation = make_representation_set("valid", 10)
    (
        mlp_latent_score,
        mlp_reconstruction_score,
        linear_latent_score,
        linear_reconstruction_score,
    ) = scores

    mlp_latent = make_named_mlp(
        "latent_logits",
        10,
        mlp_latent_score,
    )
    mlp_reconstruction = make_named_mlp(
        "reconstructed_data",
        123,
        mlp_reconstruction_score,
    )

    mlp_result = PrimaryMLPLeakageResult(
        latent_logits=mlp_latent,
        reconstructed_data=mlp_reconstruction,
        inner_partition=make_probe_inner_partition(20),
        leakage_worst=max(
            mlp_latent_score,
            mlp_reconstruction_score,
        ),
    )

    linear_latent = make_named_linear(
        "latent_logits",
        linear_latent_score,
    )
    linear_reconstruction = make_named_linear(
        "reconstructed_data",
        linear_reconstruction_score,
    )

    linear_result = PrimaryLinearProbeResult(
        latent_logits=linear_latent,
        reconstructed_data=linear_reconstruction,
    )

    monkeypatch.setattr(
        leakage_probe_evaluation,
        "evaluate_primary_mlp_probes",
        Mock(return_value=mlp_result),
    )
    monkeypatch.setattr(
        leakage_probe_evaluation,
        "evaluate_primary_linear_probes",
        Mock(return_value=linear_result),
    )
    shuffled_controls = make_shuffled_controls()

    monkeypatch.setattr(
        leakage_probe_evaluation,
        "evaluate_shuffled_target_mlp_controls",
        Mock(return_value=shuffled_controls),
    )
    result = evaluate_four_leakage_probes(
        train,
        validation,
    )

    assert isinstance(result, FourProbeEvaluationResult)

    probes = (
        result.mlp_latent_logits,
        result.mlp_reconstructed_data,
        result.linear_latent_logits,
        result.linear_reconstructed_data,
    )

    assert len(probes) == 4
    assert result.mlp_latent_logits is mlp_latent
    assert result.mlp_reconstructed_data is mlp_reconstruction
    assert result.linear_latent_logits is linear_latent
    assert (
        result.linear_reconstructed_data
        is linear_reconstruction
    )

    assert result.worst_probe == expected_worst_probe
    assert result.leakage_worst == pytest.approx(max(scores))


def test_four_probe_results_are_written_to_required_path(
    monkeypatch,
    tmp_path,
) -> None:
    train = make_representation_set("train", 20)
    validation = make_representation_set("valid", 10)

    mlp_result = PrimaryMLPLeakageResult(
        latent_logits=make_named_mlp(
            "latent_logits",
            10,
            0.2,
        ),
        reconstructed_data=make_named_mlp(
            "reconstructed_data",
            123,
            0.7,
        ),
        inner_partition=make_probe_inner_partition(20),
        leakage_worst=0.7,
    )

    linear_result = PrimaryLinearProbeResult(
        latent_logits=make_named_linear(
            "latent_logits",
            0.8,
        ),
        reconstructed_data=make_named_linear(
            "reconstructed_data",
            0.3,
        ),
    )

    monkeypatch.setattr(
        leakage_probe_evaluation,
        "evaluate_primary_mlp_probes",
        Mock(return_value=mlp_result),
    )
    monkeypatch.setattr(
        leakage_probe_evaluation,
        "evaluate_primary_linear_probes",
        Mock(return_value=linear_result),
    )
    shuffled_controls = make_shuffled_controls()

    monkeypatch.setattr(
        leakage_probe_evaluation,
        "evaluate_shuffled_target_mlp_controls",
        Mock(return_value=shuffled_controls),
    )
    result = evaluate_four_leakage_probes(
        train,
        validation,
    )

    run_folder = (
        tmp_path
        / "checkpoints"
        / "experiment_name"
        / "run_name"
    )

    output_path = write_leakage_probe_results(
        result,
        run_folder,
    )

    expected_path = (
        run_folder
        / "plots"
        / "val"
        / "loss_total"
        / "probes"
        / "leakage_probes.json"
    )

    assert output_path == expected_path
    assert output_path.is_file()

    payload = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert set(payload["probes"]) == {
        "mlp/z_logits",
        "mlp/reconstruction",
        "linear/z_logits",
        "linear/reconstruction",
    }

    assert payload["worst_probe"] == "linear/z_logits"
    assert payload["leakage_worst"] == pytest.approx(0.8)

    assert payload["probes"]["mlp/z_logits"][
        "r2_clipped"
    ] == pytest.approx(0.2)

    assert payload["probes"]["mlp/reconstruction"][
        "r2_clipped"
    ] == pytest.approx(0.7)

    assert payload["probes"]["linear/z_logits"][
        "r2_clipped"
    ] == pytest.approx(0.8)

    assert payload["probes"]["linear/reconstruction"][
        "r2_clipped"
    ] == pytest.approx(0.3)

    assert payload["probe_valid"] is True
    assert payload["rejection_reason"] is None
    assert payload["rejection_message"] is None

    assert set(payload["diagnostics"]) == {
        "shuffled_targets",
    }
    assert set(
        payload["diagnostics"]["shuffled_targets"]
    ) == {
        "shuffle_seed",
        "permutation_manifest_hash",
        "z_logits",
        "reconstruction",
    }

    # Diagnostics must never be mixed into the four primary probes.
    assert len(payload["probes"]) == 4


def test_loss_total_orchestrator_loads_extracts_and_writes(
    monkeypatch,
    tmp_path,
) -> None:
    checkpoint_path = tmp_path / "loss_total.ckpt"
    checkpoint_path.touch()
    state_dict = {"encoder.weight": object()}

    model = Mock()
    datamodule = Mock()
    train_representations = make_representation_set(
        "train",
        20,
    )
    validation_representations = make_representation_set(
        "valid",
        10,
    )
    expected_result = Mock(spec=FourProbeEvaluationResult)
    expected_output_path = (
        tmp_path
        / "plots"
        / "val"
        / "loss_total"
        / "probes"
        / "leakage_probes.json"
    )

    load_mock = Mock(return_value={"state_dict": state_dict})
    extract_mock = Mock(
        side_effect=[
            train_representations,
            validation_representations,
        ]
    )
    evaluate_mock = Mock(return_value=expected_result)
    write_mock = Mock(return_value=expected_output_path)

    monkeypatch.setattr(
        leakage_probe_persistence.torch,
        "load",
        load_mock,
    )
    monkeypatch.setattr(
        leakage_probe_persistence,
        "extract_probe_split",
        extract_mock,
    )
    monkeypatch.setattr(
        leakage_probe_persistence,
        "evaluate_four_leakage_probes",
        evaluate_mock,
    )
    monkeypatch.setattr(
        leakage_probe_persistence,
        "write_leakage_probe_results",
        write_mock,
    )

    result, output_path = (
        evaluate_and_write_loss_total_leakage_probes(
            model,
            datamodule,
            tmp_path,
            device="cpu",
        )
    )

    assert result is expected_result
    assert output_path == expected_output_path
    load_mock.assert_called_once_with(
        checkpoint_path,
        weights_only=False,
        map_location="cpu",
    )
    model.load_state_dict.assert_called_once_with(
        state_dict,
        strict=True,
    )
    assert extract_mock.call_count == 2
    assert extract_mock.call_args_list[0].args == (
        model,
        datamodule,
        "train",
    )
    assert extract_mock.call_args_list[0].kwargs == {
        "device": "cpu"
    }
    assert extract_mock.call_args_list[1].args == (
        model,
        datamodule,
        "valid",
    )
    assert extract_mock.call_args_list[1].kwargs == {
        "device": "cpu"
    }
    evaluate_mock.assert_called_once_with(
        train_representations,
        validation_representations,
    )
    write_mock.assert_called_once_with(
        expected_result,
        tmp_path,
    )


def test_loss_total_orchestrator_requires_checkpoint(
    tmp_path,
) -> None:
    with pytest.raises(ProbeExtractionError) as error:
        evaluate_and_write_loss_total_leakage_probes(
            Mock(),
            Mock(),
            tmp_path,
        )

    assert error.value.reason == "loss_total_checkpoint_missing"
