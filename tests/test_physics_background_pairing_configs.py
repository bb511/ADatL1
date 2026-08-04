"""Configuration contracts for the physics background-pairing campaign."""

from scripts import physics_background_pairing_campaign as campaign
from src.utils.pairing.io import compose_config

MODELS = ("ae", "vae", "dsae", "dsvae", "svdd", "realnvp")
SELECTION_METRICS = {
    "cap_mapping": ("cap", "maximize"),
    "wasserstein": ("wasserstein", "minimize"),
    "drift": ("thres_drift", "minimize"),
}


def test_all_paper_models_compose_with_background_selection_streams(monkeypatch, tmp_path) -> None:
    """Every paper model must select on the same named held-out sources."""
    monkeypatch.setenv(
        "PHYSICS_PAIRING_DIR",
        str(tmp_path / "ZB_run396102_to_ZB_run398183"),
    )
    for model in MODELS:
        cfg = compose_config(
            overrides=[
                f"experiment=physics/{model}_background_pairing",
                "physics_pairing.strategy=jetclr",
            ]
        )
        assert cfg.data.expose_zerobias_sources is True
        assert cfg.callbacks.cap_sn_zb.dataset_1 == "ZB_run396102"
        assert cfg.callbacks.cap_sn_zb.dataset_2 == "ZB_run398183"
        assert cfg.callbacks.thres_drift.dataset_1 == "ZB_run396102"
        assert cfg.callbacks.thres_drift.dataset_2 == "ZB_run398183"
        assert cfg.callbacks.wasserstein_dist.dataset_1 == "ZB_run396102"
        assert cfg.callbacks.wasserstein_dist.dataset_2 == "ZB_run398183"
        assert cfg.callbacks.cap_sn_zb.pairing_index_path.endswith(
            "validate_jetclr_cap_n163840.pt"
        )
        assert cfg.evaluation.callbacks.cap_sn_zb.pairing_test_index_path.endswith(
            "test_jetclr_cap_n163840.pt"
        )


def test_physics_ae_installs_bounded_train_only_mahalanobis_state() -> None:
    """Physics AE must fit bounded train-only OAS state for its canonical score."""
    cfg = compose_config(overrides=["experiment=physics/ae_background_pairing"])
    assert cfg.algorithm.anomaly_score == "residual_oas"
    assert cfg.callbacks.residual_oas_state.max_samples == 163840
    assert cfg.callbacks.anomaly_eff.output_name == "ascore/full"
    assert cfg.evaluation.callbacks.anomaly_efficiency.output_name == "ascore/full"


def test_physics_vae_routes_bounded_train_only_oas_as_canonical_score() -> None:
    """All generic VAE selection metrics must consume the configured OAS score."""
    cfg = compose_config(overrides=["experiment=physics/vae_background_pairing"])
    assert cfg.algorithm.anomaly_score == "residual_oas"
    assert cfg.callbacks.vae_residual_state.max_samples == 163840
    assert cfg.callbacks.cap_sn_zb.output_name == "ascore/full"
    assert cfg.callbacks.wasserstein_dist.output_name == "ascore/full"
    assert cfg.callbacks.thres_drift.output_name == "ascore/full"


def test_cdf_control_uses_background_domains_without_pair_tables() -> None:
    """CDF must change only CAP pairing while retaining background0/background1."""
    cfg = compose_config(
        overrides=[
            "experiment=physics/ae_background_pairing",
            "pairing=physics_cdf",
        ]
    )
    assert cfg.callbacks.cap_sn_zb.dataset_1 == "ZB_run396102"
    assert cfg.callbacks.cap_sn_zb.dataset_2 == "ZB_run398183"
    assert cfg.physics_pairing.validation_table is None
    assert cfg.physics_pairing.test_table is None
    assert cfg.callbacks.cap_sn_zb.pairing_type == "cdf"
    assert cfg.callbacks.cap_sn_zb.pairing_index_path is None
    assert cfg.evaluation.callbacks.cap_sn_zb.pairing_test_index_path is None


def test_every_shared_model_study_composes() -> None:
    """All six master studies must contain the full ordered objective vector."""
    for study in campaign._studies():
        model = study["model"]
        overrides = [
            f"experiment=physics/{model}_background_all",
            f"hparams_search={model}_shared_optuna",
        ]
        cfg = compose_config(overrides=overrides)
        assert cfg.optimized_metric_artifact is True
        assert list(cfg.optimized_metric_configs) == [
            objective["id"] for objective in study["objectives"]
        ]
        assert [
            direction
            for config in cfg.optimized_metric_configs.values()
            for direction in (
                config.main_metric.direction,
                config.sec_metric.direction,
            )
        ] == [
            direction for objective in study["objectives"] for direction in objective["directions"]
        ]
        assert cfg.data.validation_aux_datasets == []
        assert cfg.logger.mlflow is not None


def test_shared_ae_and_vae_keep_native_training_score() -> None:
    """OAS supplies extra Q-prime views but never replaces native Q-double-prime."""
    ae = compose_config(overrides=["experiment=physics/ae_background_all"])
    vae = compose_config(overrides=["experiment=physics/vae_background_all"])
    assert ae.algorithm.anomaly_score == "mse"
    assert vae.algorithm.anomaly_score == "kl_raw"
    assert ae.physics_selection.native_output == "ascore/full"
    assert vae.physics_selection.native_output == "ascore/full"
    assert len(ae.optimized_metric_configs) == 14
    assert len(vae.optimized_metric_configs) == 14


def test_shared_suite_exposes_all_four_tables_and_cdf_on_both_scores() -> None:
    """AE/VAE native and OAS views must use identical background pairings."""
    cfg = compose_config(overrides=["experiment=physics/ae_background_all"])
    for score in ("native", "residual_oas"):
        for strategy in campaign.CAP_STRATEGIES:
            callback = cfg.callbacks[f"cap_{score}_{strategy}"]
            assert callback.dataset_1 == "ZB_run396102"
            assert callback.dataset_2 == "ZB_run398183"
            if strategy == "cdf":
                assert callback.pairing_type == "cdf"
                assert callback.get("pairing_index_path") is None
            else:
                assert callback.pairing_type == "mapping"
                assert callback.pairing_index_path.endswith(f"validate_{strategy}_cap_n163840.pt")
                evaluation = cfg.evaluation.callbacks[f"cap_{score}_{strategy}"]
                assert evaluation.pairing_test_index_path.endswith(
                    f"test_{strategy}_cap_n163840.pt"
                )


def test_retrain_suite_restores_20_signals_with_score_specific_thresholds() -> None:
    """Native and OAS downstream efficiencies use separately calibrated thresholds."""
    cfg = compose_config(
        overrides=[
            "experiment=physics/ae_background_all",
            "+retrain_suite=physics_background_native_oas",
        ]
    )
    assert cfg.test is True
    assert cfg.callbacks.anomaly_eff.output_name == "ascore/full"
    assert cfg.callbacks.anomaly_eff_oas.output_name == "ascore/residual_oas"
    assert cfg.callbacks.anomaly_eff_oas.threshold_namespace == "residual_oas"
    assert len(cfg.evaluation.callbacks.anomaly_efficiency.ds) == 20
    assert len(cfg.evaluation.callbacks.anomaly_efficiency_oas.ds) == 20
    assert cfg.evaluation.callbacks.anomaly_efficiency.name == "eff_native"
    assert cfg.evaluation.callbacks.anomaly_efficiency_oas.name == "eff_residual_oas"


def test_search_overlays_retain_only_requested_primary_metric() -> None:
    """Each search must evaluate only its requested primary selection metric."""
    for overlay, (callback_name, direction) in SELECTION_METRICS.items():
        cfg = compose_config(
            overrides=[
                "experiment=physics/vae_background_pairing",
                f"+selection_metric={overlay}",
            ]
        )
        assert cfg.optimized_metric_config.main_metric.callback.name == callback_name
        assert cfg.optimized_metric_config.main_metric.direction == direction
        assert cfg.evaluation.evaluator.ckpts.last is True
        enabled = {
            "cap": cfg.callbacks.cap_sn_zb is not None,
            "wasserstein": cfg.callbacks.wasserstein_dist is not None,
            "thres_drift": cfg.callbacks.thres_drift is not None,
        }
        assert enabled[callback_name]
        assert sum(enabled.values()) == 1


def test_retrain_overlays_restore_downstream_only_after_selection() -> None:
    """Frozen-Pareto retraining must restore all downstream physics signals."""
    for overlay in SELECTION_METRICS:
        cfg = compose_config(
            overrides=[
                "experiment=physics/vae_background_pairing",
                f"+selection_metric={overlay}_retrain",
            ]
        )
        assert cfg.test is True
        assert cfg.callbacks.anomaly_eff is None
        assert cfg.evaluation.callbacks.anomaly_efficiency is not None
        assert len(cfg.evaluation.callbacks.anomaly_efficiency.ds) == 20
