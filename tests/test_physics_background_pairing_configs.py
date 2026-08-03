"""Configuration contracts for the physics background-pairing campaign."""

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
    assert cfg.callbacks.residual_oas_state.max_samples == 163840
    assert cfg.callbacks.anomaly_eff.output_name == "ascore/full"
    assert cfg.evaluation.callbacks.anomaly_efficiency.output_name == "ascore/full"


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
        enabled = {
            "cap": cfg.callbacks.cap_sn_zb is not None,
            "wasserstein": cfg.callbacks.wasserstein_dist is not None,
            "thres_drift": cfg.callbacks.thres_drift is not None,
        }
        assert enabled[callback_name]
        assert sum(enabled.values()) == 1
