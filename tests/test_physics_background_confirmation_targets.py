"""Composition contracts for paper-style two-objective confirmation targets."""

import pytest

from src import train as train_module
from src.utils.pairing.io import compose_config

NATIVE_TARGETS = {
    "native_cap_flat_physical": ("cap_native_flat_physical", "maximize"),
    "native_cap_physics_summary": ("cap_native_physics_summary", "maximize"),
    "native_cap_typed_sliced_wasserstein": (
        "cap_native_typed_sliced_wasserstein",
        "maximize",
    ),
    "native_cap_jetclr": ("cap_native_jetclr", "maximize"),
    "native_cap_cdf": ("cap_native_cdf", "maximize"),
    "native_wasserstein": ("wasserstein_native", "minimize"),
    "native_drift": ("drift_native", "minimize"),
}
OAS_TARGETS = {
    "residual_oas_cap_flat_physical": (
        "cap_residual_oas_flat_physical",
        "maximize",
    ),
    "residual_oas_cap_physics_summary": (
        "cap_residual_oas_physics_summary",
        "maximize",
    ),
    "residual_oas_cap_typed_sliced_wasserstein": (
        "cap_residual_oas_typed_sliced_wasserstein",
        "maximize",
    ),
    "residual_oas_cap_jetclr": ("cap_residual_oas_jetclr", "maximize"),
    "residual_oas_cap_cdf": ("cap_residual_oas_cdf", "maximize"),
    "residual_oas_wasserstein": ("wasserstein_residual_oas", "minimize"),
    "residual_oas_drift": ("drift_residual_oas", "minimize"),
}


def _active_checkpoint_candidates(cfg) -> list[tuple[str, str, object]]:
    """Return non-null checkpoint candidates from one composed target."""
    candidates = []
    for group, entries in cfg.evaluation.evaluator.ckpts.items():
        if isinstance(entries, bool):
            if entries:
                candidates.append((str(group), str(group), entries))
            continue
        for name, value in entries.items():
            if value is not None:
                candidates.append((str(group), str(name), value))
    return candidates


def _assert_target(overlay: str, callback: str, direction: str, *, model: str, baseline) -> None:
    """Assert the exact objective and candidate contract for one overlay."""
    cfg = compose_config(
        overrides=[
            f"experiment=physics/{model}_background_all",
            f"+confirmation_target={overlay}",
        ]
    )

    assert cfg.optimized_metric_configs is None
    objective = cfg.optimized_metric_config
    assert objective.main_metric.callback.name == callback
    assert objective.main_metric.callback.params == {}
    assert objective.main_metric.direction == direction
    assert objective.sec_metric.callback.name == "ascore_operational"
    assert objective.sec_metric.direction == "minimize"

    candidates = _active_checkpoint_candidates(cfg)
    assert len(candidates) == 3
    assert ("last", "last", True) in candidates
    assert ("single", "ascore_operational", ["stable"]) in candidates
    primary = [candidate for candidate in candidates if candidate[0] == "summary"]
    assert primary == [
        (
            "summary",
            objective.primary_checkpoint_metric,
            ["max" if direction == "maximize" else "min"],
        )
    ]

    # Targeting changes selection, not what is measured and logged each epoch.
    assert cfg.callbacks == baseline.callbacks
    assert cfg.evaluation.callbacks == baseline.evaluation.callbacks


@pytest.mark.parametrize("model", ("ae", "vae", "dsae", "dsvae", "svdd", "realnvp"))
def test_all_seven_native_confirmation_targets_compose_for_every_model(model: str) -> None:
    """Every model must compose each native paper-style target exactly."""
    baseline = compose_config(overrides=[f"experiment=physics/{model}_background_all"])
    for overlay, (callback, direction) in NATIVE_TARGETS.items():
        _assert_target(overlay, callback, direction, model=model, baseline=baseline)


@pytest.mark.parametrize("model", ("ae", "vae"))
def test_all_seven_oas_confirmation_targets_compose_for_ae_and_vae(model: str) -> None:
    """AE and VAE must compose every masked residual-OAS target exactly."""
    baseline = compose_config(overrides=[f"experiment=physics/{model}_background_all"])
    for overlay, (callback, direction) in OAS_TARGETS.items():
        _assert_target(overlay, callback, direction, model=model, baseline=baseline)


def test_confirmation_target_with_paper_hpo_returns_exactly_two_objectives() -> None:
    """Paper HPO plus a target must expose one Q-prime/Q-double-prime pair."""
    cfg = compose_config(
        overrides=[
            "experiment=physics/ae_background_all",
            "hparams_search=ae_optuna",
            "+confirmation_target=native_cap_cdf",
        ]
    )
    assert cfg.optimized_metric_configs is None
    assert cfg.optimized_metric_config.main_metric.direction == "maximize"
    assert cfg.optimized_metric_config.sec_metric.direction == "minimize"
    assert train_module._get_directions(cfg) == ["maximize", "minimize"]
