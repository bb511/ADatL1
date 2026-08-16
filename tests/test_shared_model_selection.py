"""Unit contracts for shared-pool checkpoint and score selection."""

from types import SimpleNamespace

import pytest
import torch

from src.callbacks.efficiency import AnomalyEfficiencyCallback as TrainingEfficiency
from src.evaluation.callbacks.efficiency import (
    AnomalyEfficiencyCallback as EvaluationEfficiency,
)
from src.evaluation.evaluator import Evaluator, _register_checkpoint_threshold_buffers


def test_fixed_relative_change_rule_respects_weight_and_direction() -> None:
    """Q-prime receives weight 1.25 and Q-double-prime weight 1.0."""
    assert Evaluator._contender_wins([1.1, 1.05], [1.0, 1.0], ["maximize", "minimize"])
    assert not Evaluator._contender_wins([1.01, 1.02], [1.0, 1.0], ["maximize", "minimize"])


def test_each_view_accepts_exactly_last_primary_and_stable_native() -> None:
    """No unrelated metric checkpoint may enter a logical strategy's selector."""
    config = {
        "primary_checkpoint_metric": "cap_native_cdf_ema_A_vs_B",
        "main_metric": {"direction": "maximize"},
    }
    assert Evaluator._candidate_enabled(config, "last")
    assert Evaluator._candidate_enabled(config, "single/ascore_operational/stable")
    assert Evaluator._candidate_enabled(config, "summary/cap_native_cdf_ema_A_vs_B/max")
    assert not Evaluator._candidate_enabled(config, "summary/cap_native_jetclr_ema_A_vs_B/max")


def test_flattened_objectives_preserve_config_insertion_order() -> None:
    """Hydra and campaign vector indices rely on deterministic insertion order."""
    evaluator = object.__new__(Evaluator)
    evaluator.optimized_metric_configs = {"second": {}, "first": {}}
    evaluator.optimized_metrics = {"second": [2.0, 3.0], "first": [0.0, 1.0]}
    evaluator.optimized_metric = None
    evaluator.optimized_ckpt_name = "stale"
    evaluator._sync_flattened_optimized_metric()
    assert evaluator.optimized_metric == (2.0, 3.0, 0.0, 1.0)
    assert evaluator.optimized_ckpt_name is None


def test_shared_selection_persists_canonical_checkpoint_family() -> None:
    """Metric-local dataset names must not leak into the checkpoint identity."""
    evaluator = object.__new__(Evaluator)
    evaluator.optimized_metrics = {"cdf": None}
    evaluator.optimized_ckpt_names = {"cdf": None}
    evaluator._get_metric = lambda *_: ("normal", 2.0)
    evaluator._get_secondary_metric = lambda *_: (3.0, "minimize")
    config = {
        "main_metric": {
            "callback": {"name": "cap", "params": {}},
            "direction": "maximize",
        },
        "sec_metric": {"callback": {"name": "efficiency"}},
    }

    evaluator._update_named_optimized_metric("cdf", config, "single/ascore_operational/stable")

    assert evaluator.optimized_metrics["cdf"] == [2.0, 3.0]
    assert evaluator.optimized_ckpt_names["cdf"] == "single/ascore_operational/stable"


def test_primary_candidate_runs_only_its_qprime_and_shared_qdoubleprime() -> None:
    """Shared evaluation avoids recomputing every expensive CAP for every checkpoint."""
    evaluator = object.__new__(Evaluator)
    evaluator.set_optimized_metric = True
    evaluator.optimized_metric_configs = {
        "cdf": {
            "primary_checkpoint_metric": "cap_cdf",
            "main_metric": {
                "callback": {"name": "cdf"},
                "direction": "maximize",
            },
            "sec_metric": {"callback": {"name": "native"}},
        },
        "jetclr": {
            "primary_checkpoint_metric": "cap_jetclr",
            "main_metric": {
                "callback": {"name": "jetclr"},
                "direction": "maximize",
            },
            "sec_metric": {"callback": {"name": "native"}},
        },
    }
    callbacks = [
        SimpleNamespace(name="cdf"),
        SimpleNamespace(name="jetclr"),
        SimpleNamespace(name="native"),
    ]
    evaluator._all_evaluator_callbacks = callbacks
    evaluator.evaluator = SimpleNamespace(
        strat_name="summary",
        metric_name="cap_cdf",
        criterion_name="max",
        callbacks=list(callbacks),
    )

    evaluator._activate_candidate_callbacks()

    assert [callback.name for callback in evaluator.evaluator.callbacks] == [
        "cdf",
        "native",
    ]


def test_oas_operating_threshold_has_an_independent_checkpoint_buffer() -> None:
    """OAS downstream efficiency must not reuse the native-score threshold."""
    callback = TrainingEfficiency(
        output_name="ascore/residual_oas",
        threshold_namespace="residual_oas",
        metric_suffix="residual_oas",
    )
    callback.module_target_rate = 0.25
    module = torch.nn.Module()
    callback._set_thres_on_module(module, 0.25, torch.tensor(3.5))
    assert module.thres_operational__residual_oas.item() == pytest.approx(3.5)
    assert not hasattr(module, "thres_operational")

    evaluator_callback = EvaluationEfficiency(
        output_name="ascore/residual_oas",
        ds=[],
        threshold_namespace="residual_oas",
    )
    evaluator_callback.target_rates_resolved = [0.25]
    evaluator_callback.operational_rate = 0.25
    thresholds = evaluator_callback._get_thres(module)
    assert thresholds[0.25].item() == pytest.approx(3.5)


def test_threshold_buffers_exist_before_checkpointing_and_restore_per_score() -> None:
    """Native and OAS thresholds are persistent before the first checkpoint save."""

    class Module(torch.nn.Module):
        """Minimal module exposing the rate configuration needed by callbacks."""

        def __init__(self):
            """Initialize the callback-facing hyperparameters."""
            super().__init__()
            self.hparams = SimpleNamespace(target_rate=0.25, base_rate=28608.8064)

        @property
        def device(self):
            """Return the CPU device used by this unit contract."""
            return torch.device("cpu")

    module = Module()
    native = TrainingEfficiency(output_name="ascore/full")
    oas = TrainingEfficiency(
        output_name="ascore/residual_oas",
        threshold_namespace="residual_oas",
        metric_suffix="residual_oas",
    )

    native.on_fit_start(None, module)
    oas.on_fit_start(None, module)

    assert set(module.state_dict()) == {
        "thres_operational",
        "thres_operational__residual_oas",
    }
    assert torch.isnan(module.thres_operational)
    assert torch.isnan(module.thres_operational__residual_oas)

    native._set_thres_on_module(module, 0.25, torch.tensor(1.25))
    oas._set_thres_on_module(module, 0.25, torch.tensor(3.5))
    restored = Module()
    native.on_fit_start(None, restored)
    oas.on_fit_start(None, restored)
    restored.load_state_dict(module.state_dict(), strict=True)

    assert restored.thres_operational.item() == pytest.approx(1.25)
    assert restored.thres_operational__residual_oas.item() == pytest.approx(3.5)


def test_evaluation_only_model_registers_exact_checkpoint_threshold_buffers() -> None:
    """Known saved thresholds are declared before a fresh model's strict load."""
    module = torch.nn.Linear(2, 1)
    state = {
        **module.state_dict(),
        "thres_operational": torch.tensor(1.25),
        "thres_operational__residual_oas": torch.tensor(3.5),
    }

    _register_checkpoint_threshold_buffers(module, state)
    module.load_state_dict(state, strict=True)

    assert module.thres_operational.item() == pytest.approx(1.25)
    assert module.thres_operational__residual_oas.item() == pytest.approx(3.5)


@pytest.mark.parametrize("value", [torch.tensor(float("nan")), torch.ones(2)])
def test_evaluation_only_threshold_restoration_rejects_malformed_values(value) -> None:
    """Non-finite or non-scalar checkpoint thresholds fail closed."""
    module = torch.nn.Linear(2, 1)

    with pytest.raises(ValueError, match="Checkpoint threshold"):
        _register_checkpoint_threshold_buffers(module, {"thres_operational": value})


def test_evaluation_only_threshold_restoration_keeps_other_state_strict() -> None:
    """Dynamic threshold support does not relax unrelated checkpoint state."""
    module = torch.nn.Linear(2, 1)
    state = {**module.state_dict(), "unexpected": torch.tensor(1.0)}

    _register_checkpoint_threshold_buffers(module, state)

    with pytest.raises(RuntimeError, match="Unexpected key"):
        module.load_state_dict(state, strict=True)


def test_invalid_threshold_namespace_is_rejected() -> None:
    """Buffer namespaces are deliberately limited to state-dict-safe names."""
    with pytest.raises(ValueError, match="Namespaces"):
        TrainingEfficiency(output_name="ascore/full", threshold_namespace="bad/name")
