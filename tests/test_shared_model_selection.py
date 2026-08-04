"""Unit contracts for shared-pool checkpoint and score selection."""

from types import SimpleNamespace

import pytest
import torch

from src.callbacks.efficiency import AnomalyEfficiencyCallback as TrainingEfficiency
from src.evaluation.callbacks.efficiency import (
    AnomalyEfficiencyCallback as EvaluationEfficiency,
)
from src.evaluation.evaluator import Evaluator


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


def test_invalid_threshold_namespace_is_rejected() -> None:
    """Buffer namespaces are deliberately limited to state-dict-safe names."""
    with pytest.raises(ValueError, match="Namespaces"):
        TrainingEfficiency(output_name="ascore/full", threshold_namespace="bad/name")
