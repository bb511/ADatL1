"""Tests for the six shared physics background-pairing studies."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import yaml

SCRIPT = Path(__file__).parents[1] / "scripts/physics_background_pairing_campaign.py"
SPEC = importlib.util.spec_from_file_location("physics_background_pairing_campaign", SCRIPT)
campaign = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(campaign)


def test_campaign_has_six_studies_and_56_logical_fronts() -> None:
    """AE/VAE add OAS views without duplicating trained trial pools."""
    studies = campaign._studies()
    assert len(studies) == 6
    assert {study["model"] for study in studies} == set(campaign.MODELS)
    assert sum(len(study["objectives"]) for study in studies) == 56
    for study in studies:
        expected = 14 if study["model"] in {"ae", "vae"} else 7
        assert len(study["objectives"]) == expected
        assert len({objective["id"] for objective in study["objectives"]}) == expected
        assert [
            index for objective in study["objectives"] for index in objective["value_indices"]
        ] == list(range(2 * expected))


def test_shared_sweep_command_preserves_paper_budget_and_logging(tmp_path) -> None:
    """One model command trains 50 epochs and retains the configured MLflow logger."""
    command = campaign._base_train_command(tmp_path, "ae")
    joined = " ".join(command)
    assert "-m" in command
    assert "experiment=physics/ae_background_all" in command
    assert "hparams_search=ae_shared_optuna" in command
    assert f"trainer.max_epochs={campaign.SEARCH_EPOCHS}" in command
    assert "test=false" in command
    assert "sqlite:///" in joined
    assert "?timeout=600" in joined
    assert not any(value.startswith("logger=") for value in command)
    assert not any("selection_metric" in value for value in command)


def test_logical_objective_directions_match_the_metrics() -> None:
    """CAP is maximized; W1, drift, and native Q-double-prime are minimized."""
    objectives = campaign._logical_objectives("ae")
    for objective in objectives:
        expected = (
            ["maximize", "minimize"]
            if objective["metric"] == "cap"
            else [
                "minimize",
                "minimize",
            ]
        )
        assert objective["directions"] == expected


def test_shared_hpo_files_match_campaign_objective_order() -> None:
    """Optuna receives exactly the directions expected by offline front slicing."""
    root = Path(__file__).parents[1]
    for model in campaign.MODELS:
        config = yaml.safe_load(
            (root / "configs" / "hparams_search" / f"{model}_shared_optuna.yaml").read_text(
                encoding="utf-8"
            )
        )
        assert config["hydra"]["sweeper"]["direction"] == [
            direction
            for objective in campaign._logical_objectives(model)
            for direction in objective["directions"]
        ]


def test_mixed_direction_pareto_front_uses_only_requested_pair() -> None:
    """Offline fronts must ignore all other values from the shared objective vector."""
    trials = [
        SimpleNamespace(number=0, values=(0.8, 2.0, 999.0), params={}),
        SimpleNamespace(number=1, values=(0.7, 3.0, -999.0), params={}),
        SimpleNamespace(number=2, values=(0.9, 4.0, 0.0), params={}),
        SimpleNamespace(number=3, values=(0.8, 1.0, 0.0), params={}),
    ]
    front = campaign._pareto_front(trials, [0, 1], ["maximize", "minimize"])
    assert [trial.number for trial in front] == [2, 3]


def test_only_full_finite_objective_vectors_count_toward_600() -> None:
    """Missing-metric fallback values must be replaced, not treated as usable trials."""
    assert campaign._usable_trial(SimpleNamespace(values=(0.5, 1.0)), expected_values=2)
    assert not campaign._usable_trial(
        SimpleNamespace(values=(0.5, float("inf"))), expected_values=2
    )
    assert not campaign._usable_trial(SimpleNamespace(values=(0.5,)), expected_values=2)


def test_checkpoint_identity_matches_evaluator_normalization() -> None:
    """Aggregation must match stable dataset-aware and ordinary checkpoint names."""
    assert campaign._checkpoint_identity("ascore_operational__ds=normal__stable") == "normal"
    assert campaign._checkpoint_identity("cap_native_cdf_ema") == "cap_native_cdf_ema"
    assert campaign._checkpoint_identity("last") == "last"


def test_retraining_uses_score_aware_downstream_suite() -> None:
    """Only AE/VAE need the additional OAS threshold and efficiency callback."""
    assert campaign._retrain_suite("ae") == "physics_background_native_oas"
    assert campaign._retrain_suite("vae") == "physics_background_native_oas"
    assert campaign._retrain_suite("dsae") == "physics_background_native"


def test_pilot_parser_exercises_the_complete_model_suite(monkeypatch) -> None:
    """A pilot validates all pairings and scores together, not one isolated cell."""
    monkeypatch.setattr(
        "sys.argv",
        ["physics_background_pairing_campaign.py", "pilot", "--model", "vae"],
    )
    args = campaign.parse_args()
    assert args.model == "vae"


def test_pilot_reduces_only_inner_cap_optimization(monkeypatch, tmp_path) -> None:
    """The unified pilot checks every path without changing data or pairing tables."""
    commands = []
    monkeypatch.setattr(campaign, "_load_design", lambda root: {"code_commit": "a" * 40})
    monkeypatch.setattr(
        campaign.subprocess, "run", lambda command, **kwargs: commands.append(command)
    )
    artifact = tmp_path / "pilots" / "dsae" / "optimized_metric.json"
    artifact.parent.mkdir(parents=True)
    objectives = [objective["id"] for objective in campaign._logical_objectives("dsae")]
    artifact.write_text(
        campaign.json.dumps(
            {
                "schema_version": 2,
                "objective_order": objectives,
                "selections": {
                    name: {
                        "optimized_ckpt_name": "last",
                        "optimized_metric": [1.0, 1.0],
                    }
                    for name in objectives
                },
            }
        ),
        encoding="utf-8",
    )

    campaign.pilot(tmp_path, "dsae")

    assert "physics_selection.cap_metric_config.n_epochs=1" in commands[0]
    assert not any(value.startswith("data.max_val_batches=") for value in commands[0])
