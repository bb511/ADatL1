"""Tests for the frozen physics background-pairing campaign matrix."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts/physics_background_pairing_campaign.py"
SPEC = importlib.util.spec_from_file_location("physics_background_pairing_campaign", SCRIPT)
campaign = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(campaign)


def test_campaign_contains_exactly_the_56_score_stratified_search_cells() -> None:
    """AE/VAE have native and OAS variants; other models retain native scores."""
    cells = campaign._cells()
    assert len(cells) == 56
    assert len({cell["id"] for cell in cells}) == 56
    for model in campaign.MODELS:
        model_cells = [cell for cell in cells if cell["model"] == model]
        score_count = len(campaign.MODEL_SCORES[model])
        assert len(model_cells) == 7 * score_count
        assert sum(cell["metric"] == "cap_mapping" for cell in model_cells) == 5 * score_count
        assert sum(cell["metric"] == "wasserstein" for cell in model_cells) == score_count
        assert sum(cell["metric"] == "drift" for cell in model_cells) == score_count


def test_sweep_command_preserves_paper_budget_and_selection_contract(tmp_path) -> None:
    """Hydra sweep commands must preserve epochs, data pairing, and persistent storage."""
    cell = {
        "id": "ae__residual_oas__cap__jetclr",
        "model": "ae",
        "score": "residual_oas",
        "metric": "cap_mapping",
        "strategy": "jetclr",
    }
    command = campaign._base_train_command(tmp_path, cell)
    joined = " ".join(command)
    assert "-m" in command
    assert "experiment=physics/ae_background_pairing" in command
    assert "+selection_metric=cap_mapping" in command
    assert "physics_pairing.strategy=jetclr" in command
    assert "algorithm.anomaly_score=residual_oas" in command
    assert f"trainer.max_epochs={campaign.SEARCH_EPOCHS}" in command
    assert "test=false" in command
    assert "sqlite:///" in joined
    assert "?timeout=600" in joined


def test_cdf_control_uses_no_precomputed_pair_table(tmp_path) -> None:
    """CDF is a score-rank control on the same domains, not a scratch-table lookup."""
    cell = {
        "id": "vae__kl_raw__cap__cdf",
        "model": "vae",
        "score": "kl_raw",
        "metric": "cap_mapping",
        "strategy": "cdf",
    }
    command = campaign._base_train_command(tmp_path, cell)
    assert "pairing=physics_cdf" in command
    assert not any(item.startswith("physics_pairing.strategy=") for item in command)


def test_cdf_control_is_preserved_for_pareto_retraining() -> None:
    """Search and retraining must select the same CAP pairing implementation."""
    row = {
        "model": "ae",
        "score": "mse",
        "metric": "cap_mapping",
        "strategy": "cdf",
    }
    assert campaign._pairing_overrides(row) == ["pairing=physics_cdf"]


def test_checkpoint_identity_matches_evaluator_normalization() -> None:
    """Aggregation must match stable dataset-aware and ordinary checkpoint names."""
    assert campaign._checkpoint_identity("ascore_operational__ds=normal__stable") == "normal"
    assert campaign._checkpoint_identity("cap_ema_ZB_run396102_vs_ZB_run398183") == (
        "cap_ema_ZB_run396102_vs_ZB_run398183"
    )
    assert campaign._checkpoint_identity("last") == "last"


def test_native_and_oas_score_overrides_are_explicit() -> None:
    """Native branches skip OAS fitting while OAS branches route the canonical score."""
    assert campaign._score_overrides({"model": "ae", "score": "mse"}) == [
        "algorithm.anomaly_score=mse",
        "callbacks.residual_oas_state=null",
    ]
    assert campaign._score_overrides({"model": "ae", "score": "residual_oas"}) == [
        "algorithm.anomaly_score=residual_oas"
    ]
    assert campaign._score_overrides({"model": "vae", "score": "kl_raw"}) == [
        "algorithm.anomaly_score=kl_raw",
        "callbacks.vae_residual_state=null",
    ]
    assert campaign._score_overrides({"model": "vae", "score": "residual_oas"}) == [
        "algorithm.anomaly_score=residual_oas"
    ]


def test_pilot_parser_exposes_cdf_pairing_control(monkeypatch) -> None:
    """A real-data pilot must be able to exercise the table-free CDF path."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "physics_background_pairing_campaign.py",
            "pilot",
            "--metric",
            "cap_mapping",
            "--strategy",
            "cdf",
        ],
    )
    args = campaign.parse_args()
    assert args.strategy == "cdf"
