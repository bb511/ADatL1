"""Tests for the frozen physics background-pairing campaign matrix."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts/physics_background_pairing_campaign.py"
SPEC = importlib.util.spec_from_file_location("physics_background_pairing_campaign", SCRIPT)
campaign = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(campaign)


def test_campaign_contains_exactly_the_36_primary_search_cells() -> None:
    """Six models must each have four CAP, one W1, and one drift cell."""
    cells = campaign._cells()
    assert len(cells) == 36
    assert len({cell["id"] for cell in cells}) == 36
    for model in campaign.MODELS:
        model_cells = [cell for cell in cells if cell["model"] == model]
        assert len(model_cells) == 6
        assert sum(cell["metric"] == "cap_mapping" for cell in model_cells) == 4
        assert sum(cell["metric"] == "wasserstein" for cell in model_cells) == 1
        assert sum(cell["metric"] == "drift" for cell in model_cells) == 1


def test_sweep_command_preserves_paper_budget_and_selection_contract(tmp_path) -> None:
    """Hydra sweep commands must preserve epochs, data pairing, and persistent storage."""
    cell = {
        "id": "ae__cap__jetclr",
        "model": "ae",
        "metric": "cap_mapping",
        "strategy": "jetclr",
    }
    command = campaign._base_train_command(tmp_path, cell)
    joined = " ".join(command)
    assert "-m" in command
    assert "experiment=physics/ae_background_pairing" in command
    assert "+selection_metric=cap_mapping" in command
    assert "physics_pairing.strategy=jetclr" in command
    assert f"trainer.max_epochs={campaign.SEARCH_EPOCHS}" in command
    assert "test=false" in command
    assert "sqlite:///" in joined


def test_checkpoint_identity_matches_evaluator_normalization() -> None:
    """Aggregation must match stable dataset-aware and ordinary checkpoint names."""
    assert campaign._checkpoint_identity("ascore_operational__ds=normal__stable") == "normal"
    assert campaign._checkpoint_identity("cap_ema_ZB_run396102_vs_ZB_run398183") == (
        "cap_ema_ZB_run396102_vs_ZB_run398183"
    )
    assert campaign._checkpoint_identity("last") == "last"
