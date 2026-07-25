from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts import cchamber_campaign


def test_cchamber_candidate_pools_are_deterministic_and_complete() -> None:
    for model in cchamber_campaign.MODELS:
        first = cchamber_campaign._sample_pool(model, 65)
        second = cchamber_campaign._sample_pool(model, 65)

        assert first == second
        assert len(first) == 65
        assert first[0]["baseline"] is True
        assert first[0]["params"] == cchamber_campaign.BASELINES[model]
        assert len({record["candidate_id"] for record in first}) == 65
        assert all(
            set(record["params"]) == set(cchamber_campaign.SPACES[model]) for record in first
        )


def test_cchamber_search_metrics_cover_the_paper_strategies() -> None:
    assert tuple(cchamber_campaign.METRICS) == cchamber_campaign.STRATEGIES
    assert cchamber_campaign.DATA_SEED not in cchamber_campaign.DEV_SEEDS
    assert set(cchamber_campaign.DEV_SEEDS).isdisjoint(cchamber_campaign.REPORTING_SEEDS)


def test_cchamber_selection_uses_one_winner_and_expands_reporting_seeds(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(cchamber_campaign, "_assert_campaign_revision", lambda _: None)
    campaign = {
        "n_candidates_per_model": 2,
        "development_seeds": list(cchamber_campaign.DEV_SEEDS),
        "reporting_seeds": list(cchamber_campaign.REPORTING_SEEDS),
        "pool_sha256": {model: f"{model}-pool" for model in cchamber_campaign.MODELS},
        "git_commit": "abc123",
    }
    (tmp_path / "campaign.json").write_text(json.dumps(campaign), encoding="utf-8")
    selection_dir = tmp_path / "selection"
    selection_dir.mkdir()
    rows = []
    for model in cchamber_campaign.MODELS:
        for strategy in cchamber_campaign.STRATEGIES:
            direction = cchamber_campaign.METRICS[strategy][1]
            for seed in cchamber_campaign.DEV_SEEDS:
                for candidate_id in ("000", "001"):
                    wins = candidate_id == "000"
                    value = float(wins if direction == "maximize" else not wins)
                    rows.append(
                        {
                            "model": model,
                            "strategy": strategy,
                            "seed": seed,
                            "candidate_id": candidate_id,
                            "value": value,
                            "params_json": json.dumps({"algorithm.optimizer.lr": 1e-3}),
                        }
                    )
    pd.DataFrame(rows).to_csv(selection_dir / "candidate_metrics.csv", index=False)

    cchamber_campaign.select_candidates(tmp_path)

    selected = pd.read_csv(selection_dir / "selected_trials.csv")
    retrain = json.loads((selection_dir / "retrain_manifest.json").read_text(encoding="utf-8"))
    assert len(selected) == len(cchamber_campaign.MODELS) * len(cchamber_campaign.STRATEGIES)
    assert set(selected["candidate_id"].astype(str).str.zfill(3)) == {"000"}
    assert len(retrain) == len(selected) * len(cchamber_campaign.REPORTING_SEEDS)
    assert {item["seed"] for item in retrain} == set(cchamber_campaign.REPORTING_SEEDS)


def test_final_value_contract_checks_identity_and_bounds(tmp_path) -> None:
    path = tmp_path / "values.csv"
    pd.DataFrame(
        {
            "checkpoint": ["chosen", "chosen"],
            "intervention": ["uniform_red_weak", "uniform_red_mid"],
            "metric": ["auprc", "auprc"],
            "value": [0.2, 0.8],
        }
    ).to_csv(path, index=False)

    values = cchamber_campaign._validate_final_values(
        path,
        checkpoint_stem="chosen",
        interventions=["uniform_red_weak", "uniform_red_mid"],
        metric="auprc",
    )
    assert np.allclose(values["value"], [0.2, 0.8])

    values.loc[0, "value"] = np.inf
    values.to_csv(path, index=False)
    with pytest.raises(ValueError, match="finite"):
        cchamber_campaign._validate_final_values(
            path,
            checkpoint_stem="chosen",
            interventions=["uniform_red_weak", "uniform_red_mid"],
            metric="auprc",
        )
