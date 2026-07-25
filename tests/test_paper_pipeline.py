import json

import pandas as pd
import pytest

from scripts import paper_pipeline


def _candidate_rows():
    rows = []
    values = {
        "cap_metadata_nearest": {"001": 0.2, "002": 0.9},
        "cap_encoder_nearest": {"001": 0.8, "002": 0.4},
        "cap_random": {"001": 0.3, "002": 0.2},
        "drift": {"001": 0.5, "002": 0.1},
        "wasserstein": {"001": 0.2, "002": 0.4},
    }
    parameters = {
        "001": {"algorithm.optimizer.lr": 0.001, "algorithm.encoder.nodes": [16, 4]},
        "002": {"algorithm.optimizer.lr": 0.002, "algorithm.encoder.nodes": [32, 8]},
    }
    for strategy, candidates in values.items():
        for candidate_id, value in candidates.items():
            rows.append(
                {
                    "dataset": "cchamber",
                    "model": "ae",
                    "seed": 11,
                    "candidate_id": candidate_id,
                    "strategy": strategy,
                    "value": value,
                    "params_json": json.dumps(parameters[candidate_id]),
                }
            )
    return rows


def test_select_trials_uses_shared_label_free_pool(tmp_path) -> None:
    input_path = tmp_path / "candidate_metrics.csv"
    pd.DataFrame(_candidate_rows()).to_csv(input_path, index=False)

    written = paper_pipeline.select_trials(input_path, tmp_path / "selection")

    selected = pd.read_csv(
        tmp_path / "selection" / "selected_trials.csv",
        dtype={"candidate_id": str},
    )
    winners = dict(zip(selected["strategy"], selected["candidate_id"]))
    assert winners == {
        "cap_encoder_nearest": "001",
        "cap_metadata_nearest": "002",
        "cap_random": "001",
        "drift": "002",
        "wasserstein": "001",
    }
    assert len(written) == 4

    retrain = json.loads(
        (tmp_path / "selection" / "retrain_manifest.json").read_text(encoding="utf-8")
    )
    assert {item["spec_name"] for item in retrain} == {
        "cchamber_ae_cap_metadata_nearest",
        "cchamber_ae_cap_encoder_nearest",
        "cchamber_ae_cap_random",
        "cchamber_ae_drift",
        "cchamber_ae_wasserstein",
    }
    encoder = next(
        item for item in retrain if item["spec_name"] == "cchamber_ae_cap_encoder_nearest"
    )
    assert "algorithm.encoder.nodes=[16,4]" in encoder["overrides"]


def test_select_trials_rejects_oracle_and_non_shared_pool(tmp_path) -> None:
    oracle = pd.DataFrame(_candidate_rows())
    oracle.loc[0, "strategy"] = "auprc"
    oracle_path = tmp_path / "oracle.csv"
    oracle.to_csv(oracle_path, index=False)
    with pytest.raises(ValueError, match="label-free"):
        paper_pipeline.select_trials(oracle_path, tmp_path / "oracle-output")

    incomplete = pd.DataFrame(_candidate_rows()).iloc[1:]
    incomplete_path = tmp_path / "incomplete.csv"
    incomplete.to_csv(incomplete_path, index=False)
    with pytest.raises(ValueError, match="not replayed"):
        paper_pipeline.select_trials(incomplete_path, tmp_path / "incomplete-output")


def test_create_checkpoint_manifest_resolves_retrained_runs(tmp_path) -> None:
    selected = pd.DataFrame(
        [
            {
                "spec_name": "cchamber_ae_cap_random",
                "run_name": "cap_random_candidate_001_seed_11",
                "seed": 11,
                "dataset": "cchamber",
                "strategy": "cap_random",
            }
        ]
    )
    selected_path = tmp_path / "selected.csv"
    selected.to_csv(selected_path, index=False)
    checkpoint = (
        tmp_path
        / "checkpoints"
        / "cchamber_ae_cap_random_retrain"
        / "cap_random_candidate_001_seed_11"
        / "summary"
        / "cap_ema_normal_vs_reference_normal"
        / "max"
        / "cap_ema_normal_vs_reference_normal.ckpt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    output = paper_pipeline.create_checkpoint_manifest(
        selected_path,
        tmp_path / "checkpoints",
        tmp_path / "checkpoint_manifest.json",
    )

    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest[0]["spec_name"] == "cchamber_ae_cap_random"
    assert manifest[0]["ckpt_path"] == str(checkpoint.resolve())


def test_collect_results_annotates_raw_callback_values(tmp_path) -> None:
    first = tmp_path / "run-a" / "values.csv"
    second = tmp_path / "run-b" / "values.csv"
    for path, offset in ((first, 0.0), (second, 0.1)):
        path.parent.mkdir()
        pd.DataFrame(
            [
                {
                    "checkpoint": "cap.ckpt",
                    "intervention": "red_low",
                    "metric": "auprc",
                    "value": 0.6 + offset,
                },
                {
                    "checkpoint": "cap.ckpt",
                    "intervention": "blue_high",
                    "metric": "auprc",
                    "value": 0.8 + offset,
                },
            ]
        ).to_csv(path, index=False)
    manifest = tmp_path / "collect.csv"
    pd.DataFrame(
        [
            {
                "path": "run-a/values.csv",
                "dataset": "synthetic",
                "model": "ae",
                "strategy": "cap_random",
                "seed": 1,
                "pairing": "random",
            },
            {
                "path": "run-b/values.csv",
                "dataset": "synthetic",
                "model": "ae",
                "strategy": "cap_random",
                "seed": 2,
                "pairing": "random",
            },
        ]
    ).to_csv(manifest, index=False)

    written = paper_pipeline.collect_results(manifest, tmp_path / "results.csv")

    combined = pd.read_csv(tmp_path / "results.csv")
    assert len(combined) == 4
    assert set(combined["seed"]) == {1, 2}
    assert set(combined["pairing"]) == {"random"}
    assert (tmp_path / "results.provenance.json").is_file()
    assert len(written) == 2


def _result_rows():
    rows = []
    for strategy, offset in (("cap_random", 0.0), ("cap_encoder_nearest", 0.1)):
        for seed, seed_offset in ((1, 0.0), (2, 0.2), (3, 0.1)):
            for intervention, base, family, strength in (
                ("red_low", 0.4, "red", "weak"),
                ("red_high", 0.8, "red", "strong"),
                ("blue_mid", 0.6, "blue", "mid"),
            ):
                rows.append(
                    {
                        "dataset": "synthetic",
                        "model": "ae",
                        "strategy": strategy,
                        "pairing": strategy.removeprefix("cap_"),
                        "seed": seed,
                        "intervention": intervention,
                        "intervention_family": family,
                        "strength": strength,
                        "metric": "auprc",
                        "value": base + offset + seed_offset,
                    }
                )
    return rows


def test_aggregate_results_creates_seed_level_statistics_and_report(tmp_path) -> None:
    results_path = tmp_path / "results.csv"
    pd.DataFrame(_result_rows()).to_csv(results_path, index=False)

    written = paper_pipeline.aggregate_results(
        results_path,
        tmp_path / "paper",
        main_metric="auprc",
    )

    summary = pd.read_csv(tmp_path / "paper" / "summary.csv")
    encoder = summary[summary["strategy"] == "cap_encoder_nearest"].iloc[0]
    assert encoder["mean"] == pytest.approx(0.8)
    assert encoder["n_seeds"] == 3
    assert encoder["ci95_low"] < encoder["mean"] < encoder["ci95_high"]

    pairwise = pd.read_csv(tmp_path / "paper" / "paired_strategy_differences.csv")
    assert abs(pairwise.iloc[0]["mean_difference"]) == pytest.approx(0.1)
    assert (tmp_path / "paper" / "synthetic_auprc_comparison.png").is_file()
    assert (tmp_path / "paper" / "synthetic_auprc_interventions.png").is_file()
    report = (tmp_path / "paper" / "report.md").read_text(encoding="utf-8")
    assert "uncertainty across seeds" in report
    assert len(written) == 9


def test_aggregate_results_requires_paired_intervention_coverage(tmp_path) -> None:
    frame = pd.DataFrame(_result_rows())
    frame = frame.drop(frame.index[-1])
    results_path = tmp_path / "unpaired.csv"
    frame.to_csv(results_path, index=False)

    with pytest.raises(ValueError, match="identical intervention coverage"):
        paper_pipeline.aggregate_results(results_path, tmp_path / "paper")
