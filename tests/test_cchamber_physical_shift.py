from __future__ import annotations

import json
import subprocess  # nosec B404
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scripts import cchamber_physical_shift, cchamber_physical_shift_slurm

INTERVENTIONS = [
    "uniform_t_measure_mid",
    "uniform_t_process_mid",
]


def _write_json(path: Path, value: object) -> None:
    """Write one deterministic JSON fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_readouts(path: Path, values: np.ndarray) -> None:
    """Write one synthetic Causal Chamber readout table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        values,
        columns=cchamber_physical_shift.READOUT_FEATURES,
    )
    frame.insert(0, "control_a", np.arange(len(frame), dtype=float))
    frame.insert(1, "control_b", values[:, 0] - values[:, 1])
    frame.to_csv(path, index=False)


def _artifact_record(path: Path) -> dict[str, object]:
    """Build a campaign-compatible artifact fingerprint."""
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": cchamber_physical_shift._sha256(path),
    }


def test_degenerate_effect_bootstrap_reports_undefined_interval() -> None:
    """Finite effect point estimates survive a degenerate descriptive bootstrap."""
    values = np.array([np.nan] * 11 + [0.5] * 39)
    low, high, finite, undefined = cchamber_physical_shift._optional_percentile_interval(
        values, 50
    )
    assert (low, high, finite, undefined) == (None, None, 39, True)

    values[10] = 0.5
    low, high, finite, undefined = cchamber_physical_shift._optional_percentile_interval(
        values, 50
    )
    assert low == pytest.approx(0.5)
    assert high == pytest.approx(0.5)
    assert finite == 40
    assert undefined is False


def _synthetic_bundle(tmp_path: Path) -> dict[str, Path | np.ndarray]:
    """Build a frozen synthetic campaign, design, selection, and dataset."""
    campaign_root = tmp_path / "campaign"
    dataset_dir = tmp_path / "data" / "lt_interventions_standard_v1"
    rng = np.random.default_rng(77)
    reference = rng.normal(size=(100, len(cchamber_physical_shift.READOUT_FEATURES)))
    process = rng.normal(size=(25, len(cchamber_physical_shift.READOUT_FEATURES)))
    process[:, 0] += 2.0
    measurement = rng.normal(size=(25, len(cchamber_physical_shift.READOUT_FEATURES)))
    measurement[:, 1] *= 2.5
    tables = {
        "uniform_reference": reference,
        "uniform_t_measure_mid": measurement,
        "uniform_t_process_mid": process,
    }
    for name, values in tables.items():
        _write_readouts(dataset_dir / f"{name}.csv", values)
    dataset_records = [_artifact_record(path) for path in sorted(dataset_dir.glob("*.csv"))]
    dataset_tree_sha256 = cchamber_physical_shift.hashlib.sha256(
        cchamber_physical_shift._canonical_json(dataset_records).encode("utf-8")
    ).hexdigest()

    campaign = {
        "schema_version": 1,
        "campaign_id": "synthetic-physical-shift",
        "git_commit": "synthetic-commit",
        "dataset": "lt_interventions_standard_v1",
        "dataset_archive_md5": "synthetic-md5",
        "dataset_files": dataset_records,
        "dataset_tree_sha256": dataset_tree_sha256,
        "feature_set": "readouts",
        "n_features": 11,
        "interventions": INTERVENTIONS,
        "models": ["ae"],
        "strategies": ["cap_random"],
        "development_seeds": [101],
        "reporting_seeds": [1001, 1002],
        "data_seed": 7,
        "pool_sha256": {"ae": "pool-ae"},
        "repository": str((tmp_path / "deployment").resolve()),
    }
    campaign_path = campaign_root / "campaign.json"
    _write_json(campaign_path, campaign)
    campaign_hash = cchamber_physical_shift._sha256(campaign_path)

    catalog = {
        "schema_version": 1,
        "campaign_id": campaign["campaign_id"],
        "campaign_manifest_sha256": campaign_hash,
        "frozen_before_intervention_outcomes": True,
        "protocol_source": {"archive_md5": campaign["dataset_archive_md5"]},
        "model_readouts": list(cchamber_physical_shift.READOUT_FEATURES),
        "targets": [
            {
                "target": "t_process",
                "physical_class": "process_or_actuator",
                "semantic_family": "process",
                "knob": "process_knob",
                "expected_readout_descendants": ["current"],
            },
            {
                "target": "t_measure",
                "physical_class": "measurement_chain",
                "semantic_family": "measurement",
                "knob": "measurement_knob",
                "expected_readout_descendants": ["angle_1"],
            },
        ],
    }
    catalog_path = campaign_root / "design" / cchamber_physical_shift_slurm.CATALOG_NAME
    _write_json(catalog_path, catalog)
    plan = {
        "schema_version": 1,
        "campaign_id": campaign["campaign_id"],
        "campaign_manifest_sha256": campaign_hash,
        "paper_analysis_plan_sha256": "not-used-by-physical-tool",
        "physical_catalog_sha256": cchamber_physical_shift._sha256(catalog_path),
        "frozen_before_intervention_outcomes": True,
        "data_contract": {
            "data_seed": campaign["data_seed"],
            "reference_experiment": "uniform_reference",
            "reference_split": "test.normal",
            "reference_count": 10,
            "signal_split": "test",
            "signal_count_per_intervention": 10,
            "signal_count": len(INTERVENTIONS),
            "feature_set": "readouts",
            "normalization": {
                "fit_split": "uniform_reference.train",
                "fit_count": 60,
                "method": "subtract per-feature median and divide by q95-q05",
                "clip": [-10.0, 10.0],
            },
        },
        "joint_shift": {
            "name": "biased_multivariate_energy_distance",
            "space": "synthetic eleven-readout space",
            "formula": cchamber_physical_shift.ENERGY_FORMULA,
            "finite_sample_rule": "Clamp negative floating-point roundoff to zero.",
            "uncertainty": (
                "50 deterministic stratified bootstrap resamples within reference "
                "and intervention, seed 1234 plus intervention index"
            ),
            "use": "outcome-independent",
        },
        "readout_effects": {
            "location": cchamber_physical_shift.LOCATION_ESTIMAND,
            "scale": cchamber_physical_shift.SCALE_ESTIMAND,
            "zero_variance_rule": cchamber_physical_shift.ZERO_VARIANCE_RULE,
            "descendant_summary": cchamber_physical_shift.DESCENDANT_ESTIMAND,
            "use": "outcome-independent",
        },
        "performance_association": {
            "unit": "intervention",
            "methods": ["later Spearman join"],
            "prohibited": ["selection revision"],
        },
    }
    plan_path = campaign_root / "design" / cchamber_physical_shift_slurm.PLAN_NAME
    _write_json(plan_path, plan)
    freeze_path = campaign_root / "design" / cchamber_physical_shift_slurm.FREEZE_NAME
    _write_json(
        freeze_path,
        {
            "schema_version": 1,
            "campaign_id": campaign["campaign_id"],
            "campaign_git_commit": campaign["git_commit"],
            "intervention_outcomes_inspected_before_freeze": False,
            "files": {
                "campaign": {
                    "path": "../campaign.json",
                    "sha256": campaign_hash,
                },
                "physical_intervention_catalog": {
                    "path": catalog_path.name,
                    "sha256": cchamber_physical_shift._sha256(catalog_path),
                },
                "physical_shift_estimand": {
                    "path": plan_path.name,
                    "sha256": cchamber_physical_shift._sha256(plan_path),
                },
            },
        },
    )

    selection = campaign_root / "selection"
    selection.mkdir(parents=True)
    candidate_metrics = selection / "candidate_metrics.csv"
    pd.DataFrame(
        [
            {
                "model": "ae",
                "strategy": "cap_random",
                "candidate_id": "000",
                "value": 0.5,
            }
        ]
    ).to_csv(candidate_metrics, index=False)
    selected_trials = selection / "selected_trials.csv"
    pd.DataFrame(
        [
            {
                "model": "ae",
                "strategy": "cap_random",
                "candidate_id": "000",
                "pool_sha256": "pool-ae",
                "git_commit": campaign["git_commit"],
            }
        ]
    ).to_csv(selected_trials, index=False)
    retrain_manifest = selection / "retrain_manifest.json"
    _write_json(
        retrain_manifest,
        [
            {
                "model": "ae",
                "strategy": "cap_random",
                "candidate_id": "000",
                "seed": seed,
            }
            for seed in campaign["reporting_seeds"]
        ],
    )
    selection_provenance = selection / "selection_provenance.json"
    _write_json(
        selection_provenance,
        {
            "candidate_metrics": str(candidate_metrics.resolve()),
            "candidate_metrics_sha256": cchamber_physical_shift._sha256(candidate_metrics),
            "development_seeds": campaign["development_seeds"],
            "intervention_labels_used": False,
            "n_selected": 1,
            "n_retrains": 2,
            "selected_trials_sha256": cchamber_physical_shift._sha256(selected_trials),
            "retrain_manifest_sha256": cchamber_physical_shift._sha256(retrain_manifest),
        },
    )
    return {
        "campaign_root": campaign_root,
        "campaign": campaign_path,
        "plan": plan_path,
        "catalog": catalog_path,
        "selection": selection_provenance,
        "selection_hash": cchamber_physical_shift._sha256(selection_provenance),
        "freeze": freeze_path,
        "freeze_hash": cchamber_physical_shift._sha256(freeze_path),
        "deployment": tmp_path / "deployment",
        "slurm_script": tmp_path / "slurm" / "physical-shift.sbatch",
        "slurm_logs": tmp_path / "slurm" / "logs",
        "scratch_root": tmp_path,
        "output": tmp_path / "physical-output",
        "output_second": tmp_path / "physical-output-second",
        "reference": reference,
    }


def _run(bundle: dict[str, Path | np.ndarray], output_key: str = "output") -> list[Path]:
    """Run the synthetic physical-shift tool."""
    return cchamber_physical_shift.analyze(
        bundle["campaign_root"],
        bundle["plan"],
        bundle["catalog"],
        bundle["selection"],
        bundle["selection_hash"],
        bundle[output_key],
    )


def test_physical_shift_is_gated_deterministic_and_outcome_independent(
    tmp_path: Path,
) -> None:
    """The complete synthetic characterization is deterministic and ordered."""
    bundle = _synthetic_bundle(tmp_path)
    outputs = _run(bundle)
    _run(bundle, "output_second")
    output = bundle["output"]
    output_second = bundle["output_second"]

    magnitude = pd.read_csv(output / "physical_shift_magnitude.csv")
    repeated = pd.read_csv(output_second / "physical_shift_magnitude.csv")
    pd.testing.assert_frame_equal(magnitude, repeated)
    assert magnitude["intervention"].tolist() == [
        "uniform_t_process_mid",
        "uniform_t_measure_mid",
    ]
    assert magnitude["bootstrap_seed"].tolist() == [1235, 1234]
    assert (magnitude["biased_energy_distance"] >= 0.0).all()
    assert (magnitude["bootstrap_ci95_low"] <= magnitude["bootstrap_ci95_high"]).all()

    effects = pd.read_csv(output / "readout_effects.csv")
    assert len(effects) == len(INTERVENTIONS) * len(cchamber_physical_shift.READOUT_FEATURES)
    process = effects[effects["intervention"] == "uniform_t_process_mid"]
    assert process.loc[process["readout"] == "current", "expected_descendant"].item()
    assert not process.loc[process["readout"] == "angle_1", "expected_descendant"].item()
    assert process.loc[process["readout"] == "current", "hedges_g"].item() > 1.0
    descendants = pd.read_csv(output / "expected_descendant_summary.csv")
    assert set(descendants["n_expected_descendants"]) == {1}

    status = json.loads((output / "component_status.json").read_text())
    assert status["components"]["performance_association"]["status"] == "pending"
    provenance = json.loads((output / "physical_shift_provenance.json").read_text())
    assert provenance["inputs"]["intervention_labels_used"] is False
    assert provenance["estimands"]["bootstrap"]["repetitions"] == 50
    assert provenance["split_and_normalization_contract"]["reference_test_normal_count"] == 10

    reference = bundle["reference"]
    generator = torch.Generator().manual_seed(7)
    train = torch.randperm(len(reference), generator=generator)[:60]
    quantiles = torch.quantile(
        torch.as_tensor(reference[train.numpy()], dtype=torch.float32),
        torch.tensor([0.05, 0.5, 0.95]),
        dim=0,
    )
    reference_permutation = torch.randperm(
        len(reference),
        generator=torch.Generator().manual_seed(7),
    )
    reference_test = reference_permutation[80:]
    test_order = torch.randperm(
        len(reference_test),
        generator=torch.Generator().manual_seed(7 + len(reference_test)),
    )
    normal_pool = reference_test[test_order[:10]]
    pairing_pool = reference_test[test_order[10:20]]
    pairing_values = torch.as_tensor(
        np.column_stack(
            (
                np.arange(len(reference), dtype=float),
                reference[:, 0] - reference[:, 1],
            )
        ),
        dtype=torch.float32,
    )
    normal_pairing = pairing_values[normal_pool]
    reference_pairing = pairing_values[pairing_pool]
    combined_pairing = torch.cat((normal_pairing, reference_pairing), dim=0)
    pairing_center = combined_pairing.mean(dim=0)
    pairing_scale = combined_pairing.std(dim=0).clamp_min(1.0e-6)
    pairs = cchamber_physical_shift.one_to_one_nearest_pairs(
        (normal_pairing - pairing_center) / pairing_scale,
        (reference_pairing - pairing_center) / pairing_scale,
        k=None,
    )
    reference_normal = normal_pool[pairs.idx_1].numpy()
    signal_permutation = torch.randperm(
        25,
        generator=torch.Generator().manual_seed(7 + 25),
    )
    signal_test = signal_permutation[15:].numpy()
    split_contract = provenance["split_and_normalization_contract"]
    assert split_contract["reference_test_normal_indices_sha256"] == (
        cchamber_physical_shift._hash_indices(reference_normal)
    )
    assert set(split_contract["signal_test_indices_sha256"].values()) == {
        cchamber_physical_shift._hash_indices(signal_test)
    }
    assert provenance["split_and_normalization_contract"]["normalizer"]["center"] == pytest.approx(
        quantiles[1].tolist()
    )
    assert provenance["split_and_normalization_contract"]["normalizer"]["scale"] == pytest.approx(
        (quantiles[2] - quantiles[0]).tolist()
    )
    assert provenance["split_and_normalization_contract"]["normalizer"]["clip_value"] == 10.0
    assert (output / "ordered_readout_effects.png").stat().st_size > 0
    assert (output / "ordered_shift_magnitude.png").stat().st_size > 0
    assert {path.name for path in outputs} >= {
        "physical_shift_magnitude.csv",
        "readout_effects.csv",
        "expected_descendant_summary.csv",
        "physical_shift_provenance.json",
    }


def test_selection_hash_mismatch_fails_before_output(tmp_path: Path) -> None:
    """An unpinned selection provenance is rejected before output creation."""
    bundle = _synthetic_bundle(tmp_path)
    bundle["selection_hash"] = "0" * 64
    with pytest.raises(ValueError, match="Selection provenance SHA-256 mismatch"):
        _run(bundle)
    assert not bundle["output"].exists()


def test_incomplete_frozen_selection_fails_before_output(tmp_path: Path) -> None:
    """Hash-consistent but incomplete selection artifacts are rejected."""
    bundle = _synthetic_bundle(tmp_path)
    retrain_path = bundle["campaign_root"] / "selection" / "retrain_manifest.json"
    retrain = json.loads(retrain_path.read_text())
    _write_json(retrain_path, retrain[:-1])
    selection = json.loads(bundle["selection"].read_text())
    selection["retrain_manifest_sha256"] = cchamber_physical_shift._sha256(retrain_path)
    selection["n_retrains"] = 1
    _write_json(bundle["selection"], selection)
    bundle["selection_hash"] = cchamber_physical_shift._sha256(bundle["selection"])

    with pytest.raises(ValueError, match="exact reporting coverage"):
        _run(bundle)
    assert not bundle["output"].exists()


def test_non_label_free_selection_fails_before_dataset_fingerprints(
    tmp_path: Path,
) -> None:
    """Selection gating precedes any intervention data access."""
    bundle = _synthetic_bundle(tmp_path)
    selection = json.loads(bundle["selection"].read_text())
    selection["intervention_labels_used"] = True
    _write_json(bundle["selection"], selection)
    bundle["selection_hash"] = cchamber_physical_shift._sha256(bundle["selection"])
    campaign = json.loads(bundle["campaign"].read_text())
    intervention_path = Path(campaign["dataset_files"][0]["path"])
    intervention_path.write_text("unreadable after failed selection gate\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not certify label-free selection"):
        _run(bundle)
    assert not bundle["output"].exists()


def test_catalog_hash_and_campaign_identity_are_frozen(tmp_path: Path) -> None:
    """A changed physical target catalog is rejected by the frozen plan hash."""
    bundle = _synthetic_bundle(tmp_path)
    catalog = json.loads(bundle["catalog"].read_text())
    catalog["targets"][0]["expected_readout_descendants"].append("v_reg")
    _write_json(bundle["catalog"], catalog)
    with pytest.raises(ValueError, match="catalog hash"):
        _run(bundle)
    assert not bundle["output"].exists()

    identity_bundle = _synthetic_bundle(tmp_path / "identity")
    campaign = json.loads(identity_bundle["campaign"].read_text())
    campaign["campaign_id"] = "changed-campaign"
    _write_json(identity_bundle["campaign"], campaign)
    with pytest.raises(ValueError, match="Campaign identity differs"):
        _run(identity_bundle)
    assert not identity_bundle["output"].exists()


def test_refuses_campaign_internal_output_and_estimators_match_definitions(
    tmp_path: Path,
) -> None:
    """Writes stay external and core estimators match direct definitions."""
    bundle = _synthetic_bundle(tmp_path)
    bundle["output"] = bundle["campaign_root"] / "physical"
    with pytest.raises(ValueError, match="outside the immutable campaign root"):
        _run(bundle)

    reference = np.array([[0.0], [1.0], [2.0]])
    intervention = np.array([[2.0], [3.0], [4.0]])
    rr, ii, ri = cchamber_physical_shift._distance_matrices(
        reference,
        intervention,
    )
    expected_energy = max(
        0.0,
        2.0 * ri.mean() - rr.mean() - ii.mean(),
    )
    assert cchamber_physical_shift._biased_energy_from_distances(
        rr,
        ii,
        ri,
    ) == pytest.approx(expected_energy)
    hedges, log_sd, identical, undefined = cchamber_physical_shift._readout_effects(
        reference,
        intervention,
    )
    correction = cchamber_physical_shift._hedges_correction(3, 3)
    assert hedges.item() == pytest.approx(correction * 2.0)
    assert log_sd.item() == pytest.approx(0.0)
    assert not identical.item()
    assert not undefined.item()

    (
        null_hedges,
        null_log_sd,
        null_identical,
        null_undefined,
    ) = cchamber_physical_shift._readout_effects(
        np.ones((3, 1)),
        np.ones((3, 1)),
    )
    assert np.isnan(null_hedges.item())
    assert np.isnan(null_log_sd.item())
    assert null_identical.item()
    assert not null_undefined.item()
    undefined_hedges, undefined_log_sd, _, undefined = cchamber_physical_shift._readout_effects(
        np.ones((3, 1)),
        np.array([[1.0], [2.0], [3.0]]),
    )
    assert np.isnan(undefined_hedges.item())
    assert np.isnan(undefined_log_sd.item())
    assert undefined.item()


def test_slurm_wrapper_is_pinned_cpu_only_and_shell_valid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generation authenticates light inputs and leaves all data work to Slurm."""
    bundle = _synthetic_bundle(tmp_path)
    bundle["deployment"].mkdir()
    monkeypatch.setattr(
        cchamber_physical_shift_slurm,
        "_require_campaign_deployment",
        lambda campaign: bundle["deployment"].resolve(),
    )
    monkeypatch.setattr(
        cchamber_physical_shift_slurm,
        "_require_analysis_deployment",
        lambda repository, commit: (bundle["deployment"].resolve(), "a" * 64),
    )
    # A generator must not open or authenticate intervention outcomes on the login node.
    campaign = json.loads(bundle["campaign"].read_text())
    intervention = next(
        Path(record["path"])
        for record in campaign["dataset_files"]
        if Path(record["path"]).stem != "uniform_reference"
    )
    intervention.write_text("sealed outcome remains unread by generator\n", encoding="utf-8")

    script = cchamber_physical_shift_slurm.generate_slurm(
        campaign_root=bundle["campaign_root"],
        freeze_manifest_sha256=bundle["freeze_hash"],
        selection_provenance_sha256=bundle["selection_hash"],
        output_dir=bundle["output"],
        script_output=bundle["slurm_script"],
        slurm_log_dir=bundle["slurm_logs"],
        scratch_root=bundle["scratch_root"],
        analysis_repository=bundle["deployment"],
        analysis_commit="a" * 40,
        uv=Path(sys.executable),
    )

    text = script.read_text(encoding="utf-8")
    assert "#SBATCH --account=a0166" in text
    assert "#SBATCH --partition=normal" in text
    assert "#SBATCH --time=04:00:00" in text
    assert "#SBATCH --cpus-per-task=72" in text
    assert "#SBATCH --mem=120G" in text
    assert "#SBATCH --gpus" not in text
    assert "run --frozen --no-sync python scripts/cchamber_physical_shift.py" in text
    assert 'sha256sum "$FREEZE"' in text
    assert 'sha256sum "$SELECTION"' in text
    assert not bundle["output"].exists()
    subprocess.run(["bash", "-n", str(script)], check=True)  # nosec B603 B607


def test_slurm_wrapper_rejects_unfrozen_or_internal_execution_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No job script is emitted for changed design or campaign-internal outputs."""
    bundle = _synthetic_bundle(tmp_path)
    bundle["deployment"].mkdir()
    monkeypatch.setattr(
        cchamber_physical_shift_slurm,
        "_require_campaign_deployment",
        lambda campaign: bundle["deployment"].resolve(),
    )
    monkeypatch.setattr(
        cchamber_physical_shift_slurm,
        "_require_analysis_deployment",
        lambda repository, commit: (bundle["deployment"].resolve(), "a" * 64),
    )
    common = {
        "campaign_root": bundle["campaign_root"],
        "freeze_manifest_sha256": bundle["freeze_hash"],
        "selection_provenance_sha256": bundle["selection_hash"],
        "script_output": bundle["slurm_script"],
        "slurm_log_dir": bundle["slurm_logs"],
        "scratch_root": bundle["scratch_root"],
        "analysis_repository": bundle["deployment"],
        "analysis_commit": "a" * 40,
        "uv": Path(sys.executable),
    }
    changed = json.loads(bundle["plan"].read_text())
    changed["joint_shift"]["finite_sample_rule"] = "changed after freeze"
    _write_json(bundle["plan"], changed)
    with pytest.raises(ValueError, match="failed SHA-256 authentication"):
        cchamber_physical_shift_slurm.generate_slurm(
            output_dir=bundle["output"],
            **common,
        )
    assert not bundle["slurm_script"].exists()

    internal = _synthetic_bundle(tmp_path / "internal")
    internal["deployment"].mkdir()
    monkeypatch.setattr(
        cchamber_physical_shift_slurm,
        "_require_campaign_deployment",
        lambda campaign: internal["deployment"].resolve(),
    )
    monkeypatch.setattr(
        cchamber_physical_shift_slurm,
        "_require_analysis_deployment",
        lambda repository, commit: (internal["deployment"].resolve(), "a" * 64),
    )
    with pytest.raises(ValueError, match="outside the immutable campaign root"):
        cchamber_physical_shift_slurm.generate_slurm(
            campaign_root=internal["campaign_root"],
            freeze_manifest_sha256=internal["freeze_hash"],
            selection_provenance_sha256=internal["selection_hash"],
            output_dir=internal["campaign_root"] / "physical-output",
            script_output=internal["slurm_script"],
            slurm_log_dir=internal["slurm_logs"],
            scratch_root=internal["scratch_root"],
            analysis_repository=internal["deployment"],
            analysis_commit="a" * 40,
            uv=Path(sys.executable),
        )
    assert not internal["slurm_script"].exists()
