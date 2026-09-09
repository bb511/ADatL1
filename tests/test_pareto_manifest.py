"""Phase 0 contract tests for the FET.Et validation Pareto study."""

from pathlib import Path

from hydra import compose, initialize
from omegaconf import OmegaConf

from src.utils.omegaconf import register_resolvers
from src.utils.pareto_manifest import write_resolved_pareto_manifest


def test_pareto_fet_manifest_composes_and_freezes_protocol(
    monkeypatch,
) -> None:
    """The dedicated experiment must expose every Phase 0 decision."""
    monkeypatch.setenv("PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))
    if not OmegaConf.has_resolver("reverse"):
        register_resolvers()

    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["experiment=physics/pareto_fet"],
        )

    assert cfg.test is False
    assert cfg.evaluation.leakage_probes.mode == "validation"
    assert cfg.evaluation.leakage_probes.smoke_test.enabled is False
    assert cfg.evaluation.leakage_probes.smoke_test.max_events_per_split is None
    assert cfg.evaluation.evaluator.ckpts.last is False
    assert cfg.evaluation.evaluator.ckpts.loss_total is True
    assert cfg.evaluation.callbacks.latent_collapse.ckpts.loss_total is True
    assert cfg.evaluation.callbacks.latent_collapse.dataset == "normal"
    assert cfg.evaluation.callbacks.latent_collapse.evaluation_split == "val"
    assert cfg.evaluation.leakage_probes.enabled is True
    assert cfg.evaluation.callbacks.anomaly_efficiency.write_pareto_summary is True
    assert cfg.evaluation.callbacks.anomaly_efficiency.write_plots is False
    assert cfg.evaluation.callbacks.correlation_matrix.enabled is True
    assert cfg.evaluation.callbacks.correlation_matrix.write_details is False
    assert cfg.evaluation.callbacks.anomaly_auroc.ckpts.loss_total is True
    assert cfg.evaluation.callbacks.anomaly_auroc.score_direction == (
        "higher_score_is_more_anomalous"
    )
    assert float(
        cfg.evaluation.callbacks.anomaly_auroc.max_false_positive_rate
    ) == 0.25 / 28608.8064
    assert cfg.callbacks.loss_total_ckpt.monitor == "val/loss_total"
    assert cfg.callbacks.loss_total_ckpt.mode == "min"
    assert cfg.data.model_input_exclude_features == ["FET.Et"]
    assert cfg.algorithm.encoder.nodes[-1] == 8
    assert cfg.pareto_study.paired_autoencoder_seeds == [123, 456, 789]
    assert cfg.pareto_study.search_space.gamma_zero_baseline.mi_gamma == 0.0
    assert (
        cfg.pareto_study.collapse_constraint.seed_level_rule.minimum_joint_code_entropy_bits
        == 1.0
    )
    assert (
        cfg.pareto_study.collapse_constraint.seed_level_rule.minimum_fraction_of_paired_gamma_zero_joint_entropy
        == 0.5
    )
    assert cfg.pareto_study.minimum_efficiency_constraint.max_relative_degradation == 0.05
    assert cfg.pareto_study.objectives.residual_correlation.role == "pareto_objective"
    assert OmegaConf.select(cfg, "pareto_study.partial_auroc.normalization") == (
        "raw_partial_auc / max_false_positive_rate"
    )


def test_pareto_manifest_is_full_and_resolved(tmp_path: Path) -> None:
    """A Pareto run writes the explicit study policy beside its other outputs."""
    cfg = OmegaConf.create(
        {
            "paths": {"output_dir": str(tmp_path)},
            "algorithm": {"target_rate": 0.25, "base_rate": 28608.8064},
            "pareto_study": {
                "enabled": True,
                "configuration_id": "fet-et-pareto-v1__gamma-0.0",
                "resolved_manifest": {
                    "filename": "pareto_manifest.resolved.yaml",
                },
            },
        }
    )

    output_path = write_resolved_pareto_manifest(cfg)

    assert output_path == tmp_path / "pareto_manifest.resolved.yaml"
    saved = OmegaConf.load(output_path)
    assert saved.pareto_study.configuration_id == "fet-et-pareto-v1__gamma-0.0"
    assert saved.paths.output_dir == str(tmp_path)
