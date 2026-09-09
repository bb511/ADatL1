"""Shared lightweight leakage-probe fixtures."""

from src.evaluation.leakage_probe import (
    LeakageProbeRunMetadata,
    ProbeEvaluationContext,
    ProbeSplitProvenance,
)


def make_probe_evaluation_context() -> ProbeEvaluationContext:
    return ProbeEvaluationContext(
        mode="validation",
        development_data=ProbeSplitProvenance(
            split="train",
            source_splits=("train",),
            n_events=100,
            sample_seed=12345,
            max_samples=None,
            event_manifest_hash="train-manifest",
            data_cache_id="test-cache",
            data_cache_path="/test/cache",
        ),
        held_out_data=ProbeSplitProvenance(
            split="valid",
            source_splits=("valid",),
            n_events=40,
            sample_seed=12345,
            max_samples=None,
            event_manifest_hash="valid-manifest",
            data_cache_id="test-cache",
            data_cache_path="/test/cache",
        ),
    )


def make_probe_run_metadata() -> LeakageProbeRunMetadata:
    return LeakageProbeRunMetadata(
        autoencoder_seed=123,
        configuration_id="test-configuration",
    )
